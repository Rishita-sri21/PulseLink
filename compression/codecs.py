"""Codec interface shared by raw transmission, the DWT baseline and PulseLink.

Every codec turns a (C, T) segment in microvolts into a self-describing
byte string (see :mod:`compression.quantization`) and back. Codecs also
report their operation counts so the energy model can charge for compute.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from compression.autoencoder import ConvAutoencoder, count_macs
from compression.quantization import (
    SegmentHeader,
    build_segment,
    dequantize_uniform,
    encode_prd,
    pack_bits,
    packed_size,
    parse_segment,
    quantize_uniform,
    unpack_bits,
)
from compression.wavelet_compression import dwt_compress, dwt_decompress, dwt_ops
from preprocessing.segmentation import denormalize_segment, normalize_segment

CODEC_RAW = 0
CODEC_DWT = 1
CODEC_AE = 2

# Per-byte cost of zlib (deflate) on an MCU: a few tens of ops per byte.
ZLIB_OPS_PER_BYTE = 40
# Normalisation + quantisation: a handful of ops per sample.
QUANT_OPS_PER_SAMPLE = 6


def prd(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Percentage root-mean-square difference, mean-removed (PRDN)."""
    xc = x - x.mean(axis=-1, keepdims=True)
    return float(100.0 * np.sqrt(np.sum((x - x_hat) ** 2) / (np.sum(xc**2) + 1e-12)))


@dataclass
class EncodedSegment:
    blob: bytes
    codec_id: int
    mode: str
    encode_ops: int
    prd_estimate: float | None = None

    @property
    def nbytes(self) -> int:
        return len(self.blob)


class Codec:
    codec_id: int = -1
    name: str = "codec"

    def mode_name(self) -> str:
        return self.name

    def _payload(self, xn: np.ndarray) -> bytes:  # pragma: no cover - abstract
        raise NotImplementedError

    def _decode_payload(self, payload: bytes, c: int, t: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_ops(self, c: int, t: int) -> int:
        return QUANT_OPS_PER_SAMPLE * c * t

    def decode_ops(self, c: int, t: int) -> int:
        return QUANT_OPS_PER_SAMPLE * c * t

    def encode(self, x: np.ndarray, flags: int = 0, verify: bool = False, prd_hint: float | None = None) -> EncodedSegment:
        """Compress a (C, T) segment.

        ``prd_hint`` (e.g. from the adaptive controller's distortion
        predictor) is written into the header as the quality estimate. With
        ``verify=True`` the transmitter instead decodes its own output and
        writes the *measured* PRD (exact, but the decode is charged to the
        transmitter's compute budget).
        """
        c, t = x.shape
        xn, mean, std = normalize_segment(x)
        payload = self._payload(xn)
        header = SegmentHeader(self.codec_id, c, t, mean, std, flags, quality=encode_prd(prd_hint))
        blob = build_segment(header, payload)
        ops = self.encode_ops(c, t) + ZLIB_OPS_PER_BYTE * len(payload)
        est = prd_hint
        if verify:
            est = prd(x, self.decode(blob))
            header.quality = encode_prd(est)
            blob = build_segment(header, payload)
            ops += self.decode_ops(c, t)
        return EncodedSegment(blob, self.codec_id, self.mode_name(), ops, est)

    def decode(self, blob: bytes) -> np.ndarray:
        header, payload = parse_segment(blob)
        if header.codec_id != self.codec_id:
            raise ValueError(f"{self.name} cannot decode codec id {header.codec_id}")
        xn = self._decode_payload(payload, header.n_channels, header.n_samples)
        return denormalize_segment(xn, header.mean, header.std)


class RawCodec(Codec):
    """Uncompressed 16-bit transmission (the reference everything beats)."""

    codec_id = CODEC_RAW
    name = "raw"
    RANGE = 8.0  # +-8 std covers EEG; int16 step = 16/65535 std

    def _payload(self, xn):
        return quantize_uniform(xn, 16, -self.RANGE, self.RANGE).astype("<u2").tobytes()

    def _decode_payload(self, payload, c, t):
        q = np.frombuffer(payload, dtype="<u2")[: c * t].reshape(c, t)
        return dequantize_uniform(q, 16, -self.RANGE, self.RANGE)

    def encode(self, x, flags=0, verify=False, prd_hint=None):
        # Raw is sent without entropy coding, exactly like streaming samples.
        c, t = x.shape
        xn, mean, std = normalize_segment(x)
        header = SegmentHeader(self.codec_id, c, t, mean, std, flags, quality=0)
        blob = build_segment(header, self._payload(xn), try_zlib=False)
        return EncodedSegment(blob, self.codec_id, self.name, 2 * c * t, 0.0)


class WaveletCodec(Codec):
    codec_id = CODEC_DWT
    name = "dwt"

    def __init__(self, keep_ratio: float = 0.1, bits: int = 8, wavelet: str = "db4", level: int = 5):
        self.keep_ratio, self.bits, self.wavelet, self.level = keep_ratio, bits, wavelet, level

    def mode_name(self):
        return f"dwt(k={self.keep_ratio:g},b={self.bits})"

    def _payload(self, xn):
        return dwt_compress(xn, self.keep_ratio, self.bits, self.wavelet, self.level)

    def _decode_payload(self, payload, c, t):
        return dwt_decompress(payload, c, t)

    def encode_ops(self, c, t):
        # forward DWT + threshold selection (~log2 N compares per coeff) + quantise
        return dwt_ops(c, t, self.wavelet, self.level) + int(c * t * np.log2(c * t)) + QUANT_OPS_PER_SAMPLE * c * t

    def decode_ops(self, c, t):
        return dwt_ops(c, t, self.wavelet, self.level) + QUANT_OPS_PER_SAMPLE * c * t


class AutoencoderCodec(Codec):
    """PulseLink learned codec: encoder -> b-bit latent -> bitstream -> decoder."""

    codec_id = CODEC_AE
    name = "pulselink"

    def __init__(self, model: ConvAutoencoder, bits: int = 6, keep: int | None = None):
        self.model = model.eval()
        self.bits = bits
        self.keep = min(keep or model.cfg.latent_channels, model.cfg.latent_channels)
        self._macs: dict[tuple, int] = {}

    def mode_name(self):
        return f"pulselink(k={self.keep},b={self.bits})"

    def _payload(self, xn):
        with torch.no_grad():
            z = self.model.encode(torch.as_tensor(np.clip(xn, -8, 8), dtype=torch.float32)[None])[0].numpy()
        z = z[: self.keep]  # only the first k (most important) channels are sent
        return bytes([self.bits, self.keep]) + pack_bits(quantize_uniform(z, self.bits, -1.0, 1.0), self.bits)

    def _decode_payload(self, payload, c, t):
        bits, keep = payload[0], payload[1]
        lc, lt = self.model.cfg.latent_channels, self.model.latent_length(t)
        codes = unpack_bits(payload[2 : 2 + packed_size(keep * lt, bits)], bits, keep * lt)
        z = np.zeros((1, lc, lt))
        z[0, :keep] = dequantize_uniform(codes, bits, -1.0, 1.0).reshape(keep, lt)
        with torch.no_grad():
            return self.model.decode(torch.as_tensor(z, dtype=torch.float32), t)[0].numpy().astype(np.float64)

    def _count(self, c, t, part):
        key = (c, t, part)
        if key not in self._macs:
            self._macs[key] = count_macs(self.model, c, t, part)
        return self._macs[key]

    def encode_ops(self, c, t):
        return self._count(c, t, "encoder") + QUANT_OPS_PER_SAMPLE * c * t

    def decode_ops(self, c, t):
        return self._count(c, t, "decoder") + QUANT_OPS_PER_SAMPLE * c * t


class CodecRegistry:
    """Receiver-side dispatcher: picks the decoder from the header codec id."""

    def __init__(self, model: ConvAutoencoder | None = None):
        self.codecs: dict[int, Codec] = {CODEC_RAW: RawCodec(), CODEC_DWT: WaveletCodec()}
        if model is not None:
            self.codecs[CODEC_AE] = AutoencoderCodec(model)

    def decode(self, blob: bytes) -> tuple[np.ndarray, SegmentHeader]:
        header, _ = parse_segment(blob)
        codec = self.codecs.get(header.codec_id)
        if codec is None:
            raise ValueError(f"no decoder for codec id {header.codec_id}")
        return codec.decode(blob), header

    def decode_ops(self, blob: bytes) -> int:
        header, _ = parse_segment(blob)
        return self.codecs[header.codec_id].decode_ops(header.n_channels, header.n_samples)
