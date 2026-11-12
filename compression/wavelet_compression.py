"""Conventional DWT compression baseline.

Algorithm (per segment):

1. per-channel z-score (mean/std go in the header);
2. ``pywt.wavedec`` per channel (default db4, 5 levels);
3. keep the largest ``keep_ratio`` fraction of coefficients across all
   channels (hard thresholding);
4. quantise kept coefficients with a symmetric ``bits``-bit quantiser;
5. transmit a significance bitmap + packed codes (zlib if it helps).

Decoding inverts the steps with ``pywt.waverec``.
"""
from __future__ import annotations

import struct

import numpy as np
import pywt

from compression.quantization import (
    dequantize_symmetric,
    pack_bits,
    packed_size,
    quantize_symmetric,
    unpack_bits,
)

WAVELETS = ["db4", "db6", "sym4", "coif3", "bior4.4", "haar"]
_PAYLOAD_HEAD = struct.Struct("<BBBf")  # bits, level, wavelet index, scale


def _coeff_layout(n_samples: int, wavelet: str, level: int) -> list[int]:
    dummy = pywt.wavedec(np.zeros(n_samples), wavelet, level=level)
    return [c.size for c in dummy]


def dwt_compress(xn: np.ndarray, keep_ratio: float = 0.1, bits: int = 8, wavelet: str = "db4", level: int = 5) -> bytes:
    """Compress a normalised (C, T) segment into a DWT payload."""
    if wavelet not in WAVELETS:
        raise ValueError(f"unsupported wavelet {wavelet}")
    level = min(level, pywt.dwt_max_level(xn.shape[-1], pywt.Wavelet(wavelet).dec_len))
    coeffs = np.concatenate([np.concatenate(pywt.wavedec(ch, wavelet, level=level)) for ch in xn])
    k = max(1, int(round(keep_ratio * coeffs.size)))
    if k >= coeffs.size:
        mask = np.ones(coeffs.size, dtype=bool)
    else:
        thresh = np.partition(np.abs(coeffs), coeffs.size - k)[coeffs.size - k]
        mask = np.abs(coeffs) >= thresh
        # Ties can exceed k; trim deterministically.
        idx = np.flatnonzero(mask)
        if idx.size > k:
            order = np.argsort(-np.abs(coeffs[idx]), kind="stable")
            mask[:] = False
            mask[idx[order[:k]]] = True
    kept = coeffs[mask]
    codes, scale = quantize_symmetric(kept, bits)
    head = _PAYLOAD_HEAD.pack(bits, level, WAVELETS.index(wavelet), scale)
    return head + np.packbits(mask).tobytes() + pack_bits(codes, bits)


def dwt_decompress(payload: bytes, n_channels: int, n_samples: int) -> np.ndarray:
    bits, level, widx, scale = _PAYLOAD_HEAD.unpack_from(payload)
    wavelet = WAVELETS[widx]
    layout = _coeff_layout(n_samples, wavelet, level)
    per_ch = sum(layout)
    total = per_ch * n_channels
    off = _PAYLOAD_HEAD.size
    mask_bytes = (total + 7) // 8
    mask = np.unpackbits(np.frombuffer(payload[off : off + mask_bytes], dtype=np.uint8))[:total].astype(bool)
    off += mask_bytes
    n_kept = int(mask.sum())
    codes = unpack_bits(payload[off : off + packed_size(n_kept, bits)], bits, n_kept)
    coeffs = np.zeros(total)
    coeffs[mask] = dequantize_symmetric(codes, bits, scale)
    out = np.empty((n_channels, n_samples))
    splits = np.cumsum(layout)[:-1]
    for c in range(n_channels):
        parts = np.split(coeffs[c * per_ch : (c + 1) * per_ch], splits)
        out[c] = pywt.waverec(parts, wavelet)[:n_samples]
    return out


def dwt_ops(n_channels: int, n_samples: int, wavelet: str = "db4", level: int = 5) -> int:
    """Approximate MAC count of a forward (or inverse) DWT.

    Each level convolves the current approximation with a low- and a
    high-pass filter and downsamples by two, so the total work is about
    2 * filter_len * N * (1 + 1/2 + 1/4 + ...) < 4 * filter_len * N.
    """
    flen = pywt.Wavelet(wavelet).dec_len
    work = sum(2 * flen * n_samples / (2**i) for i in range(level))
    return int(n_channels * work)
