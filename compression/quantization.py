"""Quantisation, bit packing and the PulseLink segment bitstream format.

A compressed segment is a self-describing byte string::

    +--------+-----+-------+------+----+----+---------+-------+----------------+---------+
    | "PL"   | ver | codec | flags| C  | T (uint16)   | qual  | C x (mean,std) | payload |
    | 2 B    | 1 B | 1 B   | 1 B  |1 B | 2 B          | 1 B   | float16 x 2C   | ...     |
    +--------+-----+-------+------+----+--------------+-------+----------------+---------+

``qual`` is the encoder-side PRD estimate in 0.5 % steps (255 = not
measured). ``flags`` carries the adaptive controller's decisions
(important-event, poor-signal-quality, payload-is-zlib) so the receiver
knows how much to trust each segment. All sizes reported by PulseLink are
``len()`` of these byte strings: nothing is estimated.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

MAGIC = b"PL"
VERSION = 1
_HEADER = struct.Struct("<2sBBBBHB")  # magic, ver, codec, flags, C, T, qual

FLAG_IMPORTANT = 0x01
FLAG_POOR_QUALITY = 0x02
FLAG_ZLIB = 0x04


# --------------------------------------------------------------------------- #
# Scalar quantisers
# --------------------------------------------------------------------------- #
def quantize_uniform(x: np.ndarray, bits: int, lo: float, hi: float) -> np.ndarray:
    """Map values in [lo, hi] to unsigned integers in [0, 2**bits - 1]."""
    levels = (1 << bits) - 1
    q = np.round((np.clip(x, lo, hi) - lo) / (hi - lo) * levels)
    return q.astype(np.uint32)


def dequantize_uniform(q: np.ndarray, bits: int, lo: float, hi: float) -> np.ndarray:
    levels = (1 << bits) - 1
    return q.astype(np.float64) / levels * (hi - lo) + lo


def quantize_symmetric(x: np.ndarray, bits: int):
    """Mid-tread symmetric quantiser. Returns (unsigned codes, scale)."""
    scale = float(np.max(np.abs(x))) if x.size else 1.0
    scale = scale if scale > 0 else 1.0
    half = (1 << (bits - 1)) - 1
    q = np.round(x / scale * half).astype(np.int64) + half
    return q.astype(np.uint32), scale


def dequantize_symmetric(q: np.ndarray, bits: int, scale: float) -> np.ndarray:
    half = (1 << (bits - 1)) - 1
    return (q.astype(np.int64) - half).astype(np.float64) / half * scale


# --------------------------------------------------------------------------- #
# Bit packing
# --------------------------------------------------------------------------- #
def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integers (< 2**bits) into a dense big-endian bitstream."""
    values = np.asarray(values, dtype=np.uint32).ravel()
    if bits < 1 or bits > 32:
        raise ValueError("bits must be in 1..32")
    if values.size and int(values.max()) >= (1 << bits):
        raise ValueError("value does not fit in the requested bit width")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bitmat = ((values[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bitmat.ravel()).tobytes()


def unpack_bits(data: bytes, bits: int, count: int) -> np.ndarray:
    raw = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: count * bits]
    bitmat = raw.reshape(count, bits).astype(np.uint32)
    weights = (1 << np.arange(bits - 1, -1, -1, dtype=np.uint64)).astype(np.uint32)
    return (bitmat * weights).sum(axis=1).astype(np.uint32)


def packed_size(count: int, bits: int) -> int:
    return (count * bits + 7) // 8


# --------------------------------------------------------------------------- #
# Segment container
# --------------------------------------------------------------------------- #
@dataclass
class SegmentHeader:
    codec_id: int
    n_channels: int
    n_samples: int
    mean: np.ndarray
    std: np.ndarray
    flags: int = 0
    quality: int = 255  # encoder-side PRD estimate in 0.5 % steps

    @property
    def prd_estimate(self) -> float | None:
        return None if self.quality == 255 else self.quality / 2.0

    @property
    def size(self) -> int:
        return _HEADER.size + 4 * self.n_channels


def encode_prd(prd: float | None) -> int:
    if prd is None or not np.isfinite(prd):
        return 255
    return int(min(254, max(0, round(prd * 2))))


def build_segment(header: SegmentHeader, payload: bytes, try_zlib: bool = True) -> bytes:
    """Serialise a header + payload. zlib is applied only if it helps."""
    flags = header.flags & ~FLAG_ZLIB
    if try_zlib:
        z = zlib.compress(payload, 9)
        if len(z) < len(payload):
            payload, flags = z, flags | FLAG_ZLIB
    head = _HEADER.pack(
        MAGIC, VERSION, header.codec_id, flags, header.n_channels, header.n_samples, header.quality
    )
    stats = np.empty(2 * header.n_channels, dtype=np.float16)
    stats[0::2] = header.mean
    stats[1::2] = header.std
    return head + stats.tobytes() + payload


def parse_segment(blob: bytes) -> tuple[SegmentHeader, bytes]:
    if len(blob) < _HEADER.size:
        raise ValueError("bitstream too short")
    magic, ver, codec_id, flags, c, t, qual = _HEADER.unpack_from(blob)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not a PulseLink v1 segment")
    off = _HEADER.size
    stats = np.frombuffer(blob[off : off + 4 * c], dtype=np.float16).astype(np.float64)
    payload = blob[off + 4 * c :]
    if flags & FLAG_ZLIB:
        payload = zlib.decompress(payload)
    header = SegmentHeader(codec_id, c, t, stats[0::2], stats[1::2], flags & ~FLAG_ZLIB, qual)
    return header, payload
