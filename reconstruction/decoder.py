"""Receiver: packet reassembly, decoding and loss concealment."""
from __future__ import annotations

import struct
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from communication.channel_simulator import PacketResult
from compression.codecs import CODEC_RAW, CodecRegistry, RawCodec
from compression.quantization import _HEADER, MAGIC, FLAG_ZLIB, SegmentHeader
from preprocessing.segmentation import denormalize_segment


@dataclass
class ReceivedSegment:
    segment_id: int
    signal: np.ndarray  # (C, T) reconstruction (concealed if lost)
    status: str  # "ok" | "partial" | "lost"
    header: SegmentHeader | None
    t_arrival: float | None
    latency_s: float | None
    late: bool
    decode_ops: int
    missing_fraction: float  # fraction of samples that had to be concealed
    flags: list[str] = field(default_factory=list)


class Receiver:
    """Reassembles fragments per segment and decodes complete segments.

    A compressed segment with any missing fragment cannot be decoded (the
    bitstream is entropy coded) and is concealed. Raw segments degrade
    gracefully: received samples are kept and gaps are interpolated, which
    is the fair comparison for uncompressed streaming.
    """

    def __init__(self, registry: CodecRegistry, n_channels: int, n_samples: int, playout_delay_s: float = 5.0, concealment: str = "zeros"):
        self.registry = registry
        self.c, self.t = n_channels, n_samples
        self.playout_delay_s = playout_delay_s
        self.concealment = concealment
        self.fragments: dict[int, dict[int, bytes]] = defaultdict(dict)
        self.n_frags: dict[int, int] = {}
        self.frag_size: dict[int, int] = {}
        self.arrival: dict[int, float] = {}
        self.created: dict[int, float] = {}
        self._last_good: np.ndarray | None = None

    def receive(self, results: list[PacketResult]) -> None:
        for r in results:
            p = r.packet
            self.n_frags[p.segment_id] = p.n_frags
            self.created[p.segment_id] = p.t_created
            if p.frag_idx == 0:
                self.frag_size[p.segment_id] = len(p.payload)
            if r.delivered:
                self.fragments[p.segment_id][p.frag_idx] = p.payload
                self.arrival[p.segment_id] = max(self.arrival.get(p.segment_id, 0.0), r.t_arrival)

    # ------------------------------------------------------------------ #
    def _conceal(self) -> np.ndarray:
        if self.concealment == "repeat" and self._last_good is not None:
            return self._last_good.copy()
        return np.zeros((self.c, self.t))

    def _partial_raw(self, sid: int) -> tuple[np.ndarray, SegmentHeader, float] | None:
        frags = self.fragments[sid]
        if 0 not in frags:
            return None  # no header -> no scale information
        first = frags[0]
        magic, ver, codec_id, flags, c, t, qual = _HEADER.unpack_from(first)
        if magic != MAGIC or codec_id != CODEC_RAW or flags & FLAG_ZLIB:
            return None
        hsize = _HEADER.size + 4 * c
        mtu = self.frag_size[sid]
        total = hsize + 2 * c * t
        buf = np.zeros(total, dtype=np.uint8)
        have = np.zeros(total, dtype=bool)
        for idx, payload in frags.items():
            buf[idx * mtu : idx * mtu + len(payload)] = np.frombuffer(payload, dtype=np.uint8)
            have[idx * mtu : idx * mtu + len(payload)] = True
        stats = np.frombuffer(buf[_HEADER.size : hsize].tobytes(), dtype=np.float16).astype(np.float64)
        header = SegmentHeader(codec_id, c, t, stats[0::2], stats[1::2], flags, qual)
        samples = np.frombuffer(buf[hsize:].tobytes(), dtype="<u2").reshape(c, t)
        ok = (have[hsize::2] & have[hsize + 1 :: 2]).reshape(c, t)
        rng = RawCodec.RANGE
        xn = samples.astype(np.float64) / 65535 * 2 * rng - rng
        for ch in range(c):
            if ok[ch].any() and not ok[ch].all():
                idx = np.arange(t)
                # interpolate interior gaps; gaps at the edges fall back to the mean
                xn[ch, ~ok[ch]] = np.interp(idx[~ok[ch]], idx[ok[ch]], xn[ch, ok[ch]], left=0.0, right=0.0)
            elif not ok[ch].any():
                xn[ch] = 0.0
        return denormalize_segment(xn, header.mean, header.std), header, float(1 - ok.mean())

    def reconstruct(self, segment_ids) -> list[ReceivedSegment]:
        out = []
        for sid in segment_ids:
            n = self.n_frags.get(sid)
            frags = self.fragments.get(sid, {})
            arrival = self.arrival.get(sid)
            created = self.created.get(sid)
            latency = None if arrival is None or created is None else arrival - created
            late = latency is not None and latency > self.playout_delay_s
            if n is not None and len(frags) == n:
                blob = b"".join(frags[i] for i in range(n))
                try:
                    sig, header = self.registry.decode(blob)
                    ops = self.registry.decode_ops(blob)
                    self._last_good = sig
                    out.append(ReceivedSegment(sid, sig, "ok", header, arrival, latency, late, ops, 0.0))
                    continue
                except Exception:  # noqa: BLE001 - corrupt bitstream -> conceal
                    pass
            partial = self._partial_raw(sid) if frags else None
            if partial is not None:
                sig, header, missing = partial
                out.append(ReceivedSegment(sid, sig, "partial", header, arrival, latency, late, 2 * self.c * self.t, missing))
            else:
                out.append(ReceivedSegment(sid, self._conceal(), "lost", None, arrival, None, False, 0, 1.0))
        return out
