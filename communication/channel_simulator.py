"""Discrete-event simulation of a constrained wireless link.

Model, per packet:

* the segment bitstream is split into fragments of at most ``mtu_bytes``;
  each packet adds ``header_bytes`` (sequence number, segment id, fragment
  index/count, CRC);
* packets queue FIFO in a transmitter buffer and are serialised at the
  current ``bandwidth_bps``;
* while the link is down (outage) packets stay in the buffer and are sent
  after reconnection; if the buffer is full, new packets are dropped
  (tail drop) and counted as ``overflow``;
* each transmission attempt is lost with the loss model's probability; with
  ``max_retries > 0`` a lost packet is retransmitted (stop-and-wait ARQ,
  costing one RTT) until it succeeds or retries run out;
* delivered packets arrive after ``base_latency + |N(0, jitter)|``.

The simulator is incremental: call :meth:`send_segment` once per encoded
segment so the adaptive controller can read the link state in between.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from communication.packet_loss import OutageSchedule, make_loss_model
from config import ChannelConfig


@dataclass
class Packet:
    seq: int
    segment_id: int
    frag_idx: int
    n_frags: int
    payload: bytes
    t_created: float
    priority: bool = False

    def wire_bytes(self, header_bytes: int) -> int:
        return len(self.payload) + header_bytes


@dataclass
class PacketResult:
    packet: Packet
    delivered: bool
    attempts: int
    t_start: float | None  # first transmission start (None if dropped in buffer)
    t_arrival: float | None
    dropped_overflow: bool = False
    waited_for_outage: bool = False


@dataclass
class ChannelStats:
    packets: int = 0
    delivered: int = 0
    lost: int = 0
    overflow: int = 0
    attempts: int = 0
    bytes_on_air: int = 0  # every attempt, headers included
    payload_bytes_offered: int = 0
    buffered_by_outage: int = 0

    @property
    def loss_rate(self) -> float:
        return (self.lost + self.overflow) / self.packets if self.packets else 0.0


def packetize(blob: bytes, segment_id: int, mtu: int, t_created: float, seq0: int, priority: bool = False) -> list[Packet]:
    n = max(1, -(-len(blob) // mtu))
    return [
        Packet(seq0 + i, segment_id, i, n, blob[i * mtu : (i + 1) * mtu], t_created, priority)
        for i in range(n)
    ]


class ChannelSimulator:
    def __init__(self, cfg: ChannelConfig | None = None, bandwidth_schedule: list[tuple[float, float]] | None = None):
        self.cfg = cfg or ChannelConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.loss = make_loss_model(self.cfg.loss_rate, self.cfg.burst_length, self.rng)
        self.outages = OutageSchedule(self.cfg.outages)
        # Piecewise-constant bandwidth: [(start_s, bps), ...]
        self.bw_schedule = sorted(bandwidth_schedule or [(0.0, self.cfg.bandwidth_bps)])
        self.t_link = 0.0  # time at which the link becomes free
        self.seq = 0
        self.pending: deque[tuple[float, int]] = deque()  # (start time, bytes) of queued packets
        self.results: list[PacketResult] = []
        self.stats = ChannelStats()
        self._recent: deque[bool] = deque(maxlen=50)

    # -- link state (what the transmitter can observe) ---------------------- #
    def bandwidth_at(self, t: float) -> float:
        bw = self.bw_schedule[0][1]
        for start, bps in self.bw_schedule:
            if t >= start:
                bw = bps
        return bw

    def link_up(self, t: float) -> bool:
        return not self.outages.is_down(t)

    def backlog_bytes(self, t: float) -> int:
        while self.pending and self.pending[0][0] <= t:
            self.pending.popleft()
        return sum(b for _, b in self.pending)

    def recent_loss_rate(self) -> float:
        return float(np.mean(self._recent)) if self._recent else 0.0

    # -- transmission -------------------------------------------------------- #
    def _tx_time(self, nbytes: int, t: float) -> float:
        return nbytes * 8.0 / self.bandwidth_at(t)

    def send_segment(self, blob: bytes, segment_id: int, t_created: float, priority: bool = False) -> list[PacketResult]:
        cfg = self.cfg
        packets = packetize(blob, segment_id, cfg.mtu_bytes, t_created, self.seq, priority)
        self.seq += len(packets)
        out = []
        for p in packets:
            wire = p.wire_bytes(cfg.header_bytes)
            self.stats.packets += 1
            self.stats.payload_bytes_offered += len(p.payload)
            if self.backlog_bytes(t_created) + wire > cfg.buffer_bytes:
                self.stats.overflow += 1
                self._recent.append(True)
                res = PacketResult(p, False, 0, None, None, dropped_overflow=True)
                out.append(res)
                continue
            start = max(self.t_link, t_created)
            # Held back by a disconnection: created during it, or queued
            # behind packets that were.
            waited = self.outages.is_down(start) or self.outages.is_down(t_created)
            start = self.outages.next_up(start)
            if waited:
                self.stats.buffered_by_outage += 1
            first_start = start
            attempts, delivered, t_arr = 0, False, None
            while True:
                attempts += 1
                end = start + self._tx_time(wire, start)
                self.stats.bytes_on_air += wire
                lost = self.loss.lost() or self.outages.is_down(end)
                if not lost:
                    delivered = True
                    t_arr = end + cfg.base_latency_s + abs(self.rng.normal(0.0, cfg.jitter_s))
                    break
                if attempts > cfg.max_retries:
                    break
                # wait one RTT for the missing ACK, then retransmit
                start = self.outages.next_up(end + 2 * cfg.base_latency_s)
            self.t_link = end if delivered or cfg.max_retries == 0 else end + 2 * cfg.base_latency_s
            self.pending.append((self.t_link, wire))
            self.stats.attempts += attempts
            self.stats.delivered += int(delivered)
            self.stats.lost += int(not delivered)
            self._recent.append(not delivered)
            out.append(PacketResult(p, delivered, attempts, first_start, t_arr, waited_for_outage=waited))
        self.results.extend(out)
        return out
