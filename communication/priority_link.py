"""Priority-aware wireless link: alerts never wait behind EEG data.

This is the transmission half of PulseLink's "fastest reliable alert" design.
Every message gets a priority class:

===== ======================== =============================================
class  what                     how it is sent
===== ======================== =============================================
P0     critical alert           first in line, may interrupt a data packet,
                                spaced copies, end-to-end ACK and retry,
                                backup radio if not ACKed in time
P1     warning / heartbeat      next in line, may interrupt data, ACK + retry
P2     event EEG                high-fidelity frames around an event,
                                pre-trigger buffer
P3     routine EEG              compressed background stream; dropped when
                                older than its deadline (stale data is useless)
P4     background               logs, statistics; first to be dropped
===== ======================== =============================================

``priority=False`` turns the link into a single FIFO queue (what a plain
"stream everything" wearable does), so the two can be compared on the same
loss and bandwidth.

The simulation is discrete-event and incremental: the engine calls
:meth:`PriorityLink.advance_to` with the current time, then
:meth:`PriorityLink.send` for new messages. A packet's fate is known only
once the link has simulated past it (a later alert may interrupt it), so
results are read back by message id with :meth:`PriorityLink.status`.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

from communication.packet_loss import OutageSchedule, make_time_loss_model
from config import ChannelConfig

P0, P1, P2, P3, P4 = range(5)
CLASS_NAMES = {P0: "P0 critical", P1: "P1 warning", P2: "P2 event EEG", P3: "P3 routine EEG", P4: "P4 background"}


@dataclass
class BackupRadioConfig:
    """Second, independent path for P0 alerts (e.g. LTE-M / NB-IoT / LoRa).

    Slow and power hungry, but it does not share the BLE link's outages.
    Numbers are order-of-magnitude assumptions, not measurements.
    """

    enabled: bool = True
    after_s: float = 0.5  # use it if the primary path has not ACKed by then
    latency_s: float = 0.8  # radio wake-up + network + push
    bandwidth_bps: float = 2_000.0
    loss_rate: float = 0.02
    retry_s: float = 1.0
    max_tries: int = 5
    energy_per_bit_j: float = 2e-6
    energy_per_packet_j: float = 20e-3  # modem wake-up dominates
    outages: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class LinkConfig:
    priority: bool = True  # False = one FIFO queue for everything
    preempt: bool = True  # P0/P1 may interrupt a P2-P4 packet in flight
    ack_classes: tuple[int, ...] = (P0, P1)  # end-to-end ACK + retry
    ack_retries: int = 8
    ack_timeout_s: float = 0.05  # wait beyond one round trip before retrying
    deadline_s: dict = field(default_factory=lambda: {P2: 60.0, P3: 20.0, P4: 120.0})
    backup: BackupRadioConfig = field(default_factory=BackupRadioConfig)


@dataclass
class _Pkt:
    seq: int
    cls: int
    msg: int
    nbytes: int  # payload
    t_created: float
    t_ready: float
    attempts: int = 0


@dataclass
class Message:
    id: int
    cls: int
    kind: str
    t_created: float
    n_packets: int
    any_copy: bool  # True: one delivered copy is enough (alerts); False: all fragments needed (frames)
    info: dict = field(default_factory=dict)
    delivered: int = 0
    failed: int = 0
    got: set = field(default_factory=set)  # packets delivered at least once
    t_first: float | None = None  # first packet arrival
    t_last: float | None = None  # last packet arrival
    t_ack: float | None = None  # sender learns of delivery (ACK classes)
    path: str | None = None  # "primary" / "backup"
    t_backup: float | None = None
    t_resolved: float | None = None  # sender-side time when the outcome became final
    dropped: str | None = None  # "stale", "overflow", "loss"

    @property
    def state(self) -> str:
        if self.t_backup is not None:
            return "delivered"
        if self.any_copy:
            if self.t_first is not None:
                return "delivered"
            return "lost" if self.failed >= self.n_packets and self.t_backup is None else "pending"
        if len(self.got) >= self.n_packets:
            return "delivered"
        return "lost" if self.failed else "pending"

    @property
    def t_arrival(self) -> float | None:
        if self.state != "delivered":
            return None
        return self.t_first if (self.any_copy or self.path == "backup") else self.t_last


@dataclass
class LinkStats:
    packets: int = 0
    attempts: int = 0
    delivered: int = 0
    lost: int = 0
    stale: int = 0
    overflow: int = 0
    preempted: int = 0
    cancelled: int = 0
    bytes_on_air: int = 0
    backup_packets: int = 0
    backup_bytes: int = 0
    by_class_bytes: dict = field(default_factory=lambda: {c: 0 for c in CLASS_NAMES})

    @property
    def loss_rate(self) -> float:
        n = self.delivered + self.lost + self.stale + self.overflow
        return (self.lost + self.stale + self.overflow) / n if n else 0.0


class PriorityLink:
    def __init__(self, channel: ChannelConfig | None = None, cfg: LinkConfig | None = None):
        self.ch = channel or ChannelConfig()
        self.cfg = cfg or LinkConfig()
        self.rng = np.random.default_rng(self.ch.seed)
        self.loss = make_time_loss_model(self.ch.loss_rate, self.ch.burst_length, self.rng)
        self.outages = OutageSchedule(self.ch.outages)
        self.bw_schedule = [(0.0, float(self.ch.bandwidth_bps))]
        self._brng = np.random.default_rng(self.ch.seed + 7919)
        self.queues: list[list] = [[] for _ in CLASS_NAMES]
        self._seq = itertools.count()
        self._mid = itertools.count()
        self.msgs: dict[int, Message] = {}
        self.cur: tuple[_Pkt, float, float] | None = None  # packet in flight: (pkt, start, end)
        self.t_free = 0.0
        self.now = 0.0
        self.queued_bytes = 0
        self.stats = LinkStats()
        self.last_rx = None  # last time the receiver heard anything (primary path)
        self._unacked: set[int] = set()  # P0 messages waiting for an ACK (backup candidates)

    # -- link state ---------------------------------------------------------- #
    def set_loss(self, loss_rate: float | None = None, burst_length: float | None = None) -> None:
        if loss_rate is not None:
            self.ch.loss_rate = loss_rate
        if burst_length is not None:
            self.ch.burst_length = burst_length
        self.loss = make_time_loss_model(self.ch.loss_rate, self.ch.burst_length, self.rng)

    def bandwidth_at(self, t: float) -> float:
        bw = self.bw_schedule[0][1]
        for s, b in self.bw_schedule:
            if t >= s:
                bw = b
        return bw

    def link_up(self, t: float) -> bool:
        return not self.outages.is_down(t)

    def backlog(self) -> dict[int, int]:
        """Bytes waiting per class (for the dashboard)."""
        out = {c: 0 for c in CLASS_NAMES}
        for q in self.queues:
            for item in q:
                p = item[-1]
                if p is not None:
                    out[p.cls] += p.nbytes + self.ch.header_bytes
        return out

    def _wire(self, p: _Pkt) -> int:
        return p.nbytes + self.ch.header_bytes

    # -- sending ------------------------------------------------------------- #
    def send(self, t: float, nbytes: int, cls: int, kind: str = "data", copies: int = 1, gap_s: float = 0.0, info: dict | None = None, backup_if_down: bool = False) -> int:
        """Queue a message at time ``t``; returns its id.

        ``copies > 1`` sends that many copies, ``gap_s`` apart (alerts:
        one delivered copy is enough). Otherwise the payload is split into
        MTU-sized fragments, all of which must arrive (EEG frames).
        P0 messages fall back to the backup radio if not ACKed in time;
        ``backup_if_down`` also sends a P1 message there if the primary
        link is down right now (e.g. an early warning during an outage).
        """
        if t < self.now - 1e-9:
            raise ValueError("messages must be sent in time order")
        mid = next(self._mid)
        mtu = self.ch.mtu_bytes
        if copies > 1:
            sizes, readies = [nbytes] * copies, [t + k * gap_s for k in range(copies)]
        else:
            n = max(1, -(-int(nbytes) // mtu))
            sizes = [mtu] * (n - 1) + [int(nbytes) - mtu * (n - 1)]
            readies = [t] * n
        m = Message(mid, cls, kind, t, len(sizes), copies > 1, info or {})
        self.msgs[mid] = m
        if self.cfg.priority and (cls == P0 or (backup_if_down and self.outages.is_down(t))):
            self._unacked.add(mid)
        q_cls = cls if self.cfg.priority else 0
        for s, tr in zip(sizes, readies):
            p = _Pkt(next(self._seq), cls, mid, max(1, s), t, tr)
            self.stats.packets += 1
            if not self._make_room(p):
                self.stats.overflow += 1
                m.failed += 1
                m.dropped = "overflow"
                self._resolve(m, t)
                continue
            self.queued_bytes += self._wire(p)
            heapq.heappush(self.queues[q_cls], (tr, p.seq, p))
        self._maybe_preempt(t, cls)
        if m.any_copy and m.failed >= m.n_packets:
            self._resolve(m, t)
        return mid

    def _make_room(self, p: _Pkt) -> bool:
        need = self.queued_bytes + self._wire(p) - self.ch.buffer_bytes
        if need <= 0:
            return True
        if not self.cfg.priority:
            return False  # FIFO: tail drop
        # Drop the oldest packets of the least important classes first.
        for c in range(P4, p.cls, -1):
            q = self.queues[c]
            while q and need > 0:
                *_, old = heapq.heappop(q)
                self._drop(old, "overflow", self.now)
                need -= self._wire(old)
        return need <= 0

    def _maybe_preempt(self, t: float, cls: int) -> None:
        if not (self.cfg.priority and self.cfg.preempt and self.cur is not None and cls <= P1):
            return
        p, s, e = self.cur
        if p.cls >= P2 and s < t < e:
            self.stats.preempted += 1
            self.stats.bytes_on_air += int(self._wire(p) * (t - s) / (e - s))  # partial airtime is wasted
            p.attempts -= 1
            self.queued_bytes += self._wire(p)
            heapq.heappush(self.queues[p.cls], (p.t_ready, p.seq, p))
            self.cur, self.t_free = None, t

    # -- simulation ---------------------------------------------------------- #
    def advance_to(self, t: float) -> None:
        """Simulate the link up to time ``t``."""
        while True:
            if self.cur is not None:
                p, s, e = self.cur
                if e > t:
                    break
                self.cur = None
                self._finish(p, s, e)
                continue
            start = self._next_start()
            if start is None or start >= t:
                break
            p = self._pop(start)
            if p is None:
                continue
            p.attempts += 1
            self.stats.attempts += 1
            self.cur = (p, start, start + self._wire(p) * 8 / self.bandwidth_at(start))
        self._backup(t)
        self.now = max(self.now, t)

    def flush(self, t_max: float) -> None:
        """Run the link until everything queued is resolved (or ``t_max``)."""
        step = max(1.0, self.now)
        t = self.now
        while t < t_max and (self.cur is not None or any(self.queues) or self._unacked):
            t = min(t_max, t + step)
            self.advance_to(t)

    def _next_start(self) -> float | None:
        tops = [q[0][0] for q in self.queues if q]
        if not tops:
            return None
        return self.outages.next_up(max(self.t_free, min(tops)))

    def _pop(self, start: float) -> _Pkt | None:
        """Highest-priority packet ready at ``start`` (drops stale / no longer needed ones)."""
        for q in self.queues:
            if not q or q[0][0] > start + 1e-12:
                continue
            *_, p = heapq.heappop(q)
            self.queued_bytes -= self._wire(p)
            m = self.msgs[p.msg]
            if m.any_copy and m.t_ack is not None and m.t_ack <= start:
                self.stats.cancelled += 1  # an earlier copy was already ACKed
                m.failed += 1
                return None
            if m.state == "lost" and not m.any_copy:
                self.stats.cancelled += 1  # another fragment already failed
                return None
            dl = self.cfg.deadline_s.get(p.cls)
            if dl is not None and start - p.t_created > dl:
                self._drop(p, "stale", start, counted=True)
                return None
            return p
        return None

    def _drop(self, p: _Pkt, why: str, t: float, counted: bool = False) -> None:
        if not counted:
            self.queued_bytes -= self._wire(p)
        if why == "stale":
            self.stats.stale += 1
        else:
            self.stats.overflow += 1
        m = self.msgs[p.msg]
        m.failed += 1
        m.dropped = m.dropped or why
        if not m.any_copy or m.failed >= m.n_packets:
            self._resolve(m, t)

    def _finish(self, p: _Pkt, s: float, e: float) -> None:
        wire = self._wire(p)
        self.stats.bytes_on_air += wire
        self.stats.by_class_bytes[p.cls] += wire
        m = self.msgs[p.msg]
        lost = self.loss.lost_at(s) or self.outages.is_down(e) or self.outages.is_down(s)
        acked = p.cls in self.cfg.ack_classes and self.cfg.priority
        rtt = 2 * self.ch.base_latency_s
        if not lost:
            arr = e + self.ch.base_latency_s + abs(self.rng.normal(0.0, self.ch.jitter_s))
            self.stats.delivered += 1
            m.delivered += 1
            m.got.add(p.seq)
            m.t_first = arr if m.t_first is None else min(m.t_first, arr)
            m.t_last = arr if m.t_last is None else max(m.t_last, arr)
            m.path = m.path or "primary"
            self.last_rx = arr if self.last_rx is None else max(self.last_rx, arr)
            if acked:
                ack_lost = self.loss.lost_at(arr) or self.outages.is_down(arr)
                if not ack_lost:
                    t_ack = arr + self.ch.base_latency_s
                    m.t_ack = t_ack if m.t_ack is None else min(m.t_ack, t_ack)
                    self._unacked.discard(m.id)
                else:
                    self._retry(p, e + rtt + self.cfg.ack_timeout_s, m)  # sender cannot tell: retry
            if m.state == "delivered":
                self._resolve(m, e)
            self.t_free = e
            return
        # lost
        if acked:
            self._retry(p, e + rtt + self.cfg.ack_timeout_s, m)
            self.t_free = e
            return
        if p.attempts <= self.ch.max_retries:
            # stop-and-wait ARQ for data: the link waits one round trip, then resends
            t_re = self.outages.next_up(e + rtt)
            p.attempts += 1
            self.stats.attempts += 1
            self.cur = (p, t_re, t_re + wire * 8 / self.bandwidth_at(t_re))
            self.t_free = t_re
            return
        self.stats.lost += 1
        m.failed += 1
        m.dropped = m.dropped or "loss"
        if not m.any_copy or m.failed >= m.n_packets:
            self._resolve(m, e)
        self.t_free = e

    def _retry(self, p: _Pkt, t_re: float, m: Message) -> None:
        if m.t_ack is not None:
            return
        if p.attempts > self.cfg.ack_retries:
            if p.seq in m.got:
                return  # it did arrive; only the ACKs were lost
            self.stats.lost += 1
            m.failed += 1
            if m.failed >= m.n_packets:
                self._resolve(m, t_re)
            return
        p.t_ready = t_re
        self.queued_bytes += self._wire(p)
        heapq.heappush(self.queues[p.cls if self.cfg.priority else 0], (t_re, p.seq, p))

    def _resolve(self, m: Message, t: float) -> None:
        if m.state != "pending" and m.t_resolved is None:
            m.t_resolved = t

    def _backup(self, t: float) -> None:
        b = self.cfg.backup
        if not (b.enabled and self.cfg.priority):
            return
        for mid in list(self._unacked):
            m = self.msgs[mid]
            # Down at the moment of the alert: no point waiting for an ACK.
            t_trig = m.t_created if self.outages.is_down(m.t_created) else m.t_created + b.after_s
            if m.t_ack is not None and m.t_ack <= t_trig:
                self._unacked.discard(mid)
                continue
            if t < t_trig:
                continue
            self._unacked.discard(mid)
            outs = OutageSchedule(b.outages)
            wire = self.msgs[mid].info.get("nbytes", 12) + 16
            tt = t_trig
            for _ in range(b.max_tries):
                tt = outs.next_up(tt)
                self.stats.backup_packets += 1
                self.stats.backup_bytes += wire
                if self._brng.random() >= b.loss_rate:
                    arr = tt + wire * 8 / b.bandwidth_bps + b.latency_s
                    m.t_backup = arr
                    t_ack = arr + b.latency_s
                    m.t_ack = t_ack if m.t_ack is None else min(m.t_ack, t_ack)
                    if m.t_first is None or arr < m.t_first:
                        m.t_first, m.path = arr, "backup"
                    break
                tt += b.retry_s
            self._resolve(m, t)

    # -- results ------------------------------------------------------------- #
    def status(self, mid: int) -> Message:
        return self.msgs[mid]

    def energy_j(self, tx_per_bit: float, tx_per_packet: float) -> dict:
        st, b = self.stats, self.cfg.backup
        primary = st.bytes_on_air * 8 * tx_per_bit + st.attempts * tx_per_packet
        backup = st.backup_bytes * 8 * b.energy_per_bit_j + st.backup_packets * b.energy_per_packet_j
        return {"primary_j": primary, "backup_j": backup}
