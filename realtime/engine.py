"""Real-time streaming engine: wearable -> priority link -> receiver.

PulseLink's goal is the **fastest reliable alert**: the receiver (phone,
caregiver, clinician) should learn about an event within seconds, even on a
slow, lossy or broken link. The engine is built around that goal.

On the wearable, every ``tick_s`` (0.5 s):

1. causal band-pass (what an MCU can do) and a ring buffer;
2. **two-stage detection**: each critical detector has a fast early stage
   (2 s window, every tick) that raises a *possible event* warning, and the
   full detector (4 s window, every frame) that *confirms* it;
3. alerts go to the :class:`~communication.priority_link.PriorityLink`:
   P0 for confirmed critical events (spaced copies, ACK and retry, backup
   radio, local alarm if nothing is ACKed), P1 for early warnings, other
   notifications and a heartbeat.

Every ``frame_s`` (2 s), the EEG frame is compressed and queued as P2
(around events, high fidelity, plus the pre-trigger buffer) or P3 (routine,
low rate). The receiver decodes frames as they arrive, conceals lost ones,
runs the same detectors on the reconstruction (the "stream everything,
detect on the phone" baseline) and keeps a notification centre.

Presets in :func:`preset` switch between the full PulseLink design and a
typical streaming wearable, so both can be compared on the same signal.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import signal

from communication.packet_loss import OutageSchedule
from communication.priority_link import P0, P1, P2, P3, P4, LinkConfig, PriorityLink
import torch

from compression.codecs import AutoencoderCodec, Codec, CodecRegistry, RawCodec, WaveletCodec, prd
from config import ChannelConfig, EnergyConfig
from realtime.detectors import Detector, DetectorOutput, EarlyOutput, EarlyStage, SleepStageDetector, default_detectors
from realtime.features import compute_features, feature_ops
from realtime.notify import NotificationCenter

POLICIES = ("raw", "dwt", "pulselink", "event")
# Frame-sized tensors are tiny: extra threads only add overhead (and oversubscribe
# the CPU when several runs share it). One thread is also what a wearable has.
torch.set_num_threads(1)


@dataclass
class EngineConfig:
    frame_s: float = 2.0
    tick_s: float = 0.5  # detection hop on the wearable
    policy: str = "event"
    normal_codec: str = "pulselink"  # used by the event policy when nothing happens
    pre_event_s: float = 10.0
    post_event_s: float = 10.0
    device_alerts: bool = True  # False = no on-device detection alerts (stream-only wearable)
    early_warning: bool = True
    alert_copies: int = 2
    alert_gap_s: float = 0.1  # spacing between copies (time diversity against bursts)
    alert_bytes: int = 12  # type, detector id, confidence, timestamp, CRC
    heartbeat_s: float = 2.0  # 0 = off; only sent while EEG data is queued (otherwise the data itself is the heartbeat)
    heartbeat_bytes: int = 4  # also says "no event as of now", so the receiver's all-clear stays fresh
    offline_after_s: float = 6.0  # receiver: silence longer than this (3 missed frames) = "monitoring offline"
    local_alarm_after_s: float = 2.0  # wearable buzzes if a P0 alert is not ACKed by then
    escalate_after_s: float = 30.0
    receiver_detection: bool = True
    link: LinkConfig = field(default_factory=LinkConfig)
    l_freq: float = 0.5
    h_freq: float = 40.0
    battery_mah: float = 100.0
    battery_v: float = 3.7
    afe_mw: float = 0.5  # always-on analog front end + MCU idle (assumption)


def preset(name: str, **kw) -> EngineConfig:
    """Named system designs for comparisons.

    * ``pulselink``: everything on (priority link, early warning, ACK, spaced
      copies, backup radio, heartbeat, event-aware compression).
    * ``stream``: a typical wearable: compressed EEG over one FIFO queue,
      events detected on the receiver only.
    * ``stream_raw``: same, uncompressed.
    * ablations ``device_fifo``, ``+priority``, ``+early``, ``+ack``: build
      PulseLink up one feature at a time.
    """
    fifo = LinkConfig(priority=False, preempt=False, ack_classes=(), deadline_s={}, backup=replace(LinkConfig().backup, enabled=False))
    no_backup = LinkConfig(backup=replace(LinkConfig().backup, enabled=False))
    base = {
        "pulselink": dict(),
        "stream": dict(policy="dwt", device_alerts=False, early_warning=False, heartbeat_s=0, link=fifo),
        "stream_raw": dict(policy="raw", device_alerts=False, early_warning=False, heartbeat_s=0, link=fifo),
        "device_fifo": dict(early_warning=False, alert_copies=1, heartbeat_s=0, link=fifo),
        "+priority": dict(early_warning=False, alert_copies=1, link=replace(no_backup, ack_classes=())),
        "+early": dict(alert_copies=1, link=replace(no_backup, ack_classes=())),
        "+ack": dict(link=no_backup),
    }[name]
    return EngineConfig(**{**base, **kw})


class StreamingBandpass:
    """Causal Butterworth band-pass with state, so frames join seamlessly."""

    def __init__(self, fs: float, n_channels: int, lo: float = 0.5, hi: float = 40.0, order: int = 2):
        hi = min(hi, 0.95 * fs / 2)
        self.sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        self.zi = None
        self.c = n_channels

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.zi is None:
            base = signal.sosfilt_zi(self.sos)
            self.zi = np.stack([base * x[c, 0] for c in range(self.c)], axis=1)  # (sections, C, 2)
        y, self.zi = signal.sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y


class DetectorBank:
    """Runs detectors on a ring buffer of the most recent signal.

    Each detector runs on its own hop (``hop_s``, default one frame); early
    stages run on every push.
    """

    def __init__(self, detectors: list[Detector], fs: float, n_channels: int, frame_s: float, early: bool = False):
        self.detectors = detectors
        self.fs, self.frame_s = fs, frame_s
        self.early = {d.name: EarlyStage(d) for d in detectors if early and d.has_early}
        wins = [d.window_s for d in detectors] + [s.window_s for s in self.early.values()] + [frame_s]
        self.max_s = max(wins)
        self.buf = np.zeros((n_channels, int(round(self.max_s * fs))))
        self.valid = np.zeros(self.buf.shape[1], dtype=bool)
        self.t = 0.0
        self.ops = 0
        self.last: dict[str, DetectorOutput] = {d.name: DetectorOutput(d.name, None, False) for d in detectors}
        self.last_early: dict[str, EarlyOutput] = {}
        self._by_name = {d.name: d for d in detectors}

    def push(self, x: np.ndarray | None, n: int) -> None:
        """Append samples (``None`` = lost frame, marked invalid)."""
        self.buf = np.roll(self.buf, -n, axis=1)
        self.valid = np.roll(self.valid, -n)
        if x is None:
            self.buf[:, -n:] = 0.0
            self.valid[-n:] = False
        else:
            self.buf[:, -n:] = x
            self.valid[-n:] = True
        self.t += n / self.fs

    def _on_grid(self, hop: float) -> bool:
        r = self.t / hop
        return abs(r - round(r)) < 1e-6

    def _feat(self, cache: dict, w: int):
        if w not in cache:
            cache[w] = compute_features(self.buf[:, -w:], self.fs)
            self.ops += feature_ops(self.buf.shape[0], w, self.fs)
        return cache[w]

    def run(self) -> tuple[dict[str, DetectorOutput], dict[str, EarlyOutput]]:
        feats: dict[int, object] = {}
        outs: dict[str, DetectorOutput] = {}
        artifact = self.last.get("artifact", DetectorOutput("artifact", None, False)).active
        for d in sorted(self.detectors, key=lambda d: d.name != "artifact"):
            if not self._on_grid(d.hop_s or self.frame_s):
                o = self.last[d.name]
                outs[d.name] = DetectorOutput(d.name, o.score, o.active, label=o.label)
                continue
            w = int(round(d.window_s * self.fs))
            if not self.valid[-w:].all():
                outs[d.name] = self.last[d.name] = DetectorOutput(d.name, None, d.active)
                continue
            if isinstance(d, SleepStageDetector) or getattr(d, "uses_raw", False):
                o = d.step_window(self.buf[:, -w:]) if isinstance(d, SleepStageDetector) else d.step(self.buf[:, -w:], artifact)
            else:
                o = d.step(self._feat(feats, w), artifact)
            if d.name == "artifact":
                artifact = o.active
            outs[d.name] = self.last[d.name] = o
        # A critical event (e.g. a seizure) mutes the lifestyle detectors.
        if any(o.active and self._det(n).critical for n, o in outs.items()):
            for n, o in outs.items():
                if not self._det(n).critical and n != "artifact":
                    outs[n] = DetectorOutput(n, o.score, False, label=o.label)
        early = {}
        for n, st in self.early.items():
            w = int(round(st.window_s * self.fs))
            if not self.valid[-w:].all():
                early[n] = EarlyOutput(n, None)
                continue
            early[n] = st.step(self._feat(feats, w), self.t, artifact)
        self.last_early = early
        return outs, early

    def _det(self, name: str) -> Detector:
        return self._by_name[name]


def make_codecs(model) -> dict[str, Codec]:
    codecs = {"raw": RawCodec(), "dwt": WaveletCodec(0.1, 8), "mid": WaveletCodec(0.3, 8), "hifi": WaveletCodec(0.5, 10)}
    if model is not None:
        codecs["pulselink"] = AutoencoderCodec(model, bits=4, keep=model.cfg.latent_channels // 2)
    return codecs


@dataclass
class EventLog:
    detector: str
    t_onset_device: float  # wearable confirmed the event
    t_alert_arrival: float | None = None  # confirmed alert reaches the receiver (None = not (yet) delivered)
    alert_copies_delivered: int = 0
    alert_path: str | None = None  # "primary" or "backup"
    priority: int | None = None
    t_early_device: float | None = None  # possible-event warning raised on the wearable
    t_early_arrival: float | None = None  # ... and received
    t_local_alarm: float | None = None  # wearable buzzed because nothing was ACKed
    t_offset_device: float | None = None
    t_onset_receiver: float | None = None  # receiver-side detection on reconstructed data
    alert_mid: int | None = field(default=None, repr=False)
    early_mid: int | None = field(default=None, repr=False)

    @property
    def t_first_alert(self) -> float | None:
        """When the receiver first knew (early warning or confirmed alert)."""
        v = [t for t in (self.t_early_arrival, self.t_alert_arrival) if t is not None]
        return min(v) if v else None


@dataclass
class EarlyLog:
    detector: str
    t_device: float
    mid: int
    t_arrival: float | None = None
    confirmed: bool = False
    cancelled_at: float | None = None


class StreamEngine:
    def __init__(self, source, model=None, channel: ChannelConfig | None = None, energy: EnergyConfig | None = None, cfg: EngineConfig | None = None, detectors: list[Detector] | None = None):
        self.source = source
        self.fs = source.fs
        self.c = source.c
        self.cfg = cfg or EngineConfig()
        self.ecfg = energy or EnergyConfig()
        self.channel_cfg = copy.deepcopy(channel or ChannelConfig())
        self.link = PriorityLink(copy.deepcopy(self.channel_cfg), copy.deepcopy(self.cfg.link))
        self.codecs = make_codecs(model)
        if self.cfg.normal_codec not in self.codecs:
            self.cfg.normal_codec = "dwt"
        self.registry = CodecRegistry(model)
        self.n = int(round(self.cfg.frame_s * self.fs))
        self.n_tick = int(round(self.cfg.tick_s * self.fs))
        if self.n % self.n_tick:
            raise ValueError("frame_s must be a multiple of tick_s")
        self.filter = StreamingBandpass(self.fs, self.c, self.cfg.l_freq, self.cfg.h_freq)
        dets = detectors if detectors is not None else default_detectors()
        self.device = DetectorBank(dets, self.fs, self.c, self.cfg.frame_s, early=self.cfg.early_warning and self.cfg.device_alerts)
        self.receiver = DetectorBank([d.clone() for d in dets], self.fs, self.c, self.cfg.frame_s)
        self.recent: deque[tuple[int, np.ndarray]] = deque(maxlen=max(1, int(round(self.cfg.pre_event_s / self.cfg.frame_s))))
        self.notify = NotificationCenter(escalate_after_s=self.cfg.escalate_after_s)
        self.frame = 0
        self.t = 0.0
        self.enc_ops = 0
        self.events: list[EventLog] = []
        self.early_logs: list[EarlyLog] = []
        self.open_events: dict[str, EventLog] = {}
        self._open_early: dict[str, EarlyLog] = {}
        self.hifi_until = -1.0
        self.records: list[dict] = []
        self.recon: dict[int, np.ndarray] = {}
        self.orig: dict[int, np.ndarray] = {}
        self._rx_time = 0.0
        self._avg_bytes: dict[str, float] = {}
        self.raw_bytes_per_frame = self.c * self.n * 2
        self._pending_frames: deque[tuple[int, int, bytes]] = deque()  # (frame id, message id, blob) in order
        self._pending_backfill: list[tuple[int, int, bytes]] = []
        self._rec_by_frame: dict[int, dict] = {}
        self._last_hb = -np.inf
        self._rx_seen = 0.0  # receiver: last arrival processed by the offline watchdog
        self.offline_since: float | None = None
        self.offline_log: list[tuple[float, float | None]] = []
        self._resolved_now: list[dict] = []
        self._backfilled_now: list[int] = []

    # -- compatibility ------------------------------------------------------- #
    @property
    def sim(self) -> PriorityLink:
        return self.link

    # -- live controls ------------------------------------------------------- #
    def set_channel(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.channel_cfg, k, v)
            setattr(self.link.ch, k, v)
        if "loss_rate" in kw or "burst_length" in kw:
            self.link.set_loss()
        if "bandwidth_bps" in kw:
            self.link.bw_schedule = [(0.0, float(kw["bandwidth_bps"]))]
        if "outages" in kw:
            self.link.outages = OutageSchedule(kw["outages"])

    def outage_now(self, duration_s: float) -> None:
        self.set_channel(outages=list(self.channel_cfg.outages) + [(self.t, duration_s)])

    # -- helpers ------------------------------------------------------------- #
    def _fits(self, codec: str, share: float = 0.6) -> bool:
        """Would this codec use at most ``share`` of the current link capacity?"""
        nbytes = self._avg_bytes.get(codec)
        if nbytes is None:
            return True
        return (nbytes + 30) * 8 / self.cfg.frame_s <= share * self.link.bandwidth_at(self.t)

    def _choose(self, outs) -> str:
        p = self.cfg.policy
        if p != "event":
            return p if p in self.codecs else "dwt"
        if self.t < self.hifi_until or any(o.active and self.device._det(n).critical for n, o in outs.items()):
            # Highest fidelity the link can carry: a slow link must not turn
            # "high fidelity" into "hopelessly delayed".
            for c in ("hifi", "mid", "dwt"):
                if self._fits(c):
                    return c
            return "dwt"
        return self.cfg.normal_codec

    def _alert(self, t: float, cls: int, kind: str, det: Detector, copies: int = 1, backup_if_down: bool = False, **info) -> int:
        cfg = self.cfg
        return self.link.send(t, cfg.alert_bytes, cls, kind=kind, copies=copies, gap_s=cfg.alert_gap_s, info={"detector": det.name, "title": det.title, "nbytes": cfg.alert_bytes, **info}, backup_if_down=backup_if_down)

    # -- wearable, one tick -------------------------------------------------- #
    def _tick(self, x: np.ndarray, tau: float) -> tuple[dict, dict, list]:
        cfg = self.cfg
        self.link.advance_to(tau)
        self.device.push(x, x.shape[1])
        outs, early = self.device.run()
        alerts = []
        if cfg.device_alerts:
            for name, eo in early.items():
                det = self.device._det(name)
                if eo.onset:
                    mid = self._alert(tau, P1, "early", det, copies=1, backup_if_down=True)
                    el = EarlyLog(name, tau, mid)
                    self.early_logs.append(el)
                    self._open_early[name] = el
                    alerts.append({"detector": name, "title": det.title, "kind": "early", "priority": P1, "t": tau})
                if eo.cancelled and name in self._open_early:
                    el = self._open_early.pop(name)
                    el.cancelled_at = tau
                    self._alert(tau, P1, "cancel", det)
            for name, o in outs.items():
                det = self.device._det(name)
                if o.onset:
                    cls = det.notify_class
                    ev = EventLog(name, tau, priority=cls)
                    if det.critical:
                        ev.alert_mid = self._alert(tau, P0, "alert", det, copies=max(1, cfg.alert_copies))
                        self.hifi_until = tau + cfg.post_event_s
                    elif cls <= P1:
                        ev.alert_mid = self._alert(tau, cls, "notice", det)
                    else:
                        ev.alert_mid = self._alert(tau, cls, "log", det)
                    el = self._open_early.pop(name, None)
                    if el is not None and tau - el.t_device <= det.confirm_s + 1e-9:
                        el.confirmed = True
                        ev.t_early_device, ev.early_mid = el.t_device, el.mid
                    self.events.append(ev)
                    self.open_events[name] = ev
                    alerts.append({"detector": name, "title": det.title, "kind": "alert" if det.critical else "notice", "critical": det.critical, "priority": P0 if det.critical else cls, "t": tau})
                if o.offset and name in self.open_events:
                    ev = self.open_events.pop(name)
                    ev.t_offset_device = tau
                    if det.notify_class <= P1:
                        self._alert(tau, P1, "resolved", det)
                if o.active and det.critical:
                    self.hifi_until = tau + cfg.post_event_s
        if cfg.heartbeat_s and tau - self._last_hb >= cfg.heartbeat_s - 1e-9:
            bl = self.link.backlog()
            if bl[P2] + bl[P3] + bl[P4] > 0 or not self.link.link_up(tau):
                self.link.send(tau, cfg.heartbeat_bytes, P1, kind="heartbeat")
            self._last_hb = tau
        # local alarm: critical alert not ACKed in time (wearer / bystanders)
        for ev in self.open_events.values():
            if ev.alert_mid is not None and ev.t_local_alarm is None and ev.priority == P0 and tau - ev.t_onset_device >= cfg.local_alarm_after_s - 1e-9:
                m = self.link.status(ev.alert_mid)
                if m.t_ack is None or m.t_ack > ev.t_onset_device + cfg.local_alarm_after_s:
                    ev.t_local_alarm = ev.t_onset_device + cfg.local_alarm_after_s
        return outs, early, alerts

    # -- one frame ----------------------------------------------------------- #
    def step(self, chunk=None) -> dict:
        """Process one frame. ``chunk`` lets a second engine share the same signal."""
        cfg = self.cfg
        chunk = chunk if chunk is not None else self.source.read(self.n)
        x = self.filter(chunk.data)
        fid = self.frame
        self.t = t = chunk.t_end
        t0 = t - cfg.frame_s
        self.orig[fid] = x
        if fid == 0:  # calibrate the expected size of each codec once
            for name in ("hifi", "mid", "dwt", cfg.normal_codec):
                self._avg_bytes[name] = float(self.codecs[name].encode(x).nbytes)

        alerts, early = [], {}
        nt = self.n_tick
        for k in range(self.n // nt):
            outs, e, a = self._tick(x[:, k * nt : (k + 1) * nt], t0 + (k + 1) * nt / self.fs)
            alerts += a
            early.update({n: v for n, v in e.items() if v.score is not None or n not in early})

        # pre-trigger buffer: re-send the recent past at high fidelity (P2)
        if cfg.policy == "event" and any(a.get("kind") == "alert" for a in alerts) and self._fits("hifi", 0.25):
            for bid, bx in list(self.recent):
                enc = self.codecs["hifi"].encode(bx)
                self.enc_ops += enc.encode_ops
                mid = self.link.send(t, enc.nbytes, P2, kind="backfill", info={"frame": bid})
                self._pending_backfill.append((bid, mid, enc.blob))
        self.recent.append((fid, x))

        # encode + queue this frame
        mode = self._choose(outs)
        codec = self.codecs[mode]
        enc = codec.encode(x)
        self.enc_ops += enc.encode_ops
        self._avg_bytes[mode] = 0.8 * self._avg_bytes.get(mode, enc.nbytes) + 0.2 * enc.nbytes
        in_event = self.t < self.hifi_until or any(o.active and self.device._det(n).critical for n, o in outs.items())
        cls = P2 if cfg.policy == "event" and in_event else P3
        mid = self.link.send(t, enc.nbytes, cls, kind="frame", info={"frame": fid})
        self._pending_frames.append((fid, mid, enc.blob))

        rec = {
            "frame": fid,
            "t": t,
            "mode": codec.mode_name(),
            "codec": mode,
            "priority": cls,
            "bytes": enc.nbytes,
            "cr": self.raw_bytes_per_frame / enc.nbytes,
            "packets": self.link.status(mid).n_packets,
            "status": "pending",
            "latency_s": None,
            "prd": None,
            "truth": sorted(chunk.truth),
            "label": chunk.label,
            "device": {n: {"score": o.score, "active": o.active, "label": o.label} for n, o in outs.items()},
            "early": {n: {"score": o.score, "pending": o.pending} for n, o in early.items()},
            "receiver": {},
            "alerts": alerts,
            "backfilled": [],
        }
        self.records.append(rec)
        self._rec_by_frame[fid] = rec
        self.frame += 1
        self._receive(t)
        rec["resolved"] = self._resolved_now
        rec["backfilled"] = self._backfilled_now
        return rec

    # -- receiver ------------------------------------------------------------ #
    def _receive(self, t: float, final: bool = False) -> None:
        """Decode every frame whose fate is known, in order; update alerts."""
        self._resolved_now, self._backfilled_now = [], []
        while self._pending_frames:
            fid, mid, blob = self._pending_frames[0]
            m = self.link.status(mid)
            if m.state == "pending":
                break
            self._pending_frames.popleft()
            rec = self._rec_by_frame.get(fid)
            ok = m.state == "delivered"
            y = self.registry.decode(blob)[0] if ok else None
            x = self.orig.get(fid)
            self.recon[fid] = y if ok else (np.zeros_like(x) if x is not None else None)
            t_rx = m.t_arrival if ok else (m.t_resolved or t)
            self._rx_time = max(self._rx_time, t_rx)
            rx = {}
            if self.cfg.receiver_detection:
                self.receiver.push(y, self.n)
                rx, _ = self.receiver.run()
                for name, o in rx.items():
                    if o.onset:
                        cand = [e for e in self.events if e.detector == name and e.t_onset_receiver is None and np.isfinite(e.t_onset_device) and e.t_onset_device >= self.receiver.t - 60]
                        if cand:
                            cand[-1].t_onset_receiver = self._rx_time
                        else:
                            self.events.append(EventLog(name, np.nan, t_onset_receiver=self._rx_time))
            if rec is not None:
                rec["status"] = "ok" if ok else "lost"
                rec["latency_s"] = (t_rx - rec["t"]) if ok else None
                rec["prd"] = prd(x, y) if ok and x is not None else 100.0
                rec["receiver"] = {n: {"score": o.score, "active": o.active, "label": o.label} for n, o in rx.items()}
                self._resolved_now.append({k: rec[k] for k in ("frame", "status", "latency_s", "prd", "codec", "receiver")})
        keep = []
        for bid, mid, blob in self._pending_backfill:
            m = self.link.status(mid)
            if m.state == "pending":
                keep.append((bid, mid, blob))
            elif m.state == "delivered" and bid in self.orig:
                self.recon[bid] = self.registry.decode(blob)[0]
                if bid in self._rec_by_frame:
                    self._rec_by_frame[bid]["backfilled_prd"] = prd(self.orig[bid], self.recon[bid])
                self._backfilled_now.append(bid)
        self._pending_backfill = keep
        self._refresh_alerts(t, final)
        self._watchdog(t, final)

    def _refresh_alerts(self, t: float, final: bool = False) -> None:
        horizon = np.inf if final else t
        for ev in self.events:
            if ev.alert_mid is not None and ev.t_alert_arrival is None:
                m = self.link.status(ev.alert_mid)
                if m.t_arrival is not None and m.t_arrival <= horizon:
                    ev.t_alert_arrival, ev.alert_copies_delivered, ev.alert_path = m.t_arrival, m.delivered, m.path
                    kind = "alert" if ev.priority == P0 else ("notice" if ev.priority == P1 else "log")
                    self.notify.add(m.t_arrival, ev.priority, kind, ev.detector, self._title(ev.detector), t_event=ev.t_onset_device, path=m.path)
            if ev.early_mid is not None and ev.t_early_arrival is None:
                m = self.link.status(ev.early_mid)
                if m.t_arrival is not None and m.t_arrival <= horizon:
                    ev.t_early_arrival = m.t_arrival
        for el in self.early_logs:
            if el.t_arrival is None:
                m = self.link.status(el.mid)
                if m.t_arrival is not None and m.t_arrival <= horizon:
                    el.t_arrival = m.t_arrival
                    self.notify.add(m.t_arrival, P1, "early", el.detector, self._title(el.detector), t_event=el.t_device)
        if self.cfg.receiver_detection and not self.cfg.device_alerts:
            # stream-only wearable: the phone raises the alert itself
            for ev in self.events:
                if ev.t_onset_receiver is not None and not getattr(ev, "_rx_notified", False):
                    ev._rx_notified = True
                    det = self.device._det(ev.detector)
                    self.notify.add(ev.t_onset_receiver, P0 if det.critical else det.notify_class, "alert" if det.critical else "notice", ev.detector, det.title, t_event=ev.t_onset_receiver, path="receiver")
        self.notify.tick(t)

    def _title(self, name: str) -> str:
        return self.device._det(name).title

    def _watchdog(self, t: float, final: bool = False) -> None:
        """Receiver: silence is not "all clear". Flag gaps in what arrives."""
        limit = self.cfg.offline_after_s if self.cfg.heartbeat_s else max(self.cfg.offline_after_s, 3 * self.cfg.frame_s)
        last = self.link.last_rx
        if last is None:
            return  # still waiting for the wearable's first packet
        if last > self._rx_seen:
            if self.offline_since is not None:
                self.offline_log[-1] = (self.offline_log[-1][0], last)
                self.notify.add(last, P1, "online", None, "Monitoring restored")
                self.offline_since = None
            self._rx_seen = last
        if self.offline_since is None and not final and t - self._rx_seen > limit:
            self.offline_since = self._rx_seen + limit
            self.offline_log.append((self.offline_since, None))
            self.notify.add(self.offline_since, P1, "offline", None, "Monitoring offline: no data from the wearable")

    def run(self, n_frames: int) -> list[dict]:
        return [self.step() for _ in range(n_frames)]

    def finish(self, extra_s: float = 600.0) -> None:
        """Let the link drain so every queued message is resolved (for scoring)."""
        self.link.flush(self.t + extra_s)
        self._receive(self.t, final=True)

    # -- summaries ----------------------------------------------------------- #
    def energy_j(self) -> dict:
        e = self.ecfg
        le = self.link.energy_j(e.tx_energy_per_bit_j, e.tx_energy_per_packet_j)
        mcu = (self.enc_ops + self.device.ops) * e.mcu_energy_per_op_j
        return {"radio_j": le["primary_j"], "backup_radio_j": le["backup_j"], "compute_j": mcu, "transmitter_j": le["primary_j"] + le["backup_j"] + mcu}

    def raw_reference_j(self) -> float:
        """Energy if every frame so far had been sent raw over an ideal link."""
        e = self.ecfg
        n_pk = -(-(self.raw_bytes_per_frame + 21) // self.link.ch.mtu_bytes)
        per = (self.raw_bytes_per_frame + 21 + n_pk * self.link.ch.header_bytes) * 8 * e.tx_energy_per_bit_j + n_pk * e.tx_energy_per_packet_j
        return per * max(1, self.frame)

    def summary(self) -> dict:
        recs = self.records
        dur = max(self.t, 1e-9)
        en = self.energy_j()
        ok = [r for r in recs if r["status"] == "ok"]
        tx_w = en["transmitter_j"] / dur
        raw_w = self.raw_reference_j() / dur
        batt_j = self.cfg.battery_mah * 3.6 * self.cfg.battery_v
        afe_w = self.cfg.afe_mw * 1e-3
        st = self.link.stats
        return {
            "seconds": dur,
            "frames": len(recs),
            "compression_ratio": self.raw_bytes_per_frame * len(recs) / max(1, sum(r["bytes"] for r in recs)),
            "bitrate_bps": 8 * sum(r["bytes"] for r in recs) / dur,
            "prd_delivered": float(np.mean([r["prd"] for r in ok])) if ok else np.nan,
            "frames_lost": sum(r["status"] == "lost" for r in recs),
            "frames_pending": sum(r["status"] == "pending" for r in recs),
            "packet_loss": st.loss_rate,
            "mean_latency_s": float(np.mean([r["latency_s"] for r in ok])) if ok else np.nan,
            "max_latency_s": float(np.max([r["latency_s"] for r in ok])) if ok else np.nan,
            "preempted": st.preempted,
            "stale_dropped": st.stale,
            "backlog_bytes": sum(self.link.backlog().values()),
            "offline": self.offline_since is not None,
            "tx_power_mw": 1e3 * tx_w,
            "raw_tx_power_mw": 1e3 * raw_w,
            "energy_saving_pct": 100 * (1 - tx_w / raw_w) if raw_w else 0.0,
            "battery_h": batt_j / (tx_w + afe_w) / 3600,
            "battery_h_raw": batt_j / (raw_w + afe_w) / 3600,
            **{f"energy_{k}": v for k, v in en.items()},
        }
