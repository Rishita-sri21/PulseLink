"""Real-time streaming engine: wearable -> wireless link -> receiver, frame by frame.

Per frame (default 2 s):

1. **Wearable**: read EEG, causal band-pass (what an MCU can do), keep a
   short ring buffer, run every detector on the *raw* signal (on-device
   detection), choose a codec with the transmission policy, encode.
2. **Event-aware transmission** (policy ``event``): low-rate codec while
   nothing is happening; when a *critical* detector (e.g. seizure) fires:
   a tiny **alert packet** is sent at once, ahead of the data queue, as
   ``alert_copies`` redundant copies; the last ``pre_event_s`` seconds are
   re-sent at high fidelity (pre-trigger buffer, like a cardiac event
   recorder); data stays high fidelity until ``post_event_s`` after the event.
3. **Link**: the existing :class:`ChannelSimulator` (bandwidth, loss,
   bursts, outages, ARQ).
4. **Receiver**: decode, conceal lost frames, and run the same detectors
   on the *reconstructed* signal, so on-device and receiver-side detection
   can be compared (latency and accuracy).

Everything is incremental, so the live dashboard calls :meth:`StreamEngine.step`
once per frame and can change the channel or policy while streaming.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from communication.channel_simulator import ChannelSimulator
from communication.packet_loss import OutageSchedule, make_loss_model
from compression.codecs import AutoencoderCodec, Codec, CodecRegistry, RawCodec, WaveletCodec, prd
from config import ChannelConfig, EnergyConfig
from realtime.detectors import Detector, DetectorOutput, SleepStageDetector, default_detectors
from realtime.features import compute_features, feature_ops

POLICIES = ("raw", "dwt", "pulselink", "event")


@dataclass
class EngineConfig:
    frame_s: float = 2.0
    policy: str = "event"
    normal_codec: str = "pulselink"  # used by the event policy when nothing happens
    pre_event_s: float = 10.0
    post_event_s: float = 10.0
    alert_copies: int = 3
    alert_gap_s: float = 0.03  # spacing between redundant alert copies
    alert_bytes: int = 12  # type, detector id, confidence, timestamp, CRC
    receiver_detection: bool = True
    l_freq: float = 0.5
    h_freq: float = 40.0
    battery_mah: float = 100.0
    battery_v: float = 3.7
    afe_mw: float = 0.5  # always-on analog front end + MCU idle (assumption)


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
    """Runs a set of detectors on a ring buffer of the most recent signal."""

    def __init__(self, detectors: list[Detector], fs: float, n_channels: int, frame_s: float):
        self.detectors = detectors
        self.fs, self.frame_s = fs, frame_s
        self.max_s = max([d.window_s for d in detectors] + [frame_s])
        self.buf = np.zeros((n_channels, int(round(self.max_s * fs))))
        self.valid = np.zeros(self.buf.shape[1], dtype=bool)
        self.t = 0.0
        self.ops = 0
        self.last: dict[str, DetectorOutput] = {}

    def push(self, x: np.ndarray | None, n: int) -> None:
        """Append a frame (``None`` = lost frame, marked invalid)."""
        self.buf = np.roll(self.buf, -n, axis=1)
        self.valid = np.roll(self.valid, -n)
        if x is None:
            self.buf[:, -n:] = 0.0
            self.valid[-n:] = False
        else:
            self.buf[:, -n:] = x
            self.valid[-n:] = True
        self.t += n / self.fs

    def run(self) -> dict[str, DetectorOutput]:
        feats, outs = {}, {}
        ordered = sorted(self.detectors, key=lambda d: d.name != "artifact")
        artifact = False
        for d in ordered:
            w = int(round(d.window_s * self.fs))
            if not self.valid[-w:].all():
                outs[d.name] = DetectorOutput(d.name, None, d.active)
                continue
            if isinstance(d, SleepStageDetector):
                hop = d.hop_s or self.frame_s
                if abs((self.t / hop) - round(self.t / hop)) < 1e-6:
                    outs[d.name] = self.last[d.name] = d.step_window(self.buf[:, -w:])
                else:
                    outs[d.name] = self.last.get(d.name, DetectorOutput(d.name, None, False))
                continue
            if w not in feats:
                feats[w] = compute_features(self.buf[:, -w:], self.fs)
                self.ops += feature_ops(self.buf.shape[0], w, self.fs)
            o = d.step(feats[w], artifact)
            if d.name == "artifact":
                artifact = o.active
            outs[d.name] = o
        # A critical event (e.g. a seizure) mutes the lifestyle detectors.
        if any(o.active and self._det(n).critical for n, o in outs.items()):
            for n, o in outs.items():
                if not self._det(n).critical and n != "artifact":
                    outs[n] = DetectorOutput(n, o.score, False, label=o.label)
        return outs

    def _det(self, name: str) -> Detector:
        return next(d for d in self.detectors if d.name == name)


def make_codecs(model) -> dict[str, Codec]:
    codecs = {"raw": RawCodec(), "dwt": WaveletCodec(0.1, 8), "mid": WaveletCodec(0.3, 8), "hifi": WaveletCodec(0.5, 10)}
    if model is not None:
        codecs["pulselink"] = AutoencoderCodec(model, bits=4, keep=model.cfg.latent_channels // 2)
    return codecs


@dataclass
class EventLog:
    detector: str
    t_onset_device: float  # when the wearable decided (end of the frame)
    t_alert_arrival: float | None  # first alert copy at the receiver (None = all lost)
    alert_copies_delivered: int
    t_offset_device: float | None = None
    t_onset_receiver: float | None = None  # receiver-side detection on reconstructed data


class StreamEngine:
    def __init__(self, source, model=None, channel: ChannelConfig | None = None, energy: EnergyConfig | None = None, cfg: EngineConfig | None = None, detectors: list[Detector] | None = None):
        self.source = source
        self.fs = source.fs
        self.c = source.c
        self.cfg = cfg or EngineConfig()
        self.ecfg = energy or EnergyConfig()
        self.channel_cfg = copy.deepcopy(channel or ChannelConfig())
        self.sim = ChannelSimulator(self.channel_cfg)
        self.codecs = make_codecs(model)
        if self.cfg.normal_codec not in self.codecs:
            self.cfg.normal_codec = "dwt"
        self.registry = CodecRegistry(model)
        self.n = int(round(self.cfg.frame_s * self.fs))
        self.filter = StreamingBandpass(self.fs, self.c, self.cfg.l_freq, self.cfg.h_freq)
        dets = detectors if detectors is not None else default_detectors()
        self.device = DetectorBank(dets, self.fs, self.c, self.cfg.frame_s)
        self.receiver = DetectorBank([d.clone() for d in dets], self.fs, self.c, self.cfg.frame_s)
        self.recent: deque[tuple[int, np.ndarray]] = deque(maxlen=max(1, int(round(self.cfg.pre_event_s / self.cfg.frame_s))))
        self.frame = 0
        self.t = 0.0
        self.enc_ops = 0
        self.alert_bytes_air = 0
        self.alert_packets = 0
        self.events: list[EventLog] = []
        self.open_events: dict[str, EventLog] = {}
        self.hifi_until = -1.0
        self.records: list[dict] = []
        self.recon: dict[int, np.ndarray] = {}
        self.orig: dict[int, np.ndarray] = {}
        self._rx_time = 0.0
        self._avg_bytes: dict[str, float] = {}
        self.raw_bytes_per_frame = self.c * self.n * 2

    # -- live controls ------------------------------------------------------- #
    def set_channel(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.channel_cfg, k, v)
            setattr(self.sim.cfg, k, v)
        if "loss_rate" in kw or "burst_length" in kw:
            self.sim.loss = make_loss_model(self.sim.cfg.loss_rate, self.sim.cfg.burst_length, self.sim.rng)
        if "bandwidth_bps" in kw:
            self.sim.bw_schedule = [(0.0, float(kw["bandwidth_bps"]))]
        if "outages" in kw:
            self.sim.outages = OutageSchedule(kw["outages"])

    def outage_now(self, duration_s: float) -> None:
        self.set_channel(outages=list(self.channel_cfg.outages) + [(self.t, duration_s)])

    # -- one frame ----------------------------------------------------------- #
    def _fits(self, codec: str, share: float = 0.6) -> bool:
        """Would this codec use at most ``share`` of the current link capacity?"""
        nbytes = self._avg_bytes.get(codec)
        if nbytes is None:
            return True
        return (nbytes + 30) * 8 / self.cfg.frame_s <= share * self.sim.bandwidth_at(self.t)

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

    def _send_alert(self, t: float, det: str) -> tuple[float | None, int]:
        cfg, ch = self.cfg, self.sim.cfg
        arrivals = []
        wire = cfg.alert_bytes + ch.header_bytes
        for k in range(max(1, cfg.alert_copies)):
            tk = self.sim.outages.next_up(t + k * cfg.alert_gap_s)
            self.alert_bytes_air += wire
            self.alert_packets += 1
            if not self.sim.loss.lost():
                arrivals.append(tk + wire * 8 / self.sim.bandwidth_at(tk) + ch.base_latency_s + abs(self.sim.rng.normal(0, ch.jitter_s)))
        return (min(arrivals) if arrivals else None), len(arrivals)

    def _transmit(self, fid: int, x: np.ndarray, codec_name: str, t: float, priority=False):
        codec = self.codecs[codec_name]
        enc = codec.encode(x)
        self.enc_ops += enc.encode_ops
        prev = self._avg_bytes.get(codec_name, enc.nbytes)
        self._avg_bytes[codec_name] = 0.8 * prev + 0.2 * enc.nbytes
        res = self.sim.send_segment(enc.blob, fid, t, priority=priority)
        ok = all(r.delivered for r in res)
        arrival = max(r.t_arrival for r in res) if ok else None
        y = None
        if ok:
            y, _ = self.registry.decode(enc.blob)
        return enc, res, ok, arrival, y

    def step(self) -> dict:
        cfg = self.cfg
        chunk = self.source.read(self.n)
        x = self.filter(chunk.data)
        fid = self.frame
        self.t = chunk.t_end
        t = self.t
        self.orig[fid] = x
        if fid == 0:  # calibrate the expected size of each codec once
            for name in ("hifi", "mid", "dwt", cfg.normal_codec):
                self._avg_bytes[name] = float(self.codecs[name].encode(x).nbytes)

        # 1) on-device detection on the clean, unfiltered-by-codec signal
        self.device.push(x, self.n)
        outs = self.device.run()
        alerts = []
        for name, o in outs.items():
            det = self.device._det(name)
            if o.onset:
                arr, n_ok = (None, 0)
                if det.critical:
                    arr, n_ok = self._send_alert(t, name)
                    self.hifi_until = t + cfg.post_event_s
                ev = EventLog(name, t, arr, n_ok)
                self.events.append(ev)
                self.open_events[name] = ev
                alerts.append({"detector": name, "title": det.title, "critical": det.critical, "t": t, "alert_latency_s": None if arr is None else arr - t, "copies_delivered": n_ok})
            if o.offset and name in self.open_events:
                self.open_events.pop(name).t_offset_device = t
            if o.active and det.critical:
                self.hifi_until = t + cfg.post_event_s

        # 2) pre-trigger buffer: re-send the recent past at high fidelity
        backfilled = []
        if cfg.policy == "event" and any(a["critical"] for a in alerts) and self._fits("hifi", 0.25):
            for bid, bx in list(self.recent):
                _, _, ok, _, y = self._transmit(bid, bx, "hifi", t, priority=True)
                if ok:
                    self.recon[bid] = y
                    backfilled.append(bid)
        self.recent.append((fid, x))

        # 3) encode + send this frame
        mode = self._choose(outs)
        enc, res, ok, arrival, y = self._transmit(fid, x, mode, t, priority=mode in ("hifi", "mid"))
        self.recon[fid] = y if ok else np.zeros_like(x)
        latency = (arrival - t) if ok else None

        # 4) receiver-side detection on reconstructed data
        rx = {}
        if cfg.receiver_detection:
            self.receiver.push(y if ok else None, self.n)
            rx = self.receiver.run()
            if ok:
                self._rx_time = max(self._rx_time, arrival)
            for name, o in rx.items():
                if o.onset:
                    ev = next((e for e in reversed(self.events) if e.detector == name and e.t_onset_receiver is None and e.t_onset_device >= t - 60), None)
                    if ev is not None:
                        ev.t_onset_receiver = self._rx_time if ok else None
                    else:
                        self.events.append(EventLog(name, np.nan, None, 0, t_onset_receiver=self._rx_time))

        rec = {
            "frame": fid,
            "t": t,
            "mode": self.codecs[mode].mode_name(),
            "codec": mode,
            "bytes": enc.nbytes,
            "cr": self.raw_bytes_per_frame / enc.nbytes,
            "packets": len(res),
            "attempts": sum(r.attempts for r in res),
            "status": "ok" if ok else "lost",
            "latency_s": latency,
            "prd": prd(x, self.recon[fid]) if ok else 100.0,
            "truth": sorted(chunk.truth),
            "label": chunk.label,
            "device": {n: {"score": o.score, "active": o.active, "label": o.label} for n, o in outs.items()},
            "receiver": {n: {"score": o.score, "active": o.active, "label": o.label} for n, o in rx.items()},
            "alerts": alerts,
            "backfilled": backfilled,
        }
        self.records.append(rec)
        self.frame += 1
        return rec

    def run(self, n_frames: int) -> list[dict]:
        return [self.step() for _ in range(n_frames)]

    # -- summaries ----------------------------------------------------------- #
    def energy_j(self) -> dict:
        e, st = self.ecfg, self.sim.stats
        radio = st.bytes_on_air * 8 * e.tx_energy_per_bit_j + st.attempts * e.tx_energy_per_packet_j
        alert = self.alert_bytes_air * 8 * e.tx_energy_per_bit_j + self.alert_packets * e.tx_energy_per_packet_j
        mcu = (self.enc_ops + self.device.ops) * e.mcu_energy_per_op_j
        return {"radio_j": radio, "alert_j": alert, "compute_j": mcu, "transmitter_j": radio + alert + mcu}

    def raw_reference_j(self) -> float:
        """Energy if every frame so far had been sent raw over an ideal link."""
        e = self.ecfg
        n_pk = -(-(self.raw_bytes_per_frame + 21) // self.sim.cfg.mtu_bytes)
        per = (self.raw_bytes_per_frame + 21 + n_pk * self.sim.cfg.header_bytes) * 8 * e.tx_energy_per_bit_j + n_pk * e.tx_energy_per_packet_j
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
        return {
            "seconds": dur,
            "frames": len(recs),
            "compression_ratio": sum(self.raw_bytes_per_frame for _ in recs) / max(1, sum(r["bytes"] for r in recs)),
            "bitrate_bps": 8 * sum(r["bytes"] for r in recs) / dur,
            "prd_delivered": float(np.mean([r["prd"] for r in ok])) if ok else np.nan,
            "frames_lost": sum(r["status"] != "ok" for r in recs),
            "packet_loss": self.sim.stats.loss_rate,
            "mean_latency_s": float(np.mean([r["latency_s"] for r in ok])) if ok else np.nan,
            "max_latency_s": float(np.max([r["latency_s"] for r in ok])) if ok else np.nan,
            "tx_power_mw": 1e3 * tx_w,
            "raw_tx_power_mw": 1e3 * raw_w,
            "energy_saving_pct": 100 * (1 - tx_w / raw_w) if raw_w else 0.0,
            "battery_h": batt_j / (tx_w + afe_w) / 3600,
            "battery_h_raw": batt_j / (raw_w + afe_w) / 3600,
            **{f"energy_{k}": v for k, v in en.items()},
        }
