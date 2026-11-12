"""Energy-aware, rule-based adaptive compression controller.

The controller owns a *ladder* of codec modes (raw, several DWT settings,
several PulseLink autoencoder widths/bit-depths). The ladder is calibrated
once on held-out data: every mode gets an expected size, an expected
transmitter energy, and a small ridge-regression **distortion predictor**
that estimates the PRD a given segment would suffer under that mode from
cheap spectral features (one FFT, ~0.1 M ops, far cheaper than decoding).
Modes that are bigger *and* worse than another mode are pruned.

Per segment, transparent rules decide:

======================  ====================================================
Condition               Behaviour
======================  ====================================================
Default                 Cheapest-energy mode whose *predicted* PRD meets the
                        quality target (``target_prd``)
Good connectivity       Link could almost carry raw -> stricter target
                        (``good_link_prd``) = lower compression ratio
Limited bandwidth       Budget (bandwidth x segment length, minus buffer
                        backlog) excludes big modes -> higher compression;
                        if nothing meets the target, best mode that fits
Outage / backlog        Budget shrinks so the buffer drains after reconnect
Poor signal quality     Learned (AE) modes excluded: the autoencoder was
                        trained on clean EEG and may hallucinate on
                        artifacts; generic DWT/raw is used and flagged
Important event         Stricter target (``important_prd``) and up to
                        ``important_borrow`` x budget; flagged as important
======================  ====================================================

The predicted PRD is written into the segment header so the receiver can
flag unreliable reconstructions without ever seeing the original.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge

from compression.codecs import CODEC_AE, CODEC_RAW, Codec, EncodedSegment, prd
from compression.quantization import FLAG_IMPORTANT, FLAG_POOR_QUALITY
from config import EnergyConfig
from reconstruction.quality_check import assess_input_quality


# --------------------------------------------------------------------------- #
# Cheap per-segment features + distortion predictor
# --------------------------------------------------------------------------- #
_FEATURE_BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 15), (15, 30), (30, 45)]


def segment_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Features that predict compressibility: spectral shape and complexity."""
    spec = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
    tot = spec[:, f >= 0.5].sum(-1) + 1e-12
    rel = np.stack([spec[:, (f >= lo) & (f < hi)].sum(-1) / tot for lo, hi in _FEATURE_BANDS], axis=-1)
    std = x.std(-1) + 1e-9
    line_len = np.abs(np.diff(x, axis=-1)).mean(-1) / std  # waveform complexity
    kurt = ((x - x.mean(-1, keepdims=True)) ** 4).mean(-1) / std**4
    per_ch = np.column_stack([rel, line_len, np.log(kurt), np.log(std)])
    return np.concatenate([per_ch.mean(0), per_ch.max(0)])


def feature_ops(n_channels: int, n_samples: int) -> int:
    """FFT (~5 N log2 N) plus a few passes over the data, per channel."""
    return int(n_channels * (5 * n_samples * np.log2(n_samples) + 10 * n_samples))


class DistortionPredictor:
    """Per-mode ridge regression on log(1 + PRD)."""

    def __init__(self):
        self.models: dict[str, Ridge] = {}
        self.mu = self.sd = None

    def fit(self, feats: np.ndarray, prds: dict[str, np.ndarray]) -> "DistortionPredictor":
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-9
        z = (feats - self.mu) / self.sd
        for name, y in prds.items():
            self.models[name] = Ridge(alpha=1.0).fit(z, np.log1p(y))
        return self

    def predict(self, feats: np.ndarray, name: str) -> float:
        if name not in self.models:
            return float("nan")
        z = (feats - self.mu) / self.sd
        return float(np.expm1(self.models[name].predict(z[None])[0]))


# --------------------------------------------------------------------------- #
# Ladder calibration
# --------------------------------------------------------------------------- #
@dataclass
class LadderMode:
    codec: Codec
    mean_bytes: float
    mean_prd: float
    mean_encode_ops: float = 0.0

    @property
    def name(self) -> str:
        return self.codec.mode_name()

    @property
    def learned(self) -> bool:
        return self.codec.codec_id == CODEC_AE

    def tx_energy_j(self, ecfg: EnergyConfig, mtu: int, header: int) -> float:
        n_pk = max(1, -(-int(self.mean_bytes) // mtu))
        return (self.mean_bytes + n_pk * header) * 8 * ecfg.tx_energy_per_bit_j + n_pk * ecfg.tx_energy_per_packet_j + self.mean_encode_ops * ecfg.mcu_energy_per_op_j


@dataclass
class Ladder:
    modes: list[LadderMode]
    predictor: DistortionPredictor | None = None

    def __iter__(self):
        return iter(self.modes)

    def __len__(self):
        return len(self.modes)

    def __getitem__(self, i):
        return self.modes[i]


def calibrate_ladder(codecs: list[Codec], segments: np.ndarray, fs: float = 100.0, prune: bool = True) -> Ladder:
    """Measure size/PRD/ops of each mode, fit the predictor, prune dominated modes."""
    modes, prds = [], {}
    feats = np.stack([segment_features(x, fs) for x in segments])
    for codec in codecs:
        sizes, p, ops = [], [], []
        for x in segments:
            enc = codec.encode(x)
            sizes.append(enc.nbytes)
            ops.append(enc.encode_ops)
            p.append(prd(x, codec.decode(enc.blob)))
        modes.append(LadderMode(codec, float(np.mean(sizes)), float(np.mean(p)), float(np.mean(ops))))
        prds[codec.mode_name()] = np.asarray(p)
    predictor = DistortionPredictor().fit(feats, prds) if len(segments) >= 4 else None
    modes.sort(key=lambda m: m.mean_bytes)
    if prune:
        front, best = [], np.inf
        for m in modes:  # increasing size: keep only modes that lower distortion
            if m.mean_prd < best - 1e-9:
                front.append(m)
                best = m.mean_prd
        # Always keep the cheapest non-learned lossy mode as artifact fallback.
        fallback = next((m for m in modes if not m.learned and m.codec.codec_id != CODEC_RAW), None)
        if fallback is not None and fallback not in front:
            front.append(fallback)
        modes = sorted(front, key=lambda m: m.mean_bytes)
    return Ladder(modes, predictor)


# --------------------------------------------------------------------------- #
# Event detection
# --------------------------------------------------------------------------- #
def detect_important(x: np.ndarray, fs: float, sigma_thresh: float = 0.05, kc_uv: float = 75.0) -> list[str]:
    """Flag segments with sleep spindles or K-complex-like transients."""
    events = []
    f, p = signal.welch(x, fs=fs, nperseg=min(x.shape[-1], int(4 * fs)), axis=-1)
    tot = p[:, (f >= 0.5) & (f <= 30)].sum(-1) + 1e-12
    sigma = p[:, (f >= 12) & (f < 15)].sum(-1) / tot
    delta = p[:, (f >= 0.5) & (f < 4)].sum(-1) / tot
    if (sigma > sigma_thresh).any():
        events.append("spindle_activity")
    sos = signal.butter(2, [0.5, 2.0], btype="bandpass", fs=fs, output="sos")
    slow = signal.sosfiltfilt(sos, x, axis=-1)
    if ((np.abs(slow).max(-1) > kc_uv) & (delta < 0.8)).any():
        events.append("k_complex")
    return events


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    mode: LadderMode
    encoded: EncodedSegment
    budget_bytes: float
    predicted_prd: float
    target_prd: float
    reasons: list[str] = field(default_factory=list)
    input_issues: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    extra_encode_ops: int = 0  # feature extraction + any rejected encodings


class AdaptiveController:
    def __init__(
        self,
        ladder: Ladder | list[LadderMode],
        fs: float,
        segment_seconds: float,
        energy_cfg: EnergyConfig | None = None,
        mtu: int = 244,
        header_bytes: int = 8,
        target_prd: float = 15.0,
        good_link_prd: float = 10.0,
        important_prd: float = 8.0,
        max_prd: float = 30.0,
        important_borrow: float = 2.0,
        utilisation: float = 0.85,
        drain_seconds: float = 60.0,
        verify: bool = False,
        retries: int = 0,
    ):
        if not isinstance(ladder, Ladder):
            ladder = Ladder(list(ladder))
        self.ladder = ladder
        self.modes = sorted(ladder.modes, key=lambda m: m.mean_bytes)
        self.fs, self.segment_seconds = fs, segment_seconds
        self.ecfg = energy_cfg or EnergyConfig()
        self.mtu, self.header_bytes = mtu, header_bytes
        self.target_prd, self.good_link_prd, self.important_prd = target_prd, good_link_prd, important_prd
        self.max_prd, self.important_borrow = max_prd, important_borrow
        self.utilisation, self.drain_seconds = utilisation, drain_seconds
        self.verify = verify
        self.retries = retries
        raw = [m for m in self.modes if m.codec.codec_id == CODEC_RAW]
        self.raw_bytes = raw[0].mean_bytes if raw else self.modes[-1].mean_bytes

    def budget(self, bandwidth_bps: float, backlog_bytes: float, link_up: bool, loss_rate: float) -> float:
        """Payload bytes this segment may use so the transmit queue stays stable."""
        overhead = 1 + self.header_bytes / self.mtu
        capacity = bandwidth_bps / 8.0 * self.segment_seconds * self.utilisation * (1.0 - min(loss_rate, 0.5)) / overhead
        b = capacity - backlog_bytes * self.segment_seconds / self.drain_seconds
        if not link_up:
            b *= 0.5  # disconnected: the buffer is filling, be conservative
        return max(b, 0.0)

    def predicted_prd(self, mode: LadderMode, feats: np.ndarray | None) -> float:
        if mode.codec.codec_id == CODEC_RAW:
            return 0.0
        if feats is not None and self.ladder.predictor is not None:
            p = self.ladder.predictor.predict(feats, mode.name)
            if np.isfinite(p):
                return max(0.0, p)
        return mode.mean_prd

    def decide(self, x: np.ndarray, bandwidth_bps: float, backlog_bytes: float = 0.0, link_up: bool = True, loss_rate: float = 0.0) -> Decision:
        c, t = x.shape
        budget = self.budget(bandwidth_bps, backlog_bytes, link_up, loss_rate)
        quality = assess_input_quality(x, self.fs)
        events = detect_important(x, self.fs) if quality.ok else []
        feats = segment_features(x, self.fs)
        extra_ops = feature_ops(c, t)
        reasons, flags = [], 0

        candidates = list(self.modes)
        if not quality.ok:
            candidates = [m for m in candidates if not m.learned] or candidates
            flags |= FLAG_POOR_QUALITY
            reasons.append(f"poor signal quality ({', '.join(quality.reasons)}) -> generic codec, no learned compression")
        allowed, target = budget, self.target_prd
        if budget >= 0.5 * self.raw_bytes:
            target = self.good_link_prd
            reasons.append("good connectivity -> stricter quality target (lower compression)")
        if events:
            flags |= FLAG_IMPORTANT
            allowed, target = budget * self.important_borrow, min(target, self.important_prd)
            reasons.append(f"important event ({', '.join(events)}) -> preserve with stricter target")

        pred = {m.name: self.predicted_prd(m, feats) for m in candidates}
        # Loss-aware: a segment is only decodable if all its packets arrive,
        # so big segments are fragile on lossy links. Expected distortion
        # counts an undecodable segment as 100 % PRD.
        p_fail = min(loss_rate, 0.99) ** (self.retries + 1)
        def p_ok(m):
            return (1.0 - p_fail) ** max(1, -(-int(m.mean_bytes) // self.mtu))
        expected = {m.name: p_ok(m) * pred[m.name] + (1 - p_ok(m)) * 100.0 for m in candidates}
        if p_fail > 1e-3:
            reasons.append(f"lossy link (~{100 * loss_rate:.0f}% loss) -> favour fewer packets")
        fitting = [m for m in candidates if m.mean_bytes <= allowed]
        meets = [m for m in fitting if expected[m.name] <= target]
        if meets:
            choice = min(meets, key=lambda m: m.tx_energy_j(self.ecfg, self.mtu, self.header_bytes))
            reasons.append(f"lowest-energy mode meeting PRD target {target:g}%")
        elif fitting:
            choice = min(fitting, key=lambda m: expected[m.name])
            reasons.append("limited bandwidth -> increased compression (target not reachable)")
        else:
            choice = candidates[0]
            reasons.append("bandwidth below cheapest mode -> maximum compression")

        enc = choice.codec.encode(x, flags=flags, verify=self.verify, prd_hint=pred[choice.name])
        return Decision(choice, enc, budget, pred[choice.name], target, reasons, quality.reasons, events, extra_ops)
