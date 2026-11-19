"""Cheap per-window EEG features shared by every detector.

Computed once per analysis window (one Welch PSD per channel plus a few
time-domain statistics), so adding detectors costs almost nothing extra on
the wearable. ``feature_ops`` gives an operation count for the energy model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
    "gamma": (30.0, 40.0),
}


@dataclass
class Features:
    fs: float
    n_channels: int
    seconds: float
    band: dict[str, float]  # absolute band power (uV^2), mean over channels
    total_power: float  # 0.5-40 Hz
    line_length: float  # mean |dx/dt| in uV/s, mean over channels
    rms: np.ndarray  # per channel, uV
    max_abs: float  # uV
    flat: bool  # any channel (almost) flat
    channel_asymmetry: float  # max rms / min rms
    rhythmicity: float  # spectral peak / mean power in 1-20 Hz
    peak_freq: float  # Hz, dominant frequency in 1-20 Hz
    extra: dict = field(default_factory=dict)

    def rel(self, name: str) -> float:
        return self.band[name] / (self.total_power + 1e-12)

    def vector(self) -> np.ndarray:
        """Fixed-length feature vector for learned detectors."""
        logs = [np.log(self.band[b] + 1e-6) for b in BANDS]
        rels = [self.rel(b) for b in BANDS]
        return np.array(
            logs + rels + [
                np.log(self.total_power + 1e-6),
                np.log(self.line_length + 1e-6),
                np.log(self.max_abs + 1e-6),
                np.log(self.channel_asymmetry),
                self.rhythmicity,
                self.peak_freq,
                np.log((self.band["theta"] + 1e-6) / (self.band["alpha"] + 1e-6)),
                np.log((self.band["beta"] + 1e-6) / (self.band["alpha"] + self.band["theta"] + 1e-6)),
            ]
        )


FEATURE_NAMES = (
    [f"log_{b}" for b in BANDS] + [f"rel_{b}" for b in BANDS]
    + ["log_total", "log_line_length", "log_max_abs", "log_asymmetry", "rhythmicity", "peak_freq", "log_theta_alpha", "log_engagement"]
)


def compute_features(x: np.ndarray, fs: float) -> Features:
    """Features of one (C, W) window in microvolts."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    c, w = x.shape
    nper = int(min(w, 2 * fs))
    f, p = signal.welch(x, fs=fs, nperseg=nper, axis=-1)
    pm = p.mean(axis=0)
    df = f[1] - f[0]

    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(pm[m].sum() * df) if m.any() else 0.0

    band = {k: bp(*v) for k, v in BANDS.items()}
    total = bp(0.5, 40.0)
    sel = (f >= 1.0) & (f <= 20.0)
    ps = np.convolve(pm[sel], np.ones(3) / 3, mode="same") if sel.sum() >= 3 else pm[sel]
    rhythm = float(ps.max() / (ps.mean() + 1e-12)) if ps.size else 0.0
    peak = float(f[sel][np.argmax(ps)]) if ps.size else 0.0
    rms = np.sqrt(np.mean((x - x.mean(-1, keepdims=True)) ** 2, axis=-1))
    mean_abs_diff = np.abs(np.diff(x, axis=-1)).mean(-1)
    return Features(
        fs=fs,
        n_channels=c,
        seconds=w / fs,
        band=band,
        total_power=total,
        line_length=float(np.mean(mean_abs_diff * fs)),
        rms=rms,
        max_abs=float(np.abs(x - np.median(x, axis=-1, keepdims=True)).max()),
        flat=bool((mean_abs_diff < 0.05).any()),  # stuck electrode / saturation
        channel_asymmetry=float(rms.max() / (rms.min() + 1e-6)) if c > 1 else 1.0,
        rhythmicity=rhythm,
        peak_freq=peak,
    )


def feature_ops(n_channels: int, n_samples: int, fs: float) -> int:
    """Rough MCU op count: Welch FFTs (~5 N log2 N per segment) + stats."""
    nper = int(min(n_samples, 2 * fs))
    n_seg = max(1, 2 * n_samples // nper - 1)
    return int(n_channels * (n_seg * 5 * nper * np.log2(max(nper, 2)) + 10 * n_samples))
