"""Downstream task: sleep-stage classification from band-power features.

The classifier is trained on *original* EEG and then evaluated on the
original and on each method's reconstruction of the same test segments. A
small accuracy drop means the compression kept the clinically relevant
information, not just the waveform's appearance.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from config import BANDS


def band_features(segments: np.ndarray, fs: float) -> np.ndarray:
    """(S, C, T) -> (S, F): log absolute + relative band power per channel,
    plus spectral edge frequency and log signal variance."""
    f, p = signal.welch(segments, fs=fs, nperseg=min(segments.shape[-1], int(4 * fs)), axis=-1)
    total = p[..., (f >= 0.5) & (f <= 30)].sum(-1) + 1e-12
    feats = []
    for lo, hi in BANDS.values():
        bp = p[..., (f >= lo) & (f < hi)].sum(-1)
        feats += [np.log10(bp + 1e-12), bp / total]
    cum = np.cumsum(p, axis=-1) / (p.sum(-1, keepdims=True) + 1e-12)
    sef95 = f[np.argmax(cum >= 0.95, axis=-1)]
    feats += [sef95, np.log10(segments.var(-1) + 1e-12)]
    return np.concatenate([np.asarray(x).reshape(len(segments), -1) for x in feats], axis=1)


class SleepStager:
    def __init__(self, fs: float, seed: int = 0):
        self.fs = fs
        self.clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)

    def fit(self, segments: np.ndarray, labels: np.ndarray) -> "SleepStager":
        self.clf.fit(band_features(segments, self.fs), labels)
        return self

    def predict(self, segments: np.ndarray) -> np.ndarray:
        return self.clf.predict(band_features(segments, self.fs))

    def score(self, segments: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        pred = self.predict(segments)
        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", labels=np.unique(labels), zero_division=0)),
            "kappa": float(cohen_kappa_score(labels, pred)) if len(np.unique(np.r_[labels, pred])) > 1 else 1.0,
        }
