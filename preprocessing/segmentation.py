"""Segmentation into fixed windows and per-segment normalisation."""
from __future__ import annotations

import numpy as np


def segment_signal(data: np.ndarray, segment_samples: int, hop: int | None = None) -> np.ndarray:
    """Split (C, N) data into (S, C, T) windows. Trailing samples are dropped."""
    hop = hop or segment_samples
    n = data.shape[-1]
    if n < segment_samples:
        return np.empty((0, data.shape[0], segment_samples))
    starts = range(0, n - segment_samples + 1, hop)
    return np.stack([data[:, s : s + segment_samples] for s in starts])


def normalize_segment(x: np.ndarray, eps: float = 1e-6):
    """Per-channel z-score of one (C, T) segment.

    Returns (normalised, mean, std); mean/std are sent as side information so
    the receiver can restore physical units.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + eps
    return (x - mean) / std, mean.squeeze(-1), std.squeeze(-1)


def denormalize_segment(xn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return xn * std[:, None] + mean[:, None]


def normalize_segments(segments: np.ndarray, eps: float = 1e-6):
    """Vectorised per-segment, per-channel z-score of (S, C, T) segments."""
    mean = segments.mean(axis=-1, keepdims=True)
    std = segments.std(axis=-1, keepdims=True) + eps
    return (segments - mean) / std, mean.squeeze(-1), std.squeeze(-1)
