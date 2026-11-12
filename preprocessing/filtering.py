"""Band-pass filtering for EEG."""
from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter applied along the last axis.

    ``h_freq`` is clipped below Nyquist so the filter is valid at any rate
    (Sleep-EDF EEG is sampled at 100 Hz, so Nyquist is 50 Hz).
    """
    nyq = fs / 2.0
    high = min(h_freq, 0.95 * nyq)
    if not 0 < l_freq < high:
        raise ValueError(f"invalid band {l_freq}-{high} Hz for fs={fs}")
    sos = signal.butter(order, [l_freq, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1)


def notch_filter(data: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Optional mains notch filter (skipped if ``freq`` is above Nyquist)."""
    if freq >= fs / 2:
        return data
    b, a = signal.iirnotch(freq, q, fs=fs)
    return signal.filtfilt(b, a, data, axis=-1)
