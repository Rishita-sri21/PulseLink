"""Waveform, spectral and band-power fidelity metrics."""
from __future__ import annotations

import numpy as np
from scipy import signal

from config import BANDS


def compression_ratio(original_bits: float, compressed_bytes: float) -> float:
    return float(original_bits / (8.0 * compressed_bytes)) if compressed_bytes else float("inf")


def rmse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - x_hat) ** 2)))


def prd(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Classic PRD (%), relative to the signal energy including its mean."""
    return float(100.0 * np.sqrt(np.sum((x - x_hat) ** 2) / (np.sum(x**2) + 1e-12)))


def prdn(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Normalised PRD (%), mean removed: the honest version for EEG."""
    xc = x - x.mean(axis=-1, keepdims=True)
    return float(100.0 * np.sqrt(np.sum((x - x_hat) ** 2) / (np.sum(xc**2) + 1e-12)))


def snr_db(x: np.ndarray, x_hat: np.ndarray) -> float:
    xc = x - x.mean(axis=-1, keepdims=True)
    return float(10 * np.log10(np.sum(xc**2) / (np.sum((x - x_hat) ** 2) + 1e-12)))


def psd(x: np.ndarray, fs: float, seconds: float = 4.0):
    """Welch PSD along the last axis; returns (freqs, power)."""
    nper = min(x.shape[-1], int(seconds * fs))
    return signal.welch(x, fs=fs, nperseg=nper, axis=-1)


def psd_difference_db(x: np.ndarray, x_hat: np.ndarray, fs: float, band=(0.5, 40.0)) -> float:
    """Mean absolute log-spectral distance (dB) inside ``band``."""
    f, p = psd(x, fs)
    _, ph = psd(x_hat, fs)
    sel = (f >= band[0]) & (f <= band[1])
    return float(np.mean(np.abs(10 * np.log10(p[..., sel] + 1e-12) - 10 * np.log10(ph[..., sel] + 1e-12))))


def band_powers(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Absolute band power per channel (uV^2) for each EEG band."""
    f, p = psd(x, fs)
    df = f[1] - f[0]
    return {name: p[..., (f >= lo) & (f < hi)].sum(-1) * df for name, (lo, hi) in BANDS.items()}


def band_power_errors(x: np.ndarray, x_hat: np.ndarray, fs: float) -> dict[str, float]:
    """Relative band-power error |P_hat - P| / P, averaged over channels."""
    bp, bph = band_powers(x, fs), band_powers(x_hat, fs)
    return {k: float(np.mean(np.abs(bph[k] - bp[k]) / (bp[k] + 1e-12))) for k in bp}


def all_metrics(x: np.ndarray, x_hat: np.ndarray, fs: float) -> dict[str, float]:
    out = {
        "prd": prd(x, x_hat),
        "prdn": prdn(x, x_hat),
        "rmse": rmse(x, x_hat),
        "snr_db": snr_db(x, x_hat),
        "psd_diff_db": psd_difference_db(x, x_hat, fs),
    }
    errs = band_power_errors(x, x_hat, fs)
    out.update({f"bp_err_{k}": v for k, v in errs.items()})
    out["bp_err_mean"] = float(np.mean(list(errs.values())))
    return out
