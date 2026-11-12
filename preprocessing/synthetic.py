"""Synthetic sleep EEG generator.

Used whenever Sleep-EDF has not been downloaded (offline laptops, CI, tests).
The generator is not a physiological model, but it reproduces the properties
that matter for PulseLink:

* stage-dependent spectra (alpha in wake, theta in N1/REM, spindles and
  K-complexes in N2, high-amplitude delta in N3), so downstream sleep staging
  is a meaningful test of "did compression preserve the information?";
* a 1/f background, two correlated channels, baseline drift and mains noise,
  so the band-pass filter has real work to do;
* occasional artifacts (movement bursts, flat lines) so the signal-quality
  detector and the adaptive controller can be exercised.
"""
from __future__ import annotations

import numpy as np

from config import PreprocessConfig

# Stage transition matrix (W, N1, N2, N3, REM): a rough sleep architecture.
_TRANSITIONS = np.array(
    [
        [0.85, 0.13, 0.01, 0.00, 0.01],
        [0.08, 0.55, 0.32, 0.00, 0.05],
        [0.03, 0.04, 0.80, 0.09, 0.04],
        [0.01, 0.00, 0.13, 0.86, 0.00],
        [0.05, 0.04, 0.03, 0.00, 0.88],
    ]
)

# (amplitude in uV) of narrow-band oscillations per stage.
_STAGE_RHYTHMS = {
    0: {"alpha": (10.0, 20.0), "beta": (20.0, 6.0)},
    1: {"theta": (6.0, 15.0), "alpha": (9.5, 5.0)},
    2: {"theta": (5.5, 10.0), "delta": (2.0, 15.0)},
    3: {"delta": (1.0, 60.0), "delta2": (2.5, 25.0)},
    4: {"theta": (6.5, 15.0), "beta": (18.0, 5.0)},
}


def _pink_noise(rng: np.random.Generator, n: int, fs: float, amp: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
    spectrum[1:] /= freqs[1:] ** 0.9  # EEG background power falls ~1/f^1.8
    spectrum[0] = 0.0
    x = np.fft.irfft(spectrum, n=n)
    return amp * x / (x.std() + 1e-12)


def _oscillation(rng, t, freq, amp):
    # Slowly wandering frequency and amplitude look more EEG-like than a sine.
    fm = freq + 0.3 * np.cumsum(rng.normal(scale=0.02, size=t.size)) / np.sqrt(t.size)
    phase = 2 * np.pi * np.cumsum(fm) * (t[1] - t[0]) + rng.uniform(0, 2 * np.pi)
    envelope = 1.0 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6.28))
    return amp * envelope * np.sin(phase)


def _spindle(rng, fs):
    dur = rng.uniform(0.5, 1.5)
    tt = np.arange(int(dur * fs)) / fs
    win = np.hanning(tt.size)
    return rng.uniform(20, 40) * win * np.sin(2 * np.pi * rng.uniform(12, 14) * tt)


def _k_complex(rng, fs):
    tt = np.arange(int(1.0 * fs)) / fs
    return -rng.uniform(80, 150) * np.sin(np.pi * tt) * np.exp(-3 * tt) + 40 * np.sin(2 * np.pi * tt) * np.exp(-4 * tt)


def generate_hypnogram(n_epochs: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.empty(n_epochs, dtype=int)
    state = 0
    for i in range(n_epochs):
        labels[i] = state
        state = rng.choice(5, p=_TRANSITIONS[state])
    return labels


def generate_synthetic_recording(
    n_epochs: int = 120,
    cfg: PreprocessConfig | None = None,
    seed: int = 0,
    artifact_rate: float = 0.04,
    raw: bool = True,
) -> dict:
    """Generate a multichannel synthetic sleep recording.

    Returns a dict with ``data`` (C, N) in microvolts, ``fs``, ``channels``,
    ``labels`` (one sleep stage per 30 s epoch) and ``artifact_epochs``.
    If ``raw`` is True, drift and 50 Hz mains noise are added (so the data
    looks like an unfiltered recording); otherwise the signal is clean.
    """
    cfg = cfg or PreprocessConfig()
    rng = np.random.default_rng(seed)
    fs = cfg.fs
    epoch_len = int(round(30.0 * fs))
    n_ch = len(cfg.channels)
    labels = generate_hypnogram(n_epochs, rng)
    total = n_epochs * epoch_len
    t_epoch = np.arange(epoch_len) / fs

    # Two latent sources mixed into the channels -> correlated channels.
    n_src = n_ch + 1
    sources = np.zeros((n_src, total))
    artifact_epochs = np.zeros(n_epochs, dtype=bool)
    for e, stage in enumerate(labels):
        sl = slice(e * epoch_len, (e + 1) * epoch_len)
        # Transitional epochs borrow rhythms from the neighbouring stage and
        # every epoch gets its own gain, so stages overlap as in real data.
        neighbour = labels[e - 1] if e > 0 else stage
        mix = rng.uniform(0.2, 0.6) if (neighbour != stage and rng.random() < 0.5) else 0.0
        gain = rng.uniform(0.6, 1.5)
        for s in range(n_src):
            sig = _pink_noise(rng, epoch_len, fs, amp=rng.uniform(8.0, 16.0))
            for _, (freq, amp) in _STAGE_RHYTHMS[stage].items():
                sig += (1 - mix) * _oscillation(rng, t_epoch, freq * rng.uniform(0.85, 1.15), amp * rng.uniform(0.5, 1.3))
            if mix:
                for _, (freq, amp) in _STAGE_RHYTHMS[neighbour].items():
                    sig += mix * _oscillation(rng, t_epoch, freq * rng.uniform(0.85, 1.15), amp * rng.uniform(0.5, 1.3))
            sources[s, sl] = gain * sig
        if stage == 2:  # spindles and K-complexes are the N2 hallmarks
            for _ in range(rng.integers(2, 5)):
                ev = _spindle(rng, fs) if rng.random() < 0.6 else _k_complex(rng, fs)
                start = e * epoch_len + rng.integers(0, epoch_len - ev.size)
                sources[:, start : start + ev.size] += ev * rng.uniform(0.7, 1.0, size=(n_src, 1))
    mixing = rng.uniform(0.3, 1.0, size=(n_ch, n_src))
    mixing /= mixing.sum(axis=1, keepdims=True) / 1.5
    data = mixing @ sources

    # Artifacts.
    for e in range(n_epochs):
        if rng.random() < artifact_rate:
            artifact_epochs[e] = True
            start = e * epoch_len + rng.integers(0, epoch_len // 2)
            length = rng.integers(int(2 * fs), int(10 * fs))
            ch = rng.integers(0, n_ch)
            if rng.random() < 0.5:  # movement burst: large broadband noise
                data[ch, start : start + length] += rng.normal(scale=150.0, size=min(length, total - start))
            else:  # electrode pop / flat line
                data[ch, start : start + length] = data[ch, start]

    if raw:
        t = np.arange(total) / fs
        data += 30.0 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6.28))  # drift
        if fs > 100:  # mains is only representable above Nyquist 50 Hz
            data += 5.0 * np.sin(2 * np.pi * 50.0 * t)
        else:
            data += 2.0 * np.sin(2 * np.pi * 45.0 * t)  # out-of-band interference

    return {
        "data": data.astype(np.float64),
        "fs": fs,
        "channels": list(cfg.channels),
        "labels": labels,
        "artifact_epochs": artifact_epochs,
        "source": f"synthetic(seed={seed})",
    }
