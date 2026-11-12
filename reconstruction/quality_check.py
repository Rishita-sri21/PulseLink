"""Signal-quality assessment on both sides of the link.

* :func:`assess_input_quality` runs on the transmitter on the *original*
  segment and detects artifacts (flat lines, clipping, movement bursts).
  The adaptive controller uses it to avoid lossy learned compression of
  out-of-distribution segments.
* :func:`assess_reconstruction` runs on the receiver **without** access to
  the original. It combines packet-level evidence (lost / concealed / late),
  the encoder's own PRD estimate carried in the header (closed-loop
  verification), and plausibility checks on the decoded waveform, and
  decides whether a segment should be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from compression.quantization import FLAG_IMPORTANT, FLAG_POOR_QUALITY
from config import QualityConfig
from reconstruction.decoder import ReceivedSegment


@dataclass
class InputQuality:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    score: float = 1.0  # 1 = clean, 0 = unusable


def assess_input_quality(x: np.ndarray, fs: float, flat_seconds: float = 1.0, amp_z: float = 8.0) -> InputQuality:
    """Detect common EEG artifacts in a (C, T) segment (microvolts)."""
    reasons = []
    win = max(2, int(flat_seconds * fs))
    for ch in x:
        # flat line: a window with (almost) no variation
        n = ch.size // win
        if n and (ch[: n * win].reshape(n, win).std(axis=1) < 0.05).any():
            reasons.append("flat_line")
        # amplitude burst relative to a robust (MAD) scale estimate
        mad = np.median(np.abs(ch - np.median(ch))) * 1.4826 + 1e-9
        if np.max(np.abs(ch - np.median(ch))) > amp_z * mad:
            reasons.append("amplitude_burst")
        # clipping: many samples stuck at the extreme value
        peak = np.max(np.abs(ch))
        if peak > 0 and np.mean(np.abs(ch) > 0.999 * peak) > 0.01:
            reasons.append("clipping")
    reasons = sorted(set(reasons))
    return InputQuality(ok=not reasons, reasons=reasons, score=max(0.0, 1.0 - 0.4 * len(reasons)))


@dataclass
class ReliabilityReport:
    segment_id: int
    reliable: bool
    flags: list[str]
    prd_estimate: float | None


def assess_reconstruction(seg: ReceivedSegment, fs: float, cfg: QualityConfig | None = None) -> ReliabilityReport:
    """Decide (without the original) whether a reconstruction is trustworthy."""
    cfg = cfg or QualityConfig()
    flags: list[str] = []
    if seg.status == "lost":
        flags.append("lost")
    elif seg.status == "partial":
        flags.append("concealed_gaps")
    if seg.late:
        flags.append("late")
    prd_est = None
    if seg.header is not None:
        prd_est = seg.header.prd_estimate
        if prd_est is not None and prd_est > cfg.max_prd_percent:
            flags.append("high_distortion")
        if seg.header.flags & FLAG_POOR_QUALITY:
            flags.append("poor_input_signal")
        if seg.header.flags & FLAG_IMPORTANT:
            flags.append("important_event")  # informational, not a failure
    if seg.status != "lost":
        x = seg.signal
        if not np.all(np.isfinite(x)):
            flags.append("implausible_values")
        elif (x.std(axis=-1) < 1e-3).any():
            flags.append("flat_reconstruction")
        else:
            f, p = signal.welch(x, fs=fs, nperseg=min(x.shape[-1], int(4 * fs)), axis=-1)
            inband = (f >= 0.5) & (f <= 40.0)
            ratio = p[:, inband].sum() / (p.sum() + 1e-12)
            if ratio < 0.8:
                flags.append("implausible_spectrum")
    severe = {"lost", "concealed_gaps", "high_distortion", "implausible_values", "flat_reconstruction", "implausible_spectrum"}
    if seg.missing_fraction > 0.2:
        severe.add("concealed_gaps")
    reliable = not (set(flags) & severe)
    return ReliabilityReport(seg.segment_id, reliable, flags, prd_est)
