"""Loading Sleep-EDF recordings (via MNE) with a synthetic fallback.

Typical use::

    from preprocessing.load_eeg import load_dataset
    ds = load_dataset(subjects=[0], synthetic=None)   # auto: real if available
    ds["segments"]  # (S, C, T) band-passed EEG in microvolts
    ds["labels"]    # (S,) sleep stage per 30 s segment
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import ANNOTATION_TO_STAGE, DATA_DIR, PreprocessConfig
from preprocessing.filtering import bandpass_filter
from preprocessing.segmentation import segment_signal
from preprocessing.synthetic import generate_synthetic_recording

log = logging.getLogger(__name__)

SLEEP_EDF_DIR = DATA_DIR / "sleep-edf"


def fetch_sleep_edf(subjects=(0,), recording=(1,), path: Path | None = None, download: bool = True):
    """Return [(psg_path, hypnogram_path), ...] for the requested subjects.

    Uses ``mne.datasets.sleep_physionet.age.fetch_data`` which downloads from
    PhysioNet on first use (about 50 MB per night).
    """
    import mne

    path = Path(path or SLEEP_EDF_DIR)
    path.mkdir(parents=True, exist_ok=True)
    if not download and not any(path.rglob("*PSG.edf")):
        raise FileNotFoundError(f"no Sleep-EDF files under {path}")
    files = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(subjects), recording=list(recording), path=str(path), on_missing="warn"
    )
    return [(Path(p), Path(h)) for p, h in files]


def find_local_recordings(path: Path | None = None):
    """Pair up any PSG/Hypnogram EDF files already present on disk."""
    path = Path(path or SLEEP_EDF_DIR)
    pairs = []
    for psg in sorted(path.rglob("*-PSG.edf")):
        stem = psg.name[:7]  # e.g. SC4001E -> hypnogram SC4001EC-Hypnogram.edf
        hyps = sorted(psg.parent.glob(f"{stem[:6]}*-Hypnogram.edf"))
        if hyps:
            pairs.append((psg, hyps[0]))
    return pairs


def load_sleep_edf_recording(
    psg_path: Path,
    hyp_path: Path,
    cfg: PreprocessConfig | None = None,
    wake_margin_min: float = 30.0,
) -> dict:
    """Load one Sleep-EDF night: selected EEG channels in uV and 30 s labels.

    Long wake periods at the start/end of the night are cropped to
    ``wake_margin_min`` minutes around sleep, as is standard practice.
    """
    import mne

    cfg = cfg or PreprocessConfig()
    raw = mne.io.read_raw_edf(str(psg_path), include=list(cfg.channels), preload=True, verbose="error")
    if raw.info["sfreq"] != cfg.fs:
        raw.resample(cfg.fs, verbose="error")
    data = raw.get_data(picks=list(cfg.channels)) * 1e6  # V -> uV

    annot = mne.read_annotations(str(hyp_path))
    epoch_len = int(round(30.0 * cfg.fs))
    n_epochs = data.shape[1] // epoch_len
    labels = np.full(n_epochs, -1, dtype=int)
    for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
        stage = ANNOTATION_TO_STAGE.get(desc)
        if stage is None:
            continue
        first = int(round(onset / 30.0))
        last = min(n_epochs, first + int(round(duration / 30.0)))
        labels[first:last] = stage

    sleep_idx = np.where((labels > 0))[0]
    if sleep_idx.size:
        margin = int(wake_margin_min * 2)
        lo = max(0, sleep_idx[0] - margin)
        hi = min(n_epochs, sleep_idx[-1] + margin + 1)
    else:
        lo, hi = 0, n_epochs
    data = data[:, lo * epoch_len : hi * epoch_len]
    labels = labels[lo:hi]
    return {
        "data": data,
        "fs": cfg.fs,
        "channels": list(cfg.channels),
        "labels": labels,
        "artifact_epochs": np.zeros(labels.size, dtype=bool),
        "source": psg_path.name,
    }


def preprocess_recording(rec: dict, cfg: PreprocessConfig | None = None) -> dict:
    """Band-pass filter and segment a recording dict into (S, C, T) windows."""
    cfg = cfg or PreprocessConfig()
    filtered = bandpass_filter(rec["data"], rec["fs"], cfg.l_freq, cfg.h_freq, cfg.filter_order)
    seg_len = cfg.segment_samples
    segments = segment_signal(filtered, seg_len)

    # Map 30 s epoch labels onto segments (segment centre -> epoch).
    epoch_len = int(round(30.0 * rec["fs"]))
    centres = np.arange(segments.shape[0]) * seg_len + seg_len // 2
    epoch_idx = np.minimum(centres // epoch_len, rec["labels"].size - 1)
    labels = rec["labels"][epoch_idx]
    artifacts = rec["artifact_epochs"][epoch_idx]
    keep = labels >= 0  # drop unscored / movement epochs
    return {
        "segments": segments[keep],
        "labels": labels[keep],
        "artifact_segments": artifacts[keep],
        "fs": rec["fs"],
        "channels": rec["channels"],
        "source": rec["source"],
        "filtered": filtered,
    }


def load_dataset(
    subjects=(0,),
    recording=(1,),
    cfg: PreprocessConfig | None = None,
    synthetic: bool | None = None,
    download: bool = False,
    n_synthetic_epochs: int = 240,
    seed: int = 0,
) -> dict:
    """Load preprocessed segments from Sleep-EDF, or synthetic data.

    ``synthetic=None`` (default) uses real data if it is already on disk (or
    ``download=True`` succeeds) and otherwise falls back to synthetic EEG.
    """
    cfg = cfg or PreprocessConfig()
    if synthetic is not True:
        try:
            pairs = fetch_sleep_edf(subjects, recording, download=download) if download else find_local_recordings()
            if not pairs:
                raise FileNotFoundError("no local Sleep-EDF recordings")
            parts = [preprocess_recording(load_sleep_edf_recording(p, h, cfg), cfg) for p, h in pairs]
            out = _concat(parts)
            out["is_synthetic"] = False
            return out
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback
            if synthetic is False:
                raise
            log.warning("Sleep-EDF unavailable (%s); using synthetic EEG", exc)

    parts = [
        preprocess_recording(generate_synthetic_recording(n_synthetic_epochs, cfg, seed=seed + i), cfg)
        for i in range(max(1, len(subjects)))
    ]
    out = _concat(parts)
    out["is_synthetic"] = True
    return out


def _concat(parts: list[dict]) -> dict:
    return {
        "segments": np.concatenate([p["segments"] for p in parts]),
        "labels": np.concatenate([p["labels"] for p in parts]),
        "artifact_segments": np.concatenate([p["artifact_segments"] for p in parts]),
        "fs": parts[0]["fs"],
        "channels": parts[0]["channels"],
        "source": ", ".join(p["source"] for p in parts),
        "recording_ids": np.concatenate([np.full(len(p["labels"]), i) for i, p in enumerate(parts)]),
    }
