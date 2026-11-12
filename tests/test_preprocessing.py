import numpy as np
from scipy import signal

from config import PreprocessConfig
from preprocessing.filtering import bandpass_filter
from preprocessing.load_eeg import load_dataset
from preprocessing.segmentation import denormalize_segment, normalize_segment, segment_signal
from preprocessing.synthetic import generate_synthetic_recording


def test_synthetic_recording_shape_and_labels():
    rec = generate_synthetic_recording(10, seed=0)
    assert rec["data"].shape == (2, 10 * 3000)
    assert rec["labels"].shape == (10,)
    assert set(rec["labels"]) <= set(range(5))


def test_bandpass_removes_out_of_band_energy():
    fs = 100.0
    t = np.arange(3000) / fs
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 0.05 * t) + np.sin(2 * np.pi * 47 * t)
    y = bandpass_filter(x[None], fs, 0.5, 40.0)[0]
    f, p = signal.welch(y, fs=fs, nperseg=1000)
    assert p[np.argmin(abs(f - 10))] > 100 * p[np.argmin(abs(f - 47))]
    assert abs(y.mean()) < 0.05  # drift gone


def test_segmentation_and_normalisation_roundtrip():
    x = np.random.default_rng(0).normal(size=(2, 10_000)) * 30 + 5
    segs = segment_signal(x, 3000)
    assert segs.shape == (3, 2, 3000)
    xn, m, s = normalize_segment(segs[0])
    assert np.allclose(xn.mean(-1), 0, atol=1e-9) and np.allclose(xn.std(-1), 1, atol=1e-3)
    assert np.allclose(denormalize_segment(xn, m, s), segs[0])


def test_load_dataset_falls_back_to_synthetic(tmp_path, monkeypatch):
    import preprocessing.load_eeg as le

    monkeypatch.setattr(le, "SLEEP_EDF_DIR", tmp_path)
    ds = load_dataset(synthetic=None, n_synthetic_epochs=12)
    assert ds["is_synthetic"]
    cfg = PreprocessConfig()
    assert ds["segments"].shape[1:] == (len(cfg.channels), cfg.segment_samples)
    assert len(ds["labels"]) == len(ds["segments"])
