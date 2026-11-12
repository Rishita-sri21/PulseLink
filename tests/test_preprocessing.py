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


def test_mirror_fallback_downloads_and_verifies(tmp_path, monkeypatch):
    import hashlib

    import preprocessing.load_eeg as le

    blobs = {"sleep-cassette/SC4001E0-PSG.edf": b"psg", "sleep-cassette/SC4001EC-Hypnogram.edf": b"hyp"}
    sums = "".join(f"{hashlib.sha256(b).hexdigest()} {k}\n" for k, b in blobs.items())

    def fake_get(url, timeout=30.0):
        if url.startswith("https://down"):
            raise OSError("blocked")
        rel = url.split("/1.0.0/", 1)[1]
        return sums.encode() if rel == "SHA256SUMS.txt" else blobs[rel]

    monkeypatch.setattr(le, "_http_get", fake_get)
    pairs = le._download_from_mirrors((0,), (1,), tmp_path, mirrors=("https://down/1.0.0", "https://up/1.0.0"))
    assert [(p.name, h.name) for p, h in pairs] == [("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf")]
    assert pairs[0][0].read_bytes() == b"psg"

    blobs["sleep-cassette/SC4001E0-PSG.edf"] = b"tampered"
    pairs[0][0].unlink()
    try:
        le._download_from_mirrors((0,), (1,), tmp_path, mirrors=("https://up/1.0.0",))
    except OSError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("checksum mismatch not detected")


def test_local_recordings_respect_subjects_and_nights(tmp_path):
    from preprocessing.load_eeg import find_local_recordings

    for name in ["SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf", "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf",
                 "SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"]:
        (tmp_path / name).write_bytes(b"")
    assert len(find_local_recordings(tmp_path)) == 3
    got = find_local_recordings(tmp_path, subjects=(0,), recording=(2,))
    assert [(p.name, h.name) for p, h in got] == [("SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf")]


def test_split_keeps_each_subject_on_one_side():
    from pipeline import split_indices

    subj = np.repeat([0, 0, 1, 1, 2, 2], 20)  # two nights per subject
    data = {"recording_ids": np.repeat(np.arange(6), 20), "subject_ids": subj}
    tr, ca, te = split_indices(data)
    assert set(subj[te]) == {2}
    assert 2 not in set(subj[tr]) | set(subj[ca])
