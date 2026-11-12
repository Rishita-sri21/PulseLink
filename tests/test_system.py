"""Adaptive control, quality flagging, metrics, energy and the full pipeline."""
from dataclasses import replace

import numpy as np

from compression.adaptive import AdaptiveController, detect_important
from compression.quantization import FLAG_POOR_QUALITY, parse_segment
from evaluation.energy_model import energy, savings_percent, segment_energy
from evaluation.signal_metrics import all_metrics, band_power_errors, prdn, psd_difference_db
from pipeline import run_stream
from reconstruction.quality_check import assess_input_quality, assess_reconstruction
from reconstruction.decoder import ReceivedSegment


def test_metrics_zero_for_perfect_reconstruction(segment):
    m = all_metrics(segment, segment.copy(), 100.0)
    assert m["prdn"] < 1e-9 and m["psd_diff_db"] < 1e-6 and m["bp_err_mean"] < 1e-9


def test_metrics_detect_distortion(segment, rng):
    noisy = segment + rng.normal(scale=segment.std(), size=segment.shape)
    assert 80 < prdn(segment, noisy) < 120
    assert psd_difference_db(segment, noisy, 100.0) > 1.0
    assert max(band_power_errors(segment, noisy, 100.0).values()) > 0.1


def test_energy_model_is_transparent():
    e = energy(bytes_on_air=1000, attempts=4, encode_ops=1e6, decode_ops=1e7)
    assert np.isclose(e.transmission_j, 1000 * 8 * 50e-9 + 4 * 5e-6)
    assert np.isclose(e.compression_j, 1e6 * 0.1e-9)
    assert np.isclose(e.net_j, e.transmission_j + e.compression_j + e.reconstruction_j)
    raw = segment_energy(12017, 12000, 12000)
    small = segment_energy(800, 600_000, 600_000)
    assert savings_percent(small.transmitter_j, raw.transmitter_j) > 80


def test_input_quality_detects_artifacts(segment):
    assert assess_input_quality(segment, 100.0).ok
    flat = segment.copy()
    flat[0, 500:900] = flat[0, 500]
    assert "flat_line" in assess_input_quality(flat, 100.0).reasons
    burst = segment.copy()
    burst[1, 1000:1100] += 2000
    assert "amplitude_burst" in assess_input_quality(burst, 100.0).reasons


def test_spindle_detection():
    fs, t = 100.0, np.arange(3000) / 100.0
    from preprocessing.synthetic import _pink_noise

    rng = np.random.default_rng(0)
    x = np.stack([_pink_noise(rng, 3000, fs, 15.0) for _ in range(2)])  # EEG-like 1/f background
    assert "spindle_activity" not in detect_important(x, fs)
    x[:, 1000:1150] += 40 * np.sin(2 * np.pi * 13 * t[1000:1150]) * np.hanning(150)
    x[:, 2000:2100] += 40 * np.sin(2 * np.pi * 13 * t[2000:2100]) * np.hanning(100)
    assert "spindle_activity" in detect_important(x, fs)


def test_receiver_flags_without_original(segment):
    lost = ReceivedSegment(0, np.zeros_like(segment), "lost", None, None, None, False, 0, 1.0)
    assert not assess_reconstruction(lost, 100.0).reliable
    ok = ReceivedSegment(1, segment, "ok", None, 1.0, 0.5, False, 0, 0.0)
    assert assess_reconstruction(ok, 100.0).reliable


def test_ladder_is_pareto_sorted(prep):
    modes = prep.ladder.modes
    sizes = [m.mean_bytes for m in modes]
    assert sizes == sorted(sizes)
    assert any(m.codec.codec_id == 0 for m in modes)  # raw stays as top mode
    assert prep.ladder.predictor is not None


def test_controller_compresses_harder_when_bandwidth_drops(prep):
    X, _ = prep.split("test")
    ctl = AdaptiveController(prep.ladder, prep.fs, 30.0)
    good = ctl.decide(X[0], bandwidth_bps=250_000)
    poor = ctl.decide(X[0], bandwidth_bps=400)
    assert poor.encoded.nbytes < good.encoded.nbytes
    assert poor.encoded.nbytes <= max(poor.budget_bytes, prep.ladder.modes[0].mean_bytes * 1.5)


def test_controller_avoids_learned_codec_on_artifacts(prep):
    X, _ = prep.split("test")
    x = X[0].copy()
    x[0, 100:600] = x[0, 100]  # flat line
    d = AdaptiveController(prep.ladder, prep.fs, 30.0).decide(x, bandwidth_bps=800)
    assert not d.mode.learned
    header, _ = parse_segment(d.encoded.blob)
    assert header.flags & FLAG_POOR_QUALITY


def test_full_stream_ideal_channel(prep):
    X, y = prep.split("test")
    X, y = X[:8], y[:8]
    for method in ("raw", "dwt", "pulselink", "adaptive"):
        r = run_stream(prep, X, method, labels=y)
        assert r.reconstructed.shape == X.shape
        assert r.summary["segments_lost"] == 0
        assert 0 <= r.summary["stage_accuracy"] <= 1
    raw = run_stream(prep, X, "raw").summary
    dwt = run_stream(prep, X, "dwt").summary
    assert raw["prdn_mean_all"] < 1.0
    assert dwt["compression_ratio"] > 8
    assert dwt["energy_transmitter_j"] < raw["energy_transmitter_j"]


def test_stream_under_loss_flags_lost_segments(prep):
    X, _ = prep.split("test")
    ch = replace(prep.cfg.channel, loss_rate=0.3, seed=2)
    r = run_stream(prep, X[:10], "dwt", ch)
    lost = r.segments["status"] == "lost"
    assert lost.any()
    assert not r.segments.loc[lost, "reliable"].any()  # every lost segment is flagged
