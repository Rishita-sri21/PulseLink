"""Tests for the real-time layer (streaming engine, detectors, live app)."""
import numpy as np
import pytest

from config import ChannelConfig
from realtime.detectors import REGISTRY, LearnedDetector, default_detectors
from realtime.engine import EngineConfig, StreamEngine, StreamingBandpass, preset
from realtime.experiments import alert_delivery_experiment, score_events, truth_intervals
from realtime.features import compute_features
from realtime.sources import LiveSynthSource, RecordingSource, demo_tour, parse_chbmit_summary


def run_tour(policy="event", model=None, frames=125, finish=True, **ch):
    eng = StreamEngine(LiveSynthSource(seed=1, schedule=demo_tour()), model, ChannelConfig(**ch), cfg=EngineConfig(policy=policy))
    eng.run(frames)
    if finish:
        eng.finish()
    return eng


def test_features_shapes_and_flatline():
    rng = np.random.default_rng(0)
    x = rng.normal(scale=20, size=(2, 400))
    f = compute_features(x, 100.0)
    assert f.total_power > 0 and not f.flat
    assert f.vector().shape == (23,)
    x[0] = 5.0
    assert compute_features(x, 100.0).flat


def test_streaming_filter_is_continuous():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 1000))
    whole = StreamingBandpass(100.0, 2)(x)
    filt = StreamingBandpass(100.0, 2)
    parts = np.concatenate([filt(x[:, i : i + 200]) for i in range(0, 1000, 200)], axis=1)
    assert np.allclose(whole, parts)


def test_tour_detects_every_use_case():
    eng = run_tour()
    rows = {r["detector"]: r for r in score_events(eng) if r["side"] == "wearable"}
    for name in ("seizure", "drowsiness", "attention", "artifact"):
        assert rows[name]["events"] >= 1
        assert rows[name]["detected"] == rows[name]["events"], name
        assert rows[name]["false_alarms"] == 0, name
    # seizure is critical: an alert packet went out and arrived quickly
    sz = [e for e in eng.events if e.detector == "seizure" and np.isfinite(e.t_onset_device)]
    assert sz and sz[0].t_alert_arrival is not None
    assert sz[0].t_alert_arrival - sz[0].t_onset_device < 0.2
    # two-stage: the early warning arrived seconds before the confirmation
    assert sz[0].t_early_arrival is not None and sz[0].t_early_arrival < sz[0].t_alert_arrival - 1.0
    assert sz[0].t_early_arrival - 160.0 < 2.5


def test_event_policy_switches_to_high_fidelity_and_backfills():
    eng = run_tour("event")
    codecs = {r["codec"] for r in eng.records}
    assert "hifi" in codecs and "dwt" in codecs  # no model -> dwt is the normal codec
    assert any("backfilled_prd" in r for r in eng.records)
    sz = [r["prd"] for r in eng.records if "seizure" in r["truth"] and r["codec"] == "hifi"]
    low = [r["prd"] for r in eng.records if r["codec"] == "dwt"]
    assert np.mean(sz) < np.mean(low)


def test_policies_cost_ordering():
    raw = run_tour("raw", frames=40).summary()
    dwt = run_tour("dwt", frames=40).summary()
    assert dwt["compression_ratio"] > 4 * raw["compression_ratio"]
    assert dwt["energy_saving_pct"] > 50 > raw["energy_saving_pct"]
    assert dwt["battery_h"] > raw["battery_h"]


def test_with_autoencoder(prep):
    eng = run_tour("pulselink", prep.model, frames=20)
    assert {r["codec"] for r in eng.records} == {"pulselink"}
    assert eng.summary()["compression_ratio"] > 3


def test_outage_alert_goes_via_backup_radio():
    src = lambda: LiveSynthSource(seed=2, schedule=[(60.0, 30.0, "seizure")])
    ch = ChannelConfig(loss_rate=0.1, outages=[(58.0, 20.0)])
    eng = StreamEngine(src(), None, ch, cfg=preset("+ack"))  # everything but the backup radio
    eng.run(50)
    eng.finish()
    assert any(r["status"] == "lost" for r in eng.records)
    ev = next(e for e in eng.events if e.detector == "seizure")
    assert ev.t_alert_arrival >= 78.0  # held until the link is back
    assert ev.t_local_alarm is not None  # ... so the wearable alarmed locally
    assert any(60 < a < 68 for a, b in eng.offline_log)  # receiver noticed the silence
    eng = StreamEngine(src(), None, ch, cfg=preset("pulselink"))
    eng.run(50)
    eng.finish()
    ev = next(e for e in eng.events if e.detector == "seizure")
    assert ev.alert_path == "backup" and ev.t_alert_arrival - ev.t_onset_device < 3.0


def test_live_controls_change_the_channel():
    eng = run_tour(frames=5, finish=False)
    eng.set_channel(loss_rate=0.3, bandwidth_bps=1600.0)
    assert eng.sim.bandwidth_at(eng.t) == 1600.0
    eng.source.inject("seizure")
    eng.run(10)
    assert any("seizure" in r["truth"] for r in eng.records)


def test_learned_detector_and_registry():
    src = LiveSynthSource(seed=3, schedule=[(30.0, 30.0, "seizure"), (100.0, 30.0, "seizure")])
    feats, y = [], []
    filt = StreamingBandpass(100.0, 2)
    for _ in range(80):
        ch = src.read(200)
        feats.append(compute_features(filt(ch.data), 100.0))
        y.append("seizure" in ch.truth)
    det = LearnedDetector(name="sz", critical=True).fit(feats, y)
    p = [det.raw_score(f, {}) for f in feats]
    assert np.mean([a for a, b in zip(p, y) if b]) > np.mean([a for a, b in zip(p, y) if not b])
    clone = det.clone()
    assert not clone.active and clone.raw_score(feats[40], {}) == pytest.approx(p[40])
    assert set(REGISTRY) >= {"seizure", "drowsiness", "attention", "artifact", "sleep"}
    assert len(default_detectors()) == 4


def test_recording_source_truth_and_scoring():
    fs = 100.0
    data = np.random.default_rng(0).normal(scale=10, size=(2, int(120 * fs)))
    src = RecordingSource(data, fs, {"seizure": [(30.0, 50.0)]}, loop=False)
    eng = StreamEngine(src, None, ChannelConfig(), cfg=EngineConfig(policy="dwt"))
    eng.run(59)
    tr = truth_intervals(eng.records, 2.0)
    assert tr["seizure"] == [(30.0, 50.0)]
    with pytest.raises(StopIteration):
        eng.run(5)


def test_chbmit_summary_parser():
    text = """File Name: chb01_03.edf
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb01_04.edf
Number of Seizures in File: 0

File Name: chb06_01.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 1724 seconds
Seizure 1 End Time: 1738 seconds
Seizure 2 Start Time: 7461 seconds
Seizure 2 End Time: 7476 seconds
"""
    sz = parse_chbmit_summary(text)
    assert sz["chb01_03.edf"] == [(2996.0, 3036.0)]
    assert sz["chb01_04.edf"] == []
    assert sz["chb06_01.edf"] == [(1724.0, 1738.0), (7461.0, 7476.0)]


def test_alert_redundancy_helps():
    df = alert_delivery_experiment(losses=(0.2,), copies=(1, 3), bursts=(1.0, 4.0), spacings=(1, 8), trials=3000)
    get = lambda b, s, n: df[(df.burst_length == b) & (df.spacing_packets == s) & (df.copies == n)]["p_alert_lost"].item()
    assert get(1.0, 1, 3) < get(1.0, 1, 1) / 5
    assert get(4.0, 8, 3) < get(4.0, 1, 3)  # spacing beats bursts


def test_live_app_state_and_control():
    from realtime.server import LiveApp

    app = LiveApp(model=None, speed=50)
    st = app.state()
    assert st["source"] == "tour" and "pulselink" not in st["policies"]
    st = app.control({"action": "set", "policy": "raw", "loss_rate": 0.1, "bandwidth_bps": 3200})
    assert st["policy"] == "raw" and st["channel"]["loss_rate"] == 0.1
    app.control({"action": "inject", "kind": "seizure"})
    app.control({"action": "outage", "seconds": 5})
    st = app.control({"action": "source", "source": "free"})
    assert st["source"] == "free" and st["policy"] == "raw"
