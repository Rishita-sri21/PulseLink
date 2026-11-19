"""Tests for the priority link, notifications, profiles and the new ML parts."""
import numpy as np

from communication.priority_link import P0, P1, P3, LinkConfig, PriorityLink
from config import ChannelConfig
from realtime.engine import StreamEngine, preset
from realtime.notify import NotificationCenter
from realtime.profiles import PROFILES, profile_detectors
from realtime.sources import LiveSynthSource, demo_tour


def _backlogged(priority: bool, **kw):
    link = PriorityLink(ChannelConfig(bandwidth_bps=1600, seed=1), LinkConfig(priority=priority, **kw))
    for k in range(20):  # 1 kB of EEG every 2 s on a 1.6 kbit/s link: the queue grows
        link.advance_to(2.0 * k)
        link.send(2.0 * k, 1000, P3)
    link.advance_to(40.0)
    alert = link.send(40.0, 12, P0, kind="alert", copies=2, gap_s=0.1)
    link.flush(2000)
    return link, link.status(alert)


def test_alert_skips_the_queue_fifo_does_not():
    _, pri = _backlogged(True)
    _, fifo = _backlogged(False, ack_classes=(), deadline_s={})
    assert pri.state == "delivered" and pri.t_arrival - 40.0 < 1.6  # at most one packet in flight ahead of it
    assert fifo.state == "delivered" and fifo.t_arrival - 40.0 > 20.0  # waits behind ~5 kB of EEG


def test_preemption_interrupts_a_data_packet():
    link = PriorityLink(ChannelConfig(bandwidth_bps=800, seed=0), LinkConfig())
    link.send(0.0, 244, P3)  # one 252-byte packet takes ~2.5 s on 800 bit/s
    link.advance_to(0.5)
    m = link.status(link.send(0.5, 12, P0, copies=1))
    link.advance_to(5.0)
    assert link.stats.preempted == 1 and m.t_arrival - 0.5 < 0.3


def test_ack_retries_through_heavy_loss_and_backup_during_outage():
    ch = ChannelConfig(bandwidth_bps=16_000, loss_rate=0.5, seed=3)
    link = PriorityLink(ch, LinkConfig(backup=LinkConfig().backup.__class__(enabled=False)))
    ids = [link.send(float(k), 12, P0, copies=1) for k in range(50) if link.advance_to(float(k)) is None]
    link.flush(500)
    assert np.mean([link.status(i).state == "delivered" for i in ids]) > 0.97  # vs 50 % without ACK
    link = PriorityLink(ChannelConfig(bandwidth_bps=16_000, outages=[(10.0, 30.0)]), LinkConfig())
    link.advance_to(12.0)
    m = link.status(link.send(12.0, 12, P0, copies=2, gap_s=0.1))
    link.advance_to(20.0)
    assert m.path == "backup" and m.t_arrival - 12.0 < 2.0


def test_stale_eeg_is_dropped_not_alerts():
    link = PriorityLink(ChannelConfig(bandwidth_bps=800), LinkConfig(deadline_s={P3: 5.0}))
    for k in range(10):
        link.advance_to(2.0 * k)
        link.send(2.0 * k, 600, P3)
    link.flush(200)
    assert link.stats.stale > 0


def test_notifications_escalate_and_acknowledge():
    nc = NotificationCenter(escalate_after_s=30)
    a = nc.add(10.0, 0, "alert", "seizure", "Seizure")
    b = nc.add(11.0, 0, "alert", "seizure", "Seizure")
    nc.acknowledge(b.id, 12.0)
    nc.tick(45.0)
    assert a.state == "escalated" and b.state == "acknowledged"
    assert nc.add(1, 4, "log", "attention", "Focus").state == "logged"


def test_profiles_make_any_detector_critical():
    drv = {d.name: d for d in profile_detectors("driver")}
    assert drv["drowsiness"].critical and drv["drowsiness"].has_early and not drv["attention"].critical
    assert set(PROFILES) >= {"epilepsy", "driver", "focus", "general"}
    eng = StreamEngine(LiveSynthSource(seed=1, schedule=demo_tour()), None, ChannelConfig(), cfg=preset("pulselink"), detectors=profile_detectors("driver"))
    eng.run(55)
    eng.finish()
    ev = next(e for e in eng.events if e.detector == "drowsiness")
    assert ev.priority == P0 and ev.t_early_arrival is not None and ev.t_early_arrival < ev.t_alert_arrival


def test_stream_baseline_is_slower_on_a_slow_link():
    times = {}
    for name in ("pulselink", "stream_raw"):
        eng = StreamEngine(LiveSynthSource(seed=1, schedule=demo_tour()), None, ChannelConfig(bandwidth_bps=1600, loss_rate=0.02), cfg=preset(name))
        eng.run(95)
        eng.finish()
        ev = next(e for e in eng.events if e.detector == "seizure")
        times[name] = ev.t_first_alert if name == "pulselink" else ev.t_onset_receiver
    assert times["pulselink"] - 160 < 2.5
    assert times["stream_raw"] is None or times["stream_raw"] - 160 > 10


def test_cnn_and_gbm_detectors_learn():
    from realtime.detectors import CNNDetector, LearnedDetector
    from realtime.features import compute_features

    rng = np.random.default_rng(0)
    X = rng.normal(scale=20, size=(240, 2, 400))
    y = rng.random(240) < 0.3
    X[y] += 80 * np.sin(np.arange(400) * 2 * np.pi * 3 / 100)
    cnn = CNNDetector(name="c").fit(X, y, epochs=4)
    p = cnn.clone().predict(X)
    assert p[y].mean() > p[~y].mean() + 0.3
    gbm = LearnedDetector(name="g", kind="gbm").fit([compute_features(x, 100.0) for x in X], y)
    assert gbm.raw_score(compute_features(X[np.argmax(y)], 100.0), {}) > 0.5


def test_task_features_are_differentiable():
    import torch

    from compression.train_streaming import task_features

    x = torch.randn(3, 2, 200, requires_grad=True)
    task_features(x, 100.0).sum().backward()
    assert x.grad is not None and task_features(x, 100.0).shape == (3, 2, 9)


def test_live_app_race_and_ack():
    from realtime.server import LiveApp

    app = LiveApp(model=None, speed=50, profile="epilepsy")
    assert app.target == "seizure" and app.state()["profile"] == "epilepsy"
    app.control({"action": "set", "loss_rate": 0.1, "bandwidth_bps": 3200})
    for k in range(45):
        if k == 30:  # after the baseline is learned
            app.control({"action": "inject", "kind": "seizure"})
        chunk = app.engine.source.read(app.engine.n)
        rec, brec = app.engine.step(chunk), app.baseline.step(chunk)
    p = app._frame_payload(rec, brec)
    assert p["race"]["onset"] is not None and p["race"]["pl_confirmed"] is not None
    assert any(n["priority"] == 0 for n in p["notifications"])
    app.control({"action": "ack"})
    assert not app.engine.notify.open_critical(app.engine.t)
    assert app.control({"action": "profile", "profile": "driver"})["target"] == "drowsiness"
