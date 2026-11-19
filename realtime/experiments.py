"""Real-time experiments: how fast does the receiver learn about an event,
how reliably, and what does it cost?

``python run_realtime_study.py`` (or ``python main.py realtime``) writes to
``results/realtime/``:

* ``alert_speed.csv``        **headline**: time from event onset until the
  receiver knows, for PulseLink and for typical streaming wearables, as the
  link gets slower, lossier or disconnects (seizure and, with the driver
  profile, drowsiness);
* ``ablation.csv``           what each PulseLink feature adds, on a hard link;
* ``usecase_detection.csv``  per use case and policy: sensitivity, false
  alarms per hour, latency, early warnings, on the wearable and at the receiver;
* ``task_compression.csv``   does the receiver still detect events after
  compression? base vs task-aware autoencoder;
* ``alert_delivery.csv``     probability that an alert is lost vs copies,
  spacing and burstiness (analytical Monte Carlo);
* ``policy_cost.csv``        bitrate, distortion, energy and battery per policy;
* ``chbmit_*.csv``           real seizures (CHB-MIT): rule-based, logistic
  regression, gradient boosting and a tiny CNN, leave-one-file-out;
* ``summary.md`` and ``figures/``.
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from communication.packet_loss import make_loss_model  # noqa: E402
from config import ChannelConfig  # noqa: E402
from realtime.detectors import CNNDetector, LearnedDetector, default_detectors  # noqa: E402
from realtime.engine import EngineConfig, StreamEngine, preset  # noqa: E402
from realtime.features import compute_features  # noqa: E402
from realtime.profiles import profile_detectors  # noqa: E402
from realtime.sources import LiveSynthSource, random_schedule  # noqa: E402

log = logging.getLogger(__name__)
DETECTOR_TRUTH = {"seizure": "seizure", "drowsiness": "drowsiness", "attention": "attention", "artifact": "artifact"}
TOLERATED = {"drowsiness": {"postictal"}}  # postictal slowing is not a false drowsiness alarm
TOL_S = 10.0  # detections up to 10 s after an event ends still count
PRE_S = 5.0  # ... and up to 5 s before the annotated onset (annotations are approximate)
MISS_S = 60.0  # receiver did not know within a minute of onset = missed
SYSTEMS = {
    "stream_raw": "Typical wearable: raw EEG, detect on phone",
    "stream": "Typical wearable: compressed EEG, detect on phone",
    "device_fifo": "Detect on wearable, alert queued with data",
    "pulselink": "PulseLink",
}
SYS_COLORS = {"stream_raw": "#6b7280", "stream": "#2563eb", "device_fifo": "#f59e0b", "pulselink": "#16a34a", "+priority": "#a3a3a3", "+early": "#86efac", "+ack": "#4ade80"}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def truth_intervals(records: list[dict], frame_s: float) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[list[float]]] = {}
    for r in records:
        t0 = r["t"] - frame_s
        for k in r["truth"]:
            iv = out.setdefault(k, [])
            if iv and abs(iv[-1][1] - t0) < 1e-6:
                iv[-1][1] = r["t"]
            else:
                iv.append([t0, r["t"]])
    return {k: [tuple(x) for x in v] for k, v in out.items()}


def _match(onsets, s, e):
    hits = [t for t in onsets if s - PRE_S <= t <= e + TOL_S]
    return hits[0] if hits else None


def score_events(engine: StreamEngine, detectors=None) -> list[dict]:
    """Match detector onsets to ground truth (call ``engine.finish()`` first).

    One row per (detector, side): true events, detections, false alarms,
    detection latency and, on the wearable, when the receiver got the alert.
    """
    truth = truth_intervals(engine.records, engine.cfg.frame_s)
    hours = engine.t / 3600
    rows = []
    names = {d.name for d in engine.device.detectors}
    for det, kind in (detectors or DETECTOR_TRUTH).items():
        if det not in names:
            continue
        ivs = truth.get(kind, [])
        ok_ivs = ivs + [iv for k in TOLERATED.get(det, ()) for iv in truth.get(k, [])]
        evs = [e for e in engine.events if e.detector == det]
        dev = sorted(e.t_onset_device for e in evs if np.isfinite(e.t_onset_device))
        rx = sorted(e.t_onset_receiver for e in evs if e.t_onset_receiver is not None)
        by_onset = {e.t_onset_device: e for e in evs if np.isfinite(e.t_onset_device)}
        early = [el for el in engine.early_logs if el.detector == det]
        for side, onsets in (("wearable", dev), ("receiver", rx)):
            if side == "wearable" and not engine.cfg.device_alerts:
                continue
            lat, alert_lat, first_lat, early_lat, alert_lost = [], [], [], [], 0
            for s, e in ivs:
                hit = _match(onsets, s, e)
                if hit is None:
                    continue
                lat.append(hit - s)
                if side == "wearable":
                    ev = by_onset[hit]
                    if ev.priority is not None and ev.priority <= 1:
                        if ev.t_alert_arrival is None:
                            alert_lost += 1
                        else:
                            alert_lat.append(ev.t_alert_arrival - s)
                        if ev.t_first_alert is not None:
                            first_lat.append(ev.t_first_alert - s)
                    if ev.t_early_device is not None:
                        early_lat.append(ev.t_early_device - s)
            fa = sum(not any(s - PRE_S <= t <= e + TOL_S for s, e in ok_ivs) for t in onsets)
            row = {"detector": det, "side": side, "events": len(ivs), "detected": len(lat), "false_alarms": fa, "hours": hours, "latency_s": lat, "alert_latency_s": alert_lat, "first_alert_s": first_lat, "early_latency_s": early_lat, "alerts_lost": alert_lost}
            if side == "wearable":
                row["early_warnings"] = len(early)
                row["early_false"] = sum(not any(s - PRE_S <= el.t_device <= e + TOL_S for s, e in ok_ivs) for el in early)
            rows.append(row)
    return rows


def _flat(g, col):
    return [x for l in g[col] for x in l] if col in g else []


def aggregate(rows: list[dict], keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        lat, al, fl, el = (_flat(g, c) for c in ("latency_s", "alert_latency_s", "first_alert_s", "early_latency_s"))
        ev = g["events"].sum()
        h = g["hours"].sum()
        row = {
            **dict(zip(keys, k)),
            "events": int(ev),
            "sensitivity": g["detected"].sum() / ev if ev else np.nan,
            "false_alarms_per_h": g["false_alarms"].sum() / h,
            "latency_median_s": float(np.median(lat)) if lat else np.nan,
            "latency_p90_s": float(np.percentile(lat, 90)) if lat else np.nan,
            "early_warning_median_s": float(np.median(el)) if el else np.nan,
            "first_alert_median_s": float(np.median(fl)) if fl else np.nan,
            "alert_latency_median_s": float(np.median(al)) if al else np.nan,
            "alerts_lost": int(g["alerts_lost"].sum()),
        }
        if "early_false" in g and g["early_false"].notna().any():
            row["false_early_warnings_per_h"] = g["early_false"].sum() / h
        out.append(row)
    return pd.DataFrame(out)


def in_event_prd(engine: StreamEngine, kind: str = "seizure") -> float:
    v = [r.get("backfilled_prd", r["prd"]) for r in engine.records if kind in r["truth"] and r["status"] == "ok"]
    return float(np.mean(v)) if v else np.nan


def time_to_alert(engine: StreamEngine, det: str = "seizure", kind: str = "seizure") -> list[dict]:
    """Per true event: when did the *receiver* first know, and when was it confirmed?"""
    truth = truth_intervals(engine.records, engine.cfg.frame_s).get(kind, [])
    evs = [e for e in engine.events if e.detector == det]
    out = []
    for s, e in truth:
        if engine.cfg.device_alerts:
            cand = [ev for ev in evs if np.isfinite(ev.t_onset_device) and s - PRE_S <= ev.t_onset_device <= e + TOL_S]
            ev = cand[0] if cand else None
            first = ev.t_first_alert if ev else None
            conf = ev.t_alert_arrival if ev else None
            path = ev.alert_path if ev else None
            local = ev.t_local_alarm if ev else None
        else:
            rx = sorted(ev.t_onset_receiver for ev in evs if ev.t_onset_receiver is not None and s - PRE_S <= ev.t_onset_receiver <= e + MISS_S)
            first = conf = rx[0] if rx else None
            path, local = "receiver", None
        f = (first - s) if first is not None and first - s <= MISS_S else np.nan
        c = (conf - s) if conf is not None and conf - s <= MISS_S else np.nan
        out.append({"onset_s": s, "first_s": f, "confirmed_s": c, "path": path, "local_alarm_s": (local - s) if local else np.nan})
    return out


def _summ(times: list[float]) -> dict:
    a = np.asarray(times, dtype=float)
    ok = a[np.isfinite(a)]
    n = len(a)
    return {
        "n": n,
        "median_s": float(np.median(ok)) if ok.size else np.nan,
        "p90_s": float(np.percentile(np.where(np.isfinite(a), a, 1e9), 90)) if n else np.nan,  # misses count as infinitely late
        "within_2s": float(np.mean(np.isfinite(a) & (a <= 2))) if n else np.nan,
        "within_5s": float(np.mean(np.isfinite(a) & (a <= 5))) if n else np.nan,
        "missed": float(np.mean(~np.isfinite(a))) if n else np.nan,
    }


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def _schedule(seed: int, minutes: float, kinds=("focus", "drowsy", "seizure", "artifact")):
    return random_schedule(np.random.default_rng(1000 + seed), minutes * 60, kinds=kinds)


def synth_session(seed: int, minutes: float, model, policy: str | None, channel: ChannelConfig, cfg: EngineConfig | None = None, profile: str = "general", kinds=("focus", "drowsy", "seizure", "artifact"), outage_before: float | None = None, outage_len: float = 30.0):
    sched = _schedule(seed, minutes, kinds)
    ch = replace(channel, seed=seed)
    if outage_before is not None:
        target = {"seizure": "seizure", "drowsiness": "drowsy"}.get(profile_target(profile), "seizure")
        ch = replace(ch, outages=[(s - outage_before, outage_len) for s, d, k in sched if k == target])
    cfg = cfg or EngineConfig()
    if policy is not None:
        cfg = replace(cfg, policy=policy)
    eng = StreamEngine(LiveSynthSource(seed=seed, schedule=sched), model, ch, cfg=cfg, detectors=profile_detectors(profile))
    eng.run(int(minutes * 60 / eng.cfg.frame_s))
    eng.finish()
    return eng


def profile_target(profile: str) -> str:
    return {"driver": "drowsiness"}.get(profile, "seizure")


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def alert_speed_experiment(model, runs: int, minutes: float, systems=tuple(SYSTEMS), profile: str = "general", bandwidths=(800, 1600, 3200, 16_000), losses=(0.0, 0.1, 0.2, 0.3, 0.4)):
    """Headline: onset -> receiver knows, as the link degrades."""
    target = profile_target(profile)
    truth = {"seizure": "seizure", "drowsiness": "drowsiness"}[target]
    kinds = ("seizure", "focus", "artifact") if target == "seizure" else ("drowsy", "focus", "artifact")
    scenarios = [("bandwidth", dict(bandwidth_bps=bw, loss_rate=0.05), bw) for bw in bandwidths]
    scenarios += [("bursty loss", dict(bandwidth_bps=16_000, loss_rate=lo, burst_length=4.0), lo) for lo in losses]
    scenarios += [("30 s outage at onset", dict(bandwidth_bps=16_000, loss_rate=0.02), 30.0)]
    rows = []
    for scen, ch, x in scenarios:
        for sysname in systems:
            for seed in range(runs):
                eng = synth_session(seed, minutes, model, None, ChannelConfig(**ch), cfg=preset(sysname), profile=profile, kinds=kinds, outage_before=2.0 if scen.startswith("30 s") else None)
                for r in time_to_alert(eng, target, truth):
                    rows.append({"profile": profile, "event": target, "scenario": scen, "x": x, "system": sysname, "seed": seed, **r, "tx_power_mw": eng.summary()["tx_power_mw"]})
        log.warning("alert speed [%s] %s %s done", profile, scen, x)
    return pd.DataFrame(rows)


def summarize_speed(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for k, g in df.groupby(["profile", "event", "scenario", "x", "system"], sort=False):
        first, conf = _summ(g["first_s"]), _summ(g["confirmed_s"])
        out.append({**dict(zip(["profile", "event", "scenario", "x", "system"], k)), "events": first["n"], "first_median_s": first["median_s"], "first_p90_s": first["p90_s"], "within_2s": first["within_2s"], "within_5s": first["within_5s"], "missed": first["missed"], "confirmed_median_s": conf["median_s"], "tx_power_mw": g["tx_power_mw"].mean()})
    return pd.DataFrame(out)


def ablation_experiment(model, runs: int, minutes: float):
    """Build PulseLink up one feature at a time on two hard links."""
    order = ["stream", "device_fifo", "+priority", "+early", "+ack", "pulselink"]
    scen = [("1.6 kbit/s, 20 % bursty loss", dict(bandwidth_bps=1600, loss_rate=0.2, burst_length=4.0), None), ("30 s outage at onset", dict(bandwidth_bps=16_000, loss_rate=0.02), 2.0)]
    rows = []
    for name, ch, ob in scen:
        for sysname in order:
            times, conf, power, local = [], [], [], []
            for seed in range(runs):
                eng = synth_session(seed, minutes, model, None, ChannelConfig(**ch), cfg=preset(sysname), kinds=("seizure", "focus", "artifact"), outage_before=ob)
                tt = time_to_alert(eng)
                times += [r["first_s"] for r in tt]
                conf += [r["confirmed_s"] for r in tt]
                local += [r["local_alarm_s"] for r in tt]
                power.append(eng.summary()["tx_power_mw"])
            s = _summ(times)
            rows.append({"scenario": name, "system": sysname, "events": s["n"], "first_median_s": s["median_s"], "first_p90_s": s["p90_s"], "within_5s": s["within_5s"], "missed": s["missed"], "confirmed_median_s": _summ(conf)["median_s"], "local_alarm_share": float(np.mean(np.isfinite(local))) if local else np.nan, "tx_power_mw": float(np.mean(power))})
        log.warning("ablation %s done", name)
    return pd.DataFrame(rows)


def usecase_experiment(models: dict, runs: int, minutes: float, policies, losses, bandwidth=16_000.0, burst=1.0):
    """``models``: label -> autoencoder (or None). Policy ``pulselink`` runs once per model."""
    det_rows, cost_rows = [], []
    for loss in losses:
        for policy in policies:
            variants = [(lab, m) for lab, m in models.items()] if policy == "pulselink" else [(policy, next(iter(models.values())))]
            for lab, model in variants:
                label = lab if policy == "pulselink" else policy
                for seed in range(runs):
                    eng = synth_session(seed, minutes, model, policy, ChannelConfig(bandwidth_bps=bandwidth, loss_rate=loss, burst_length=burst))
                    for r in score_events(eng):
                        det_rows.append({"policy": label, "loss_rate": loss, **r})
                    s = eng.summary()
                    cost_rows.append({"policy": label, "loss_rate": loss, "seed": seed, "prd_seizure": in_event_prd(eng, "seizure"), **s})
                log.warning("use cases: loss=%.2f policy=%s done", loss, label)
    det = aggregate(det_rows, ["detector", "side", "policy", "loss_rate"])
    cost = pd.DataFrame(cost_rows).groupby(["policy", "loss_rate"], sort=False).mean(numeric_only=True).reset_index()
    return det, cost


def task_compression_experiment(models: dict, n_frames_minutes: float = 12.0) -> pd.DataFrame:
    """Feature error after compression, per autoencoder (held-out synthetic frames)."""
    from compression.train_streaming import feature_error, synthetic_frames

    frames = synthetic_frames(n_frames_minutes, seed=999)
    rows = []
    for lab, m in models.items():
        if m is not None:
            rows.append({"model": lab, **feature_error(m, frames)})
    return pd.DataFrame(rows)


def alert_delivery_experiment(losses=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4), copies=(1, 2, 3, 5), bursts=(1.0, 4.0), spacings=(1, 8), trials: int = 20_000, seed: int = 0):
    """P(all copies of an alert are lost), without ACK.

    ``spacing`` = 1 sends copies back to back; ``spacing`` = 8 leaves 7
    packet slots between copies (time diversity). Under bursty loss
    back-to-back copies tend to die together, spaced copies do not.
    """
    rows = []
    for burst in bursts:
        for spacing in spacings:
            if burst <= 1 and spacing > 1:
                continue  # spacing changes nothing for independent loss
            for loss in losses:
                for n in copies:
                    rng = np.random.default_rng(seed)
                    lost = 0
                    for _ in range(trials):
                        m = make_loss_model(loss, burst, rng)
                        for _ in range(int(rng.integers(0, 20))):  # random point in the loss process
                            m.lost()
                        all_lost = True
                        for k in range(n):
                            for _ in range(spacing - 1 if k else 0):
                                m.lost()
                            all_lost &= m.lost()
                        lost += all_lost
                    rows.append({"burst_length": burst, "spacing_packets": spacing, "loss_rate": loss, "copies": n, "p_alert_lost": lost / trials, "independent_theory": loss**n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CHB-MIT (real seizures)
# --------------------------------------------------------------------------- #
def window_dataset(data: np.ndarray, fs: float, seizures, win_s=4.0, hop_s=2.0):
    from realtime.engine import StreamingBandpass

    x = StreamingBandpass(fs, data.shape[0])(data)
    w, h = int(win_s * fs), int(hop_s * fs)
    feats, y, raw = [], [], []
    for s in range(0, x.shape[1] - w, h):
        t_end = (s + w) / fs
        feats.append(compute_features(x[:, s : s + w], fs))
        raw.append(x[:, s : s + w].astype(np.float32))
        y.append(any(a <= t_end - win_s / 2 <= b for a, b in seizures))
    return feats, np.array(y), np.stack(raw)


def train_models(train_f, train_y, train_raw, seed: int = 0, cnn: bool = True) -> list:
    """Patient-specific detectors: logistic regression, boosted trees, tiny CNN."""
    kw = dict(critical=True, threshold=0.8, on_frames=2, off_frames=3)
    out = [
        LearnedDetector(name="seizure_logreg", title="Seizure (logistic regression)", kind="logreg", **kw).fit(train_f, train_y),
        LearnedDetector(name="seizure_gbm", title="Seizure (boosted trees)", kind="gbm", **kw).fit(train_f, train_y),
    ]
    if cnn:
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(train_y)
        neg = np.flatnonzero(~train_y)
        neg = rng.choice(neg, size=min(len(neg), 4000), replace=False)
        idx = np.concatenate([pos, neg])
        out.append(CNNDetector(name="seizure_cnn", title="Seizure (tiny CNN)", **kw).fit(train_raw[idx], train_y[idx], epochs=20, seed=seed))
    return out


def chbmit_experiment(patient: str, n_files: int, download: bool, model, out: Path, cnn: bool = True):
    from realtime.sources import DEFAULT_CHB_CHANNELS, RecordingSource, fetch_chbmit, load_edf

    files = fetch_chbmit(patient, n_seizure_files=n_files, n_normal_files=2, download=download)
    if not files:
        raise FileNotFoundError(f"no CHB-MIT files for {patient}")
    recs = []
    for p, sz in files:
        data, picks = load_edf(p, DEFAULT_CHB_CHANNELS, 100.0)
        recs.append((p.name, data, sz))
        log.warning("loaded %s: %.0f min, %d seizure(s)", p.name, data.shape[1] / 6000, len(sz))
    win = {name: window_dataset(d, 100.0, sz) for name, d, sz in recs}
    rows, per_seizure, speed = [], [], []
    ch = ChannelConfig(bandwidth_bps=16_000, loss_rate=0.05)
    for fold, (name, data, sz) in enumerate(recs):
        tr = [n for n in win if n != name]
        train_y = np.concatenate([win[n][1] for n in tr])
        learned = []
        if train_y.sum() >= 3:
            learned = train_models([f for n in tr for f in win[n][0]], train_y, np.concatenate([win[n][2] for n in tr]), seed=fold, cnn=cnn)
        for sysname in ("pulselink", "stream"):
            dets = default_detectors(("seizure", "artifact")) + [d.clone() for d in learned]
            src = RecordingSource(data, 100.0, {"seizure": sz}, name=name, loop=False)
            eng = StreamEngine(src, model, ch, cfg=preset(sysname), detectors=dets)
            eng.run(int(data.shape[1] / 100 / eng.cfg.frame_s) - 1)
            eng.finish()
            truth = truth_intervals(eng.records, eng.cfg.frame_s).get("seizure", [])
            names = {d.name: "seizure" for d in dets if d.name != "artifact"}
            for r in score_events(eng, names):
                r["events"] = len(truth)
                rows.append({"file": name, "system": sysname, **r})
            for dn in names:
                for t in time_to_alert(eng, dn, "seizure"):
                    speed.append({"file": name, "system": sysname, "detector": dn, **t})
            for dn in names:
                evs = [e for e in eng.events if e.detector == dn]
                for s, e in truth:
                    if sysname == "pulselink":
                        hit = _match(sorted(x.t_onset_device for x in evs if np.isfinite(x.t_onset_device)), s, e)
                    else:
                        hit = _match(sorted(x.t_onset_receiver for x in evs if x.t_onset_receiver is not None), s, e)
                    per_seizure.append({"file": name, "system": sysname, "detector": dn, "seizure_start_s": s, "duration_s": e - s, "detected": hit is not None, "latency_s": hit - s if hit is not None else np.nan})
            s = eng.summary()
            log.warning("CHB-MIT %s %s: CR %.1f, saving %.0f %%", name, sysname, s["compression_ratio"], s["energy_saving_pct"])
    # final detectors on all files, for the live demo
    all_y = np.concatenate([Y for _, Y, _ in win.values()])
    if all_y.sum() >= 3:
        final = train_models([f for F, _, _ in win.values() for f in F], all_y, np.concatenate([R for _, _, R in win.values()]), cnn=cnn)
        for d in final:
            with open(out / f"learned_{d.name}_{patient}.pkl", "wb") as f:
                pickle.dump(d, f)
    agg = aggregate(rows, ["detector", "side", "system"])
    sp = pd.DataFrame(speed)
    sp_sum = []
    for k, g in sp.groupby(["detector", "system"], sort=False):
        s = _summ(g["first_s"])
        sp_sum.append({"detector": k[0], "system": k[1], "seizures": s["n"], "receiver_knows_median_s": s["median_s"], "receiver_knows_p90_s": s["p90_s"], "within_5s": s["within_5s"], "missed": s["missed"]})
    return agg, pd.DataFrame(per_seizure), pd.DataFrame(sp_sum), [(n, sz) for n, _, sz in recs]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _never(ax, ymax):
    ax.axhline(ymax, color="k", lw=0.5, ls=":")
    ax.text(ax.get_xlim()[0], ymax, " never / > 60 s", fontsize=7, va="bottom")


def plot_headline(summary: pd.DataFrame, path: Path, profile: str = "general") -> None:
    d = summary[summary.profile == profile]
    if d.empty:
        return
    ev = d["event"].iloc[0]
    ymax = 70.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, scen, xl, logx in ((axes[0], "bursty loss", "packet loss rate (bursts of ~100 ms), 16 kbit/s", False), (axes[1], "bandwidth", "link capacity (bit/s), 5 % loss", True)):
        g0 = d[d.scenario == scen]
        for sysname, lab in SYSTEMS.items():
            g = g0[g0.system == sysname].sort_values("x")
            if g.empty:
                continue
            y = g["first_median_s"].fillna(ymax).clip(upper=ymax)
            ax.plot(g["x"], y, "o-", color=SYS_COLORS[sysname], label=lab + (" (first warning)" if sysname == "pulselink" else ""), lw=2.5 if sysname == "pulselink" else 1.5)
            if sysname == "pulselink":
                ax.plot(g["x"], g["confirmed_median_s"].fillna(ymax).clip(upper=ymax), "s--", color=SYS_COLORS[sysname], alpha=0.6, label="PulseLink (confirmed alert)")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.8, ymax * 1.3)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3, which="both")
        _never(ax, ymax)
    axes[0].set_ylabel(f"{ev} onset -> receiver knows (s, median)")
    axes[0].legend(fontsize=7, loc="upper left")
    g = d[d.scenario.str.startswith("30 s")]
    order = [s for s in SYSTEMS if s in set(g.system)]
    vals = [g[g.system == s]["first_median_s"].fillna(ymax).clip(upper=ymax).mean() for s in order]
    axes[2].bar(range(len(order)), vals, color=[SYS_COLORS[s] for s in order])
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels([s.replace("_", "\n") for s in order], fontsize=8)
    axes[2].set_ylabel("s (median)")
    axes[2].set_title("link down 2 s before onset, for 30 s", fontsize=10)
    for i, v in enumerate(vals):
        axes[2].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Fastest reliable alert: how long until the receiver knows about a {ev}?", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation(df: pd.DataFrame, path: Path) -> None:
    scen = list(dict.fromkeys(df.scenario))
    fig, axes = plt.subplots(1, len(scen), figsize=(6 * len(scen), 3.8), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), scen):
        g = df[df.scenario == s]
        y = g["first_median_s"].fillna(60).clip(upper=60)
        ax.barh(range(len(g)), y, color=[SYS_COLORS.get(x, "#999") for x in g.system])
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.system)
        for i, (v, w) in enumerate(zip(y, g["within_5s"])):
            ax.text(v, i, f" {v:.1f} s  ({w:.0%} within 5 s)", va="center", fontsize=8)
        ax.set_xlabel("onset -> receiver knows (s, median)")
        ax.set_title(s, fontsize=10)
        ax.set_xlim(0, 75)
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("What each part of PulseLink adds (seizure)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency(det: pd.DataFrame, path: Path, loss: float) -> None:
    d = det[det["loss_rate"] == loss]
    dets = [x for x in DETECTOR_TRUTH if x in set(d["detector"])]
    rxp = [p for p in ("event", "pulselink_stream", "pulselink", "raw") if p in set(d.policy)]
    fig, ax = plt.subplots(figsize=(9, 4))
    series = [("wearable", "event", "detected on the wearable", "#16a34a", "latency_median_s")]
    series += [("wearable", "event", "early warning (critical only)", "#86efac", "early_warning_median_s")]
    series += [("receiver", p, f"receiver, {p} stream", c, "latency_median_s") for p, c in zip(rxp, ("#2563eb", "#9333ea", "#c084fc", "#6b7280"))]
    w = 0.8 / len(series)
    for i, (side, pol, lab, col, colname) in enumerate(series):
        vals = [d[(d.detector == k) & (d.side == side) & (d.policy == pol)][colname].mean() if colname in d else np.nan for k in dets]
        ax.bar(np.arange(len(dets)) + i * w, vals, w, label=lab, color=col)
    ax.set_xticks(np.arange(len(dets)) + w * (len(series) - 1) / 2)
    ax.set_xticklabels(dets)
    ax.set_ylabel("median time from event onset (s)")
    ax.set_title(f"How fast is each use case detected? (packet loss {loss:.0%})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_alerts(df: pd.DataFrame, path: Path) -> None:
    combos = df[["burst_length", "spacing_packets"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(combos), figsize=(4.5 * len(combos), 4), sharey=True)
    for ax, (burst, sp) in zip(np.atleast_1d(axes), combos):
        d = df[(df.burst_length == burst) & (df.spacing_packets == sp)]
        for n, g in d.groupby("copies"):
            ax.semilogy(g["loss_rate"], g["p_alert_lost"].clip(lower=1e-5), "o-", label=f"{n} cop{'y' if n == 1 else 'ies'}")
        ax.set_title("independent loss" if burst <= 1 else f"bursty (burst {burst:g}), copies {'back to back' if sp == 1 else f'{sp} slots apart'}", fontsize=10)
        ax.set_xlabel("packet loss rate")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("P(alert never arrives)")
    axes[0].legend()
    fig.suptitle("Redundant alert packets (no ACK): back-to-back copies die together in bursts; spacing restores the benefit")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy(cost: pd.DataFrame, path: Path, loss: float) -> None:
    d = cost[cost.loss_rate == loss]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    cols = ["#6b7280", "#2563eb", "#9333ea", "#c084fc", "#16a34a"][: len(d)]
    axes[0].bar(d["policy"], d["bitrate_bps"], color=cols)
    axes[0].set_ylabel("bit/s on air")
    axes[1].bar(d["policy"], d["prd_seizure"], color=cols)
    axes[1].set_ylabel("PRD during seizures (%)")
    axes[2].bar(d["policy"], d["energy_saving_pct"], color=cols)
    axes[2].set_ylabel("radio+compute saving vs raw (%)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8, rotation=15)
    fig.suptitle(f"Event-aware streaming: low rate most of the time, high fidelity when it matters (loss {loss:.0%})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_task(task: pd.DataFrame, det: pd.DataFrame, path: Path, loss: float = 0.05) -> None:
    if task.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    names = ["delta", "theta", "alpha", "beta", "line_length", "theta_alpha", "engagement"]
    w = 0.8 / len(task)
    for i, (_, r) in enumerate(task.iterrows()):
        axes[0].bar(np.arange(len(names)) + i * w, [r[f"err_{n}"] for n in names], w, label=f"{r['model']} (CR {r['cr']:.0f}x, PRD {r['prd']:.0f} %)")
    axes[0].set_xticks(np.arange(len(names)) + w * (len(task) - 1) / 2)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel("feature error after compression (log units)")
    axes[0].legend(fontsize=8)
    d = det[(det.side == "receiver") & (det.loss_rate == loss)]
    pols = [p for p in ("raw", "dwt", "pulselink_base", "pulselink_stream") if p in set(d.policy)]
    dets = [x for x in DETECTOR_TRUTH if x in set(d.detector)]
    w = 0.8 / max(1, len(pols))
    for i, p in enumerate(pols):
        axes[1].bar(np.arange(len(dets)) + i * w, [d[(d.detector == k) & (d.policy == p)]["sensitivity"].mean() for k in dets], w, label=p)
    axes[1].set_xticks(np.arange(len(dets)) + w * (len(pols) - 1) / 2)
    axes[1].set_xticklabels(dets)
    axes[1].set_ylabel("sensitivity at the receiver")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Task-aware compression: train the codec to keep what the detectors need")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_session(eng: StreamEngine, path: Path) -> None:
    recs = [r for r in eng.records if eng.recon.get(r["frame"]) is not None]
    fs, fsec = eng.fs, eng.cfg.frame_s
    t = np.concatenate([r["t"] - fsec + np.arange(eng.n) / fs for r in recs])
    x = np.concatenate([eng.orig[r["frame"]][0] for r in recs])
    y = np.concatenate([eng.recon[r["frame"]][0] for r in recs])
    dets = [d for d in eng.device.detectors]
    fig, axes = plt.subplots(2 + len(dets), 1, figsize=(13, 2 + 0.8 * len(dets) + 3), sharex=True, gridspec_kw={"height_ratios": [3, 0.6] + [0.8] * len(dets)})
    axes[0].plot(t, x, color="k", lw=0.5, label="original (wearable)")
    axes[0].plot(t, y, color="#16a34a", lw=0.5, alpha=0.8, label="reconstructed (receiver)")
    colors = {"seizure": "#ef4444", "drowsiness": "#f59e0b", "attention": "#3b82f6", "artifact": "#a855f7", "postictal": "#94a3b8"}
    for k, ivs in truth_intervals(recs, fsec).items():
        for s, e in ivs:
            for ax in axes:
                ax.axvspan(s, e, color=colors.get(k, "gray"), alpha=0.12)
            axes[0].text(s, axes[0].get_ylim()[1] * 0.85 if axes[0].get_ylim()[1] > 0 else 0, k, fontsize=8, color=colors.get(k, "gray"))
    for ev in eng.events:
        if ev.t_early_arrival:
            axes[0].axvline(ev.t_early_arrival, color="#f59e0b", lw=1, ls="--")
        if ev.t_alert_arrival and ev.priority == 0:
            axes[0].axvline(ev.t_alert_arrival, color="#dc2626", lw=1.2)
    axes[0].set_ylabel("uV")
    axes[0].legend(loc="lower right", fontsize=8)
    codec_col = {"raw": "#6b7280", "dwt": "#2563eb", "pulselink": "#16a34a", "hifi": "#f59e0b", "mid": "#fb923c"}
    axes[1].bar([r["t"] - fsec / 2 for r in recs], [r["bytes"] for r in recs], width=fsec, color=[codec_col.get(r["codec"], "#999") if r["status"] == "ok" else "#dc2626" for r in recs])
    axes[1].set_ylabel("bytes")
    for ax, d in zip(axes[2:], dets):
        sc = [r["device"][d.name]["score"] for r in recs]
        sc = [np.nan if v is None else v for v in sc]
        ax.plot([r["t"] for r in recs], sc, color=d.color, lw=1)
        if d.name in recs[-1].get("early", {}):
            es = [(r.get("early", {}).get(d.name) or {}).get("score") for r in recs]
            ax.plot([r["t"] for r in recs], [np.nan if v is None else v for v in es], color=d.color, lw=0.7, ls=":", alpha=0.8)
        ax.axhline(d.threshold, color="k", ls=":", lw=0.8)
        act = [r["t"] for r in recs if r["device"][d.name]["active"]]
        ax.scatter(act, [d.threshold] * len(act), s=6, color=d.color)
        ax.set_ylabel(d.title, rotation=0, ha="right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("One synthetic session: events, early warning (dashed amber) and confirmed alert (red) at the receiver, codec per frame")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _md(df: pd.DataFrame, floatfmt=".3g") -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:{floatfmt}}")
    return "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.values)


def run_all(out: Path, model=None, runs: int = 12, minutes: float = 10.0, chbmit: str | None = None, chb_files: int = 4, download: bool = False, quick: bool = False, base_model=None, cnn: bool = True):
    """Every experiment. ``model`` = autoencoder used in the system (task-aware
    streaming model if available); ``base_model`` = the original 30 s model, compared in the task experiment."""
    t0 = time.time()
    if quick:
        runs, minutes = min(runs, 3), min(minutes, 6.0)
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    models = {k: v for k, v in (("pulselink_stream", model), ("pulselink_base", base_model)) if v is not None} or {"none": None}
    sys_model = model if model is not None else base_model
    policies = ["raw", "dwt", "pulselink", "event"] if sys_model is not None else ["raw", "dwt", "event"]
    losses = (0.0, 0.05, 0.2) if quick else (0.0, 0.05, 0.1, 0.2)
    sp_runs = max(3, runs // 2)
    sp_min = min(minutes, 6.0)

    print(f"[1/7] headline: time to alert vs link quality ({sp_runs} sessions x {sp_min:g} min per point)")
    speed = pd.concat([alert_speed_experiment(sys_model, sp_runs, sp_min, profile="general"), alert_speed_experiment(sys_model, max(2, sp_runs // 2), sp_min, profile="driver", bandwidths=(1600, 16_000), losses=(0.0, 0.2, 0.4))])
    speed.to_csv(out / "alert_speed_events.csv", index=False)
    speed_sum = summarize_speed(speed)
    speed_sum.to_csv(out / "alert_speed.csv", index=False)
    print("[2/7] ablation")
    abl = ablation_experiment(sys_model, sp_runs, sp_min)
    abl.to_csv(out / "ablation.csv", index=False)
    print(f"[3/7] use cases: {runs} sessions x {minutes:g} min x {len(policies) + len(models) - 1} policies x {len(losses)} loss rates")
    det, cost = usecase_experiment(models, runs, minutes, policies, losses)
    det.to_csv(out / "usecase_detection.csv", index=False)
    cost.to_csv(out / "policy_cost.csv", index=False)
    print("[4/7] task-aware compression")
    task = task_compression_experiment(models)
    task.to_csv(out / "task_compression.csv", index=False)
    print("[5/7] alert delivery")
    alerts = alert_delivery_experiment(trials=4000 if quick else 20_000)
    alerts.to_csv(out / "alert_delivery.csv", index=False)
    print("[6/7] figures")
    plot_headline(speed_sum, out / "figures" / "headline_time_to_alert.png", "general")
    plot_headline(speed_sum, out / "figures" / "headline_time_to_alert_driver.png", "driver")
    plot_ablation(abl, out / "figures" / "ablation.png")
    plot_latency(det, out / "figures" / "latency_by_usecase.png", 0.05)
    plot_alerts(alerts, out / "figures" / "alert_delivery.png")
    plot_policy(cost, out / "figures" / "policy_tradeoff.png", 0.05)
    plot_task(task, det, out / "figures" / "task_aware_compression.png", 0.05)
    plot_session(synth_session(3, 6.0, sys_model, "event", ChannelConfig(bandwidth_bps=16_000, loss_rate=0.05)), out / "figures" / "example_session.png")

    chb_txt = ""
    if chbmit:
        print(f"[7/7] CHB-MIT {chbmit}")
        try:
            agg, per, sp, files = chbmit_experiment(chbmit, chb_files, download, sys_model, out, cnn=cnn)
            agg.to_csv(out / f"chbmit_{chbmit}_detection.csv", index=False)
            per.to_csv(out / f"chbmit_{chbmit}_per_seizure.csv", index=False)
            sp.to_csv(out / f"chbmit_{chbmit}_time_to_alert.csv", index=False)
            cols = [c for c in ["detector", "side", "system", "events", "sensitivity", "false_alarms_per_h", "latency_median_s", "early_warning_median_s", "first_alert_median_s"] if c in agg]
            chb_txt = (
                f"\n## 7. CHB-MIT {chbmit}: real seizures (F7-T7 and F8-T8, 16 kbit/s, 5 % loss)\n\n"
                f"Files: {', '.join(f'{n} ({len(s)} seizure(s))' for n, s in files)}. Learned detectors are trained leave-one-file-out "
                "(each file is tested by a model that never saw it). `pulselink` = detection on the wearable; `stream` = typical wearable, detection at the receiver.\n\n"
                + _md(agg[cols]) + "\n\nWhen did the receiver know (first warning or alert, from annotated onset):\n\n" + _md(sp) + "\n"
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("CHB-MIT failed")
            chb_txt = f"\n## 7. CHB-MIT {chbmit}\n\nFailed: {exc}\n"
    else:
        print("[7/7] CHB-MIT skipped (add --chbmit chb01 --download)")

    loss = 0.05
    hs = speed_sum[speed_sum.profile == "general"]
    head = hs[(hs.scenario == "bursty loss") | (hs.scenario.str.startswith("30 s")) | ((hs.scenario == "bandwidth") & hs.x.isin([800, 16_000]))]
    head = head[["scenario", "x", "system", "events", "first_median_s", "first_p90_s", "within_2s", "within_5s", "missed", "confirmed_median_s"]]
    drv = speed_sum[speed_sum.profile == "driver"][["scenario", "x", "system", "events", "first_median_s", "within_5s", "missed", "confirmed_median_s"]]
    d = det[det.loss_rate == loss]
    wcols = [c for c in ["detector", "events", "sensitivity", "false_alarms_per_h", "early_warning_median_s", "false_early_warnings_per_h", "latency_median_s", "first_alert_median_s", "alert_latency_median_s", "alerts_lost"] if c in d]
    wear = d[(d.side == "wearable") & (d.policy == "event")][wcols]
    rx = d[d.side == "receiver"][["detector", "policy", "sensitivity", "false_alarms_per_h", "latency_median_s"]]
    c = cost[cost.loss_rate == loss][["policy", "compression_ratio", "bitrate_bps", "prd_delivered", "prd_seizure", "frames_lost", "tx_power_mw", "energy_saving_pct", "battery_h"]].copy()
    c["battery_days"] = c.pop("battery_h") / 24
    al = alerts[alerts.loss_rate.isin([0.1, 0.2, 0.4])].pivot_table(index=["burst_length", "spacing_packets", "loss_rate"], columns="copies", values="p_alert_lost").reset_index()
    al.columns = [str(x) if not isinstance(x, str) else x for x in al.columns]
    text = f"""# PulseLink results: fastest reliable alert

Synthetic sessions with random events (seizure, drowsiness, focus, movement artifact) after a 45 s
calibration period, 2 channels at 100 Hz, 0.5 s detection hop, 2 s frames. "Receiver knows" = the first
warning or alert reaches the phone, measured from the true onset. Synthetic data shows that the system
works end to end and lets us vary the link; it is **not** evidence of clinical performance (see CHB-MIT below).

Systems: `stream_raw` / `stream` = a typical wearable that streams (raw / compressed) EEG over one queue
and detects on the phone; `device_fifo` = detects on the wearable but the alert waits in the same queue;
`pulselink` = priority link (P0-P4), two-stage early warning, ACK + spaced copies, backup radio, heartbeat.

## 1. Headline: how long until the receiver knows about a seizure?

{_md(head)}

Same for drowsiness with the **driver** profile (drowsiness is the critical P0 event there):

{_md(drv)}

## 2. What each part adds (seizure)

{_md(abl)}

## 3. Detection on the wearable (packet loss {loss:.0%}, event-aware policy)

{_md(wear)}

## 4. Same detectors at the receiver, after compression

{_md(rx)}

Feature error after compression (natural-log units, lower is better), base vs task-aware autoencoder:

{_md(task) if not task.empty else "(no autoencoder)"}

## 5. Cost of each transmission policy (packet loss {loss:.0%})

{_md(c)}

## 6. Probability that an alert never arrives without ACK (columns = copies)

{_md(al)}
{chb_txt}
Energy numbers come from `config.EnergyConfig` and `communication.priority_link.BackupRadioConfig`
(assumptions, not measurements); battery life assumes a 100 mAh, 3.7 V cell and an always-on 0.5 mW front end.
Runtime {time.time() - t0:.0f} s.
"""
    (out / "summary.md").write_text(text)
    print(f"done in {time.time() - t0:.0f} s -> {out}")
    return text
