"""Real-time experiments: how fast and how reliably are events detected and
delivered, and what does it cost?

``python main.py realtime`` writes to ``results/realtime/``:

* ``usecase_detection.csv``  per use case, policy and loss rate: sensitivity,
  false alarms per hour and latency, on the wearable and at the receiver;
* ``alert_delivery.csv``      probability that an alert is lost, by loss rate,
  burstiness and number of redundant copies;
* ``bandwidth_latency.csv``   receiver-side seizure detection delay vs link
  capacity and a disconnection, compared with on-device alerts;
* ``policy_cost.csv``         bitrate, distortion (overall and during events),
  transmitter power, energy saving and battery life per policy;
* ``chbmit_*.csv``            (with ``--chbmit``) seizure detection on real
  CHB-MIT recordings: rule-based and a patient-specific learned detector;
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
from realtime.detectors import LearnedDetector, default_detectors  # noqa: E402
from realtime.engine import EngineConfig, StreamEngine  # noqa: E402
from realtime.features import compute_features  # noqa: E402
from realtime.sources import LiveSynthSource, random_schedule  # noqa: E402

log = logging.getLogger(__name__)
DETECTOR_TRUTH = {"seizure": "seizure", "drowsiness": "drowsiness", "attention": "attention", "artifact": "artifact"}
TOLERATED = {"drowsiness": {"postictal"}}  # postictal slowing is not a false drowsiness alarm
TOL_S = 10.0


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


def score_events(engine: StreamEngine, offset_s: float = 0.0) -> list[dict]:
    """Match detector onsets to ground-truth intervals.

    Returns one row per (detector, side) with true events, detections,
    false alarms and latencies. ``offset_s`` converts stream time to the
    recording's time base (for recordings that do not start at 0).
    """
    fs_ = engine.cfg.frame_s
    truth = truth_intervals(engine.records, fs_)
    hours = engine.t / 3600
    rows = []
    for det, kind in DETECTOR_TRUTH.items():
        if det not in {d.name for d in engine.device.detectors}:
            continue
        ivs = truth.get(kind, [])
        ok_ivs = ivs + [iv for k in TOLERATED.get(det, ()) for iv in truth.get(k, [])]
        dev = sorted(e.t_onset_device for e in engine.events if e.detector == det and np.isfinite(e.t_onset_device))
        rx = sorted(e.t_onset_receiver for e in engine.events if e.detector == det and e.t_onset_receiver is not None)
        alerts = {e.t_onset_device: e for e in engine.events if e.detector == det and np.isfinite(e.t_onset_device)}
        for side, onsets in (("wearable", dev), ("receiver", rx)):
            lat, alert_lat, alert_lost = [], [], 0
            for s, e in ivs:
                hits = [t for t in onsets if s <= t <= e + TOL_S]
                if hits:
                    lat.append(hits[0] - s)
                    if side == "wearable" and engine.device._det(det).critical:
                        ev = alerts[hits[0]]
                        if ev.t_alert_arrival is None:
                            alert_lost += 1
                        else:
                            alert_lat.append(ev.t_alert_arrival - s)
            fa = sum(not any(s <= t <= e + TOL_S for s, e in ok_ivs) for t in onsets)
            rows.append({
                "detector": det,
                "side": side,
                "events": len(ivs),
                "detected": len(lat),
                "false_alarms": fa,
                "hours": hours,
                "latency_s": lat,
                "alert_latency_s": alert_lat,
                "alerts_lost": alert_lost,
            })
    return rows


def aggregate(rows: list[dict], keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        lat = [x for l in g["latency_s"] for x in l]
        al = [x for l in g["alert_latency_s"] for x in l]
        ev = g["events"].sum()
        out.append({
            **dict(zip(keys, k)),
            "events": int(ev),
            "sensitivity": g["detected"].sum() / ev if ev else np.nan,
            "false_alarms_per_h": g["false_alarms"].sum() / g["hours"].sum(),
            "latency_median_s": float(np.median(lat)) if lat else np.nan,
            "latency_p90_s": float(np.percentile(lat, 90)) if lat else np.nan,
            "alert_latency_median_s": float(np.median(al)) if al else np.nan,
            "alerts_lost": int(g["alerts_lost"].sum()),
        })
    return pd.DataFrame(out)


def in_event_prd(engine: StreamEngine, kind: str = "seizure") -> float:
    v = [r["prd"] for r in engine.records if kind in r["truth"] and r["status"] == "ok"]
    return float(np.mean(v)) if v else np.nan


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def synth_session(seed: int, minutes: float, model, policy: str, channel: ChannelConfig, cfg: EngineConfig | None = None):
    rng = np.random.default_rng(1000 + seed)
    sched = random_schedule(rng, minutes * 60)
    src = LiveSynthSource(seed=seed, schedule=sched)
    eng = StreamEngine(src, model, replace(channel, seed=seed), cfg=replace(cfg or EngineConfig(), policy=policy), detectors=default_detectors())
    eng.run(int(minutes * 60 / eng.cfg.frame_s))
    return eng


def usecase_experiment(model, runs: int, minutes: float, policies, losses, bandwidth=16_000.0, burst=1.0):
    det_rows, cost_rows = [], []
    for loss in losses:
        for policy in policies:
            for seed in range(runs):
                eng = synth_session(seed, minutes, model, policy, ChannelConfig(bandwidth_bps=bandwidth, loss_rate=loss, burst_length=burst))
                for r in score_events(eng):
                    det_rows.append({"policy": policy, "loss_rate": loss, **r})
                s = eng.summary()
                cost_rows.append({"policy": policy, "loss_rate": loss, "seed": seed, "prd_seizure": in_event_prd(eng, "seizure"), **s})
            log.warning("use cases: loss=%.2f policy=%s done", loss, policy)
    det = aggregate(det_rows, ["detector", "side", "policy", "loss_rate"])
    cost = pd.DataFrame(cost_rows).groupby(["policy", "loss_rate"], sort=False).mean(numeric_only=True).reset_index()
    return det, cost


def alert_delivery_experiment(losses=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4), copies=(1, 2, 3, 5), bursts=(1.0, 4.0), spacings=(1, 8), trials: int = 20_000, seed: int = 0):
    """P(all copies of an alert are lost).

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


def bandwidth_experiment(model, runs: int, minutes: float, bandwidths=(800, 1600, 3200, 16_000), policies=("raw", "dwt", "event")):
    rows = []
    for bw in bandwidths:
        for policy in policies:
            for seed in range(runs):
                eng = synth_session(seed, minutes, model, policy, ChannelConfig(bandwidth_bps=bw, loss_rate=0.02))
                for r in score_events(eng):
                    if r["detector"] == "seizure":
                        rows.append({"bandwidth_bps": bw, "policy": policy, "scenario": "bandwidth", **r})
                rows[-1]["max_latency_s"] = eng.summary()["max_latency_s"]
    # disconnection: link drops 5 s before each seizure for 30 s
    for policy in policies:
        for seed in range(runs):
            rng = np.random.default_rng(1000 + seed)
            sched = random_schedule(rng, minutes * 60)
            outs = [(s - 5, 30.0) for s, d, k in sched if k == "seizure"]
            eng = synth_session(seed, minutes, model, policy, ChannelConfig(bandwidth_bps=16_000, loss_rate=0.02, outages=outs))
            for r in score_events(eng):
                if r["detector"] == "seizure":
                    rows.append({"bandwidth_bps": 16_000, "policy": policy, "scenario": "30 s outage at onset", **r})
    return aggregate(rows, ["scenario", "bandwidth_bps", "policy", "side"])


# --------------------------------------------------------------------------- #
# CHB-MIT (real seizures)
# --------------------------------------------------------------------------- #
def window_dataset(data: np.ndarray, fs: float, seizures, win_s=4.0, hop_s=2.0):
    from realtime.engine import StreamingBandpass

    x = StreamingBandpass(fs, data.shape[0])(data)
    w, h = int(win_s * fs), int(hop_s * fs)
    feats, y = [], []
    for s in range(0, x.shape[1] - w, h):
        t_end = (s + w) / fs
        feats.append(compute_features(x[:, s : s + w], fs))
        y.append(any(a <= t_end - win_s / 2 <= b for a, b in seizures))
    return feats, np.array(y)


def chbmit_experiment(patient: str, n_files: int, download: bool, model, out: Path, policies=("raw", "dwt", "pulselink", "event")):
    from realtime.sources import DEFAULT_CHB_CHANNELS, RecordingSource, fetch_chbmit, load_edf

    files = fetch_chbmit(patient, n_seizure_files=n_files, n_normal_files=2, download=download)
    if not files:
        raise FileNotFoundError(f"no CHB-MIT files for {patient}")
    recs = []
    for p, sz in files:
        data, picks = load_edf(p, DEFAULT_CHB_CHANNELS, 100.0)
        recs.append((p.name, data, sz))
        log.warning("loaded %s: %.0f min, %d seizure(s)", p.name, data.shape[1] / 6000, len(sz))

    # patient-specific learned detector, leave-one-file-out
    win = {name: window_dataset(d, 100.0, sz) for name, d, sz in recs}
    rows, per_seizure = [], []
    for name, data, sz in recs:
        train_f = [f for n, (F, _) in win.items() if n != name for f in F]
        train_y = np.concatenate([Y for n, (_, Y) in win.items() if n != name])
        learned = None
        if train_y.sum() >= 3:
            learned = LearnedDetector(name="seizure_learned", title="Seizure (learned)", critical=True, threshold=0.8, on_frames=2, off_frames=3).fit(train_f, train_y)
        for policy in policies:
            dets = default_detectors(("seizure", "artifact")) + ([learned.clone()] if learned else [])
            src = RecordingSource(data, 100.0, {"seizure": sz}, name=name, loop=False)
            eng = StreamEngine(src, model, ChannelConfig(bandwidth_bps=16_000, loss_rate=0.05), cfg=EngineConfig(policy=policy), detectors=dets)
            n = int(data.shape[1] / 100 / eng.cfg.frame_s) - 1
            eng.run(n)
            truth = truth_intervals(eng.records, eng.cfg.frame_s).get("seizure", [])
            for det_name in [d.name for d in dets if d.name != "artifact"]:
                for side in ("wearable", "receiver"):
                    onsets = sorted(
                        (e.t_onset_device if side == "wearable" else e.t_onset_receiver)
                        for e in eng.events
                        if e.detector == det_name and (np.isfinite(e.t_onset_device) if side == "wearable" else e.t_onset_receiver is not None)
                    )
                    lat, alert_lat, alerts_lost = [], [], 0
                    by_onset = {e.t_onset_device: e for e in eng.events if e.detector == det_name}
                    for s, e in truth:
                        hits = [t for t in onsets if s <= t <= e + TOL_S]
                        lat.append(hits[0] - s if hits else np.nan)
                        if hits and side == "wearable" and hits[0] in by_onset:
                            arr = by_onset[hits[0]].t_alert_arrival
                            if arr is None:
                                alerts_lost += 1
                            else:
                                alert_lat.append(arr - s)
                        per_seizure.append({"file": name, "policy": policy, "detector": det_name, "side": side, "seizure_start_s": s, "duration_s": e - s, "detected": bool(hits), "latency_s": hits[0] - s if hits else np.nan})
                    fa = sum(not any(s <= t <= e + TOL_S for s, e in truth) for t in onsets)
                    rows.append({"file": name, "policy": policy, "detector": det_name, "side": side, "seizures": len(truth), "detected": int(np.sum(np.isfinite(lat))), "false_alarms": fa, "hours": eng.t / 3600, "latency_s": [x for x in lat if np.isfinite(x)], "alert_latency_s": alert_lat, "alerts_lost": alerts_lost})
            s = eng.summary()
            log.warning("CHB-MIT %s policy=%s: CR %.1f, saving %.0f %%", name, policy, s["compression_ratio"], s["energy_saving_pct"])
    # final learned detector on all files, for the live demo
    all_f = [f for F, _ in win.values() for f in F]
    all_y = np.concatenate([Y for _, Y in win.values()])
    if all_y.sum() >= 3:
        final = LearnedDetector(name="seizure_learned", title="Seizure (learned)", critical=True, threshold=0.8, on_frames=2, off_frames=3).fit(all_f, all_y)
        with open(out / f"learned_seizure_{patient}.pkl", "wb") as f:
            pickle.dump(final, f)
    for r in rows:
        r["events"] = r.pop("seizures")
    agg = aggregate(rows, ["detector", "side", "policy"])
    return agg, pd.DataFrame(per_seizure), [(n, sz) for n, _, sz in recs]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_latency(det: pd.DataFrame, path: Path, loss: float) -> None:
    d = det[det["loss_rate"] == loss]
    dets = [x for x in DETECTOR_TRUTH if x in set(d["detector"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    series = [("wearable", "event", "on the wearable", "#16a34a")]
    series += [("receiver", p, f"receiver, {p} stream", c) for p, c in (("event", "#2563eb"), ("pulselink", "#9333ea"), ("raw", "#6b7280"))]
    w = 0.8 / (len(series) + 1)
    for i, (side, pol, lab, col) in enumerate(series):
        vals = [d[(d.detector == k) & (d.side == side) & (d.policy == pol)]["latency_median_s"].mean() for k in dets]
        ax.bar(np.arange(len(dets)) + i * w, vals, w, label=lab, color=col)
    al = d[(d.detector == "seizure") & (d.side == "wearable") & (d.policy == "event")]["alert_latency_median_s"].mean()
    if np.isfinite(al):
        ax.bar(dets.index("seizure") + len(series) * w, al, w, label="seizure alert at receiver", color="#dc2626")
    ax.set_xticks(np.arange(len(dets)) + w * len(series) / 2)
    ax.set_xticklabels(dets)
    ax.set_ylabel("median time from event onset (s)")
    ax.set_title(f"How fast is each use case detected? (packet loss {loss:.0%})")
    ax.legend(fontsize=8)
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
    fig.suptitle("Redundant alert packets: back-to-back copies die together in bursts; spacing them restores the benefit")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bandwidth(df: pd.DataFrame, path: Path) -> None:
    d = df[df.scenario == "bandwidth"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for pol, col in (("raw", "#6b7280"), ("dwt", "#2563eb"), ("event", "#16a34a")):
        g = d[(d.policy == pol) & (d.side == "receiver")]
        ax.semilogx(g["bandwidth_bps"], g["latency_median_s"], "o-", color=col, label=f"receiver-side, {pol} stream")
    g = d[(d.policy == "event") & (d.side == "wearable")]
    ax.semilogx(g["bandwidth_bps"], g["alert_latency_median_s"], "s--", color="#dc2626", label="on-device detection + alert")
    ax.set_xlabel("link capacity (bit/s)")
    ax.set_ylabel("seizure onset -> receiver knows (s)")
    ax.set_title("Detect on the wearable, send an alert: delay no longer depends on the link")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy(cost: pd.DataFrame, path: Path, loss: float) -> None:
    d = cost[cost.loss_rate == loss]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    cols = ["#6b7280", "#2563eb", "#9333ea", "#16a34a"]
    axes[0].bar(d["policy"], d["bitrate_bps"], color=cols)
    axes[0].set_ylabel("bit/s on air")
    axes[1].bar(d["policy"], d["prd_seizure"], color=cols)
    axes[1].set_ylabel("PRD during seizures (%)")
    axes[2].bar(d["policy"], d["energy_saving_pct"], color=cols)
    axes[2].set_ylabel("radio+compute saving vs raw (%)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Event-aware streaming: low rate most of the time, high fidelity when it matters (loss {loss:.0%})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_session(eng: StreamEngine, path: Path) -> None:
    recs = eng.records
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
    axes[0].set_ylabel("uV")
    axes[0].legend(loc="lower right", fontsize=8)
    codec_col = {"raw": "#6b7280", "dwt": "#2563eb", "pulselink": "#16a34a", "hifi": "#f59e0b", "mid": "#fb923c"}
    axes[1].bar([r["t"] - fsec / 2 for r in recs], [r["bytes"] for r in recs], width=fsec, color=[codec_col[r["codec"]] if r["status"] == "ok" else "#dc2626" for r in recs])
    axes[1].set_ylabel("bytes")
    for ax, d in zip(axes[2:], dets):
        sc = [r["device"][d.name]["score"] for r in recs]
        sc = [np.nan if v is None else v for v in sc]
        ax.plot([r["t"] for r in recs], sc, color=d.color, lw=1)
        ax.axhline(d.threshold, color="k", ls=":", lw=0.8)
        act = [r["t"] for r in recs if r["device"][d.name]["active"]]
        ax.scatter(act, [d.threshold] * len(act), s=6, color=d.color)
        ax.set_ylabel(d.title, rotation=0, ha="right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("One synthetic session: events, codec choice (amber = high fidelity) and detector scores")
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


def run_all(out: Path, model=None, runs: int = 12, minutes: float = 10.0, chbmit: str | None = None, chb_files: int = 4, download: bool = False, quick: bool = False):
    t0 = time.time()
    if quick:
        runs, minutes = min(runs, 3), min(minutes, 6.0)
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    policies = ["raw", "dwt", "pulselink", "event"] if model is not None else ["raw", "dwt", "event"]
    losses = (0.0, 0.05, 0.1, 0.2)
    print(f"[1/5] use-case detection: {runs} sessions x {minutes:g} min x {len(policies)} policies x {len(losses)} loss rates")
    det, cost = usecase_experiment(model, runs, minutes, policies, losses)
    det.to_csv(out / "usecase_detection.csv", index=False)
    cost.to_csv(out / "policy_cost.csv", index=False)
    print("[2/5] alert delivery")
    alerts = alert_delivery_experiment(trials=4000 if quick else 20_000)
    alerts.to_csv(out / "alert_delivery.csv", index=False)
    print("[3/5] link capacity and disconnection")
    bw = bandwidth_experiment(model, max(2, runs // 2), minutes)
    bw.to_csv(out / "bandwidth_latency.csv", index=False)
    print("[4/5] figures")
    plot_latency(det, out / "figures" / "latency_by_usecase.png", 0.05)
    plot_alerts(alerts, out / "figures" / "alert_delivery.png")
    plot_bandwidth(bw, out / "figures" / "receiver_latency_vs_bandwidth.png")
    plot_policy(cost, out / "figures" / "policy_tradeoff.png", 0.05)
    plot_session(synth_session(3, 6.0, model, "event", ChannelConfig(bandwidth_bps=16_000, loss_rate=0.05)), out / "figures" / "example_session.png")

    chb_txt = ""
    if chbmit:
        print(f"[5/5] CHB-MIT {chbmit}")
        try:
            agg, per, files = chbmit_experiment(chbmit, chb_files, download, model, out)
            agg.to_csv(out / f"chbmit_{chbmit}_detection.csv", index=False)
            per.to_csv(out / f"chbmit_{chbmit}_per_seizure.csv", index=False)
            chb_txt = (
                f"\n## CHB-MIT {chbmit} (real seizures, channels F7-T7 and F8-T8, 5 % packet loss)\n\n"
                f"Files: {', '.join(f'{n} ({len(s)} seizure(s))' for n, s in files)}. The learned detector is trained leave-one-file-out.\n\n"
                + _md(agg[["detector", "side", "policy", "events", "sensitivity", "false_alarms_per_h", "latency_median_s", "latency_p90_s", "alert_latency_median_s"]]) + "\n"
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("CHB-MIT failed")
            chb_txt = f"\n## CHB-MIT {chbmit}\n\nFailed: {exc}\n"
    else:
        print("[5/5] CHB-MIT skipped (add --chbmit chb01 --download)")

    loss = 0.05
    d = det[det.loss_rate == loss]
    wear = d[(d.side == "wearable") & (d.policy == "event")][["detector", "events", "sensitivity", "false_alarms_per_h", "latency_median_s", "latency_p90_s", "alert_latency_median_s", "alerts_lost"]]
    rx = d[d.side == "receiver"][["detector", "policy", "sensitivity", "false_alarms_per_h", "latency_median_s"]]
    c = cost[cost.loss_rate == loss][["policy", "compression_ratio", "bitrate_bps", "prd_delivered", "prd_seizure", "frames_lost", "tx_power_mw", "energy_saving_pct", "battery_h"]].copy()
    c["battery_days"] = c.pop("battery_h") / 24
    al = alerts[alerts.loss_rate.isin([0.1, 0.2, 0.4])].pivot_table(index=["burst_length", "spacing_packets", "loss_rate"], columns="copies", values="p_alert_lost").reset_index()
    al.columns = [str(x) if not isinstance(x, str) else x for x in al.columns]
    text = f"""# PulseLink real-time results

Synthetic sessions: {runs} x {minutes:g} min per setting, random events (focus, drowsiness, seizure,
movement artifact) after a 45 s calibration period, 2 channels at 100 Hz, 2 s frames, 16 kbit/s link.
Detectors are the rule-based ones in `realtime/detectors.py` with per-person baselines. Synthetic data shows
the system works end to end; it is **not** evidence of clinical performance.

## 1. Detection on the wearable (packet loss {loss:.0%}, event-aware policy)

Latency is from the true event onset. The seizure alert is a 12-byte packet sent 3 times.

{_md(wear)}

## 2. Same detectors at the receiver, on the decoded stream

{_md(rx)}

## 3. Cost of each transmission policy (packet loss {loss:.0%})

{_md(c)}

## 4. Probability that an alert never arrives (columns = redundant copies)

{_md(al)}

## 5. Seizure: when does the receiver know? (2 % loss)

{_md(bw[["scenario", "bandwidth_bps", "policy", "side", "sensitivity", "latency_median_s", "alert_latency_median_s"]])}
{chb_txt}
Energy numbers come from `config.EnergyConfig` (assumptions, not measurements); battery life assumes a
100 mAh, 3.7 V cell and an always-on 0.5 mW front end. Runtime {time.time() - t0:.0f} s.
"""
    (out / "summary.md").write_text(text)
    print(f"done in {time.time() - t0:.0f} s -> {out}")
    return text
