"""PulseLink command-line entry point.

Examples::

    python main.py run --quick           # whole pipeline in ~2 min (synthetic if no data)
    python main.py run --download        # fetch Sleep-EDF subjects 0-3 first
    python main.py train --retrain       # (re)train the autoencoder only
    python main.py stream --loss 0.05 --bandwidth 2000 --method adaptive
    streamlit run dashboard/app.py       # interactive demo
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_DIR, RESULTS_DIR, PulseLinkConfig  # noqa: E402
from evaluation import comparison as cmp  # noqa: E402
from pipeline import DEFAULT_MODEL, fixed_codecs, prepare, run_stream  # noqa: E402

log = logging.getLogger("pulselink")


def _config(args) -> PulseLinkConfig:
    cfg = PulseLinkConfig()
    if args.epochs:
        cfg.ae.epochs = args.epochs
    elif args.quick:
        cfg.ae.epochs = 8
    return cfg


def _prepare(args, cfg):
    synthetic = True if args.synthetic else (False if args.real else None)
    model_path = Path(args.model) if args.model else (MODEL_DIR / "pulselink_ae_quick.pt" if args.quick else DEFAULT_MODEL)
    return prepare(
        cfg,
        synthetic=synthetic,
        subjects=tuple(args.subjects),
        recording=tuple(args.nights),
        n_synthetic_epochs=120 if args.quick else 300,
        model_path=model_path,
        retrain=args.retrain,
        download=args.download,
    )


def cmd_train(args):
    cfg = _config(args)
    prep = _prepare(args, cfg)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cmp.plot_training(prep.history, out / "figures" / "training.png")
    print(pd.DataFrame(prep.history).tail(3).to_string(index=False))


def cmd_stream(args):
    cfg = _config(args)
    prep = _prepare(args, cfg)
    X, y = prep.split("test")
    X, y = X[: args.segments], y[: args.segments]
    ch = cfg.channel
    ch.loss_rate, ch.bandwidth_bps, ch.max_retries, ch.burst_length = args.loss, args.bandwidth, args.retries, args.burst
    if args.outage:
        ch.outages = [tuple(args.outage)]
    res = run_stream(prep, X, args.method, ch, labels=y)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.summary.items()}, indent=2, default=str))
    print(res.segments[["segment", "mode", "bytes", "status", "prdn", "reliable", "flags"]].head(20).to_string(index=False))


def cmd_run(args):
    t0 = time.time()
    cfg = _config(args)
    out = Path(args.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading data, training / loading the autoencoder, calibrating the adaptive ladder ...")
    prep = _prepare(args, cfg)
    X, y = prep.split("test")
    n = min(len(X), args.segments)
    X, y = X[:n], y[:n]
    src = "synthetic EEG" if prep.data["is_synthetic"] else prep.data["source"]
    print(f"      data: {src}; {prep.data['segments'].shape[0]} segments of {X.shape[1]} ch x {X.shape[2]} samples; test on {n}")
    cmp.plot_training(prep.history, fig_dir / "training.png")
    ladder = pd.DataFrame([{"mode": m.name, "mean_bytes": m.mean_bytes, "mean_prd": m.mean_prd, "encode_mops": m.mean_encode_ops / 1e6} for m in prep.ladder])
    ladder.to_csv(out / "adaptive_ladder.csv", index=False)

    print("[2/6] Codec benchmark on an ideal channel (raw vs DWT vs PulseLink) ...")
    bench = cmp.codec_benchmark(prep, X, y)
    bench.to_csv(out / "codec_benchmark.csv", index=False)
    cmp.plot_rate_distortion(bench, fig_dir / "rate_distortion.png")
    orig_score = prep.stager.score(X, y)

    fixed = fixed_codecs(prep.model)
    x0 = X[min(5, n - 1)]
    cmp.plot_example(x0, {c.mode_name(): c.decode(c.encode(x0).blob) for k, c in fixed.items() if k != "raw"}, prep.fs, fig_dir / "example_reconstruction.png", prep.data["channels"])

    print("[3/6] Packet-loss sweep (0 / 2 / 5 / 10 %) without and with ARQ ...")
    loss = cmp.loss_sweep(prep, X, y, max_retries=0)
    loss.to_csv(out / "loss_sweep.csv", index=False)
    cmp.plot_sweep(loss, "loss_rate", fig_dir / "loss_sweep.png", "Packet loss rate")
    loss_arq = cmp.loss_sweep(prep, X, y, max_retries=2)
    loss_arq.to_csv(out / "loss_sweep_arq.csv", index=False)
    cmp.plot_sweep(loss_arq, "loss_rate", fig_dir / "loss_sweep_arq.png", "Packet loss rate (ARQ, 2 retries)")

    print("[4/6] Bandwidth sweep ...")
    bw = cmp.bandwidth_sweep(prep, X, y)
    bw.to_csv(out / "bandwidth_sweep.csv", index=False)
    cmp.plot_sweep(bw, "bandwidth_bps", fig_dir / "bandwidth_sweep.png", "Link bandwidth (bit/s)", logx=True)

    print("[5/6] Outage + buffering experiment ...")
    dur = n * cfg.pre.segment_seconds
    outage = cmp.outage_experiment(prep, X, y, outage=(0.25 * dur, 0.25 * dur))
    outage.to_csv(out / "outage_buffering.csv", index=False)

    print("[6/6] Writing summary ...")
    summary = write_summary(out, prep, bench, loss, loss_arq, bw, outage, orig_score, n)
    print(summary)
    print(f"Done in {time.time() - t0:.0f} s. Results in {out.resolve()}")


def _md(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df]
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}")
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.values)


def write_summary(out, prep, bench, loss, loss_arq, bw, outage, orig_score, n) -> str:
    src = "synthetic EEG (Sleep-EDF not found locally)" if prep.data["is_synthetic"] else prep.data["source"]
    parts = [
        "# PulseLink results\n",
        f"Data: {src}. Test segments: {n} x 30 s, {len(prep.data['channels'])} channels at {prep.fs:g} Hz.\n",
        f"Sleep-stage classifier on original test EEG: accuracy {orig_score['accuracy']:.3f}, kappa {orig_score['kappa']:.3f}.\n",
        "## Codec benchmark (ideal channel)\n",
        _md(bench, ["method", "compression_ratio", "prdn", "snr_db", "psd_diff_db", "bp_err_mean", "stage_accuracy", "encode_mops", "transmitter_mj", "transmitter_saving_pct"]),
        "\n\n## Packet loss, no retransmission\n",
        _md(loss, ["loss_rate", "method", "compression_ratio", "prdn_mean_all", "segments_lost", "reliable_fraction", "flag_recall", "stage_accuracy", "transmitter_energy_per_segment_mj"]),
        "\n\n## Packet loss, ARQ with 2 retries\n",
        _md(loss_arq, ["loss_rate", "method", "compression_ratio", "prdn_mean_all", "segments_lost", "retransmissions", "stage_accuracy", "transmitter_energy_per_segment_mj"]),
        "\n\n## Bandwidth limits\n",
        _md(bw, ["bandwidth_bps", "method", "compression_ratio", "prdn_mean_all", "segments_lost", "buffer_overflow", "max_latency_s", "stage_accuracy"]),
        "\n\n## Outage with and without buffering\n",
        _md(outage, ["buffer_bytes", "method", "segments_lost", "buffered_by_outage", "buffer_overflow", "max_latency_s", "prdn_mean_all"]),
        "\n\nEnergy numbers come from the transparent model in `evaluation/energy_model.py` "
        "with the assumptions in `config.EnergyConfig`; they are estimates, not measurements.\n",
    ]
    text = "\n".join(parts)
    (Path(out) / "summary.md").write_text(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description="PulseLink: energy-aware, frequency-preserving EEG compression")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--synthetic", action="store_true", help="force synthetic EEG")
        sp.add_argument("--real", action="store_true", help="require Sleep-EDF (no fallback)")
        sp.add_argument("--download", action="store_true", help="download Sleep-EDF first (PhysioNet, or its AWS mirror)")
        sp.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3])
        sp.add_argument("--nights", type=int, nargs="+", default=[1], choices=[1, 2], help="Sleep Cassette nights per subject")
        sp.add_argument("--quick", action="store_true", help="small data + short training")
        sp.add_argument("--epochs", type=int, default=None)
        sp.add_argument("--retrain", action="store_true")
        sp.add_argument("--model", default=None, help="model checkpoint path")
        sp.add_argument("--segments", type=int, default=120, help="max test segments to stream")
        sp.add_argument("--out", default=str(RESULTS_DIR))
        sp.add_argument("-v", "--verbose", action="store_true")

    common(sub.add_parser("run", help="full pipeline + all experiments"))
    common(sub.add_parser("train", help="train / load the autoencoder"))
    s = sub.add_parser("stream", help="stream test segments through one channel setting")
    common(s)
    s.add_argument("--method", default="adaptive", choices=["raw", "dwt", "pulselink", "adaptive"])
    s.add_argument("--loss", type=float, default=0.0)
    s.add_argument("--burst", type=float, default=1.0)
    s.add_argument("--bandwidth", type=float, default=64000.0)
    s.add_argument("--retries", type=int, default=0)
    s.add_argument("--outage", type=float, nargs=2, metavar=("START", "DURATION"))

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    np.set_printoptions(precision=3, suppress=True)
    {"run": cmd_run, "train": cmd_train, "stream": cmd_stream}[args.cmd](args)


if __name__ == "__main__":
    main()
