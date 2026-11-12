"""Experiments comparing raw transmission, the DWT baseline and PulseLink.

Each function returns a tidy ``pandas.DataFrame``; ``plot_*`` helpers turn
them into the figures saved by ``main.py``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compression.codecs import AutoencoderCodec, Codec, RawCodec, WaveletCodec  # noqa: E402
from evaluation.energy_model import savings_percent, segment_energy  # noqa: E402
from evaluation.signal_metrics import all_metrics, psd  # noqa: E402


def _family(codec: Codec) -> str:
    return {RawCodec: "raw", WaveletCodec: "dwt", AutoencoderCodec: "pulselink"}[type(codec)]


def codec_benchmark(prep, segments: np.ndarray, labels: np.ndarray | None, codecs: list[Codec] | None = None) -> pd.DataFrame:
    """Ideal channel: rate, distortion, frequency preservation, task accuracy, energy."""
    from pipeline import ladder_codecs

    cfg = prep.cfg
    codecs = codecs or ladder_codecs(prep.model)
    raw_bits = segments.shape[1] * segments.shape[2] * cfg.pre.raw_bits_per_sample
    rows, raw_tx = [], None
    for codec in codecs:
        recon, sizes, mets, e_ops, d_ops = [], [], [], [], []
        for x in segments:
            enc = codec.encode(x)
            y = codec.decode(enc.blob)
            recon.append(y)
            sizes.append(enc.nbytes)
            e_ops.append(enc.encode_ops)
            d_ops.append(codec.decode_ops(*x.shape))
            mets.append(all_metrics(x, y, prep.fs))
        m = pd.DataFrame(mets).mean().to_dict()
        nbytes = float(np.mean(sizes))
        e = segment_energy(int(round(nbytes)), int(np.mean(e_ops)), int(np.mean(d_ops)), cfg.channel.mtu_bytes, cfg.channel.header_bytes, cfg.energy)
        row = {
            "method": codec.mode_name(),
            "family": _family(codec),
            "bytes_per_segment": nbytes,
            "compression_ratio": raw_bits / (8 * nbytes),
            "bits_per_sample": 8 * nbytes / (segments.shape[1] * segments.shape[2]),
            **m,
            "encode_mops": np.mean(e_ops) / 1e6,
            "decode_mops": np.mean(d_ops) / 1e6,
            "tx_radio_mj": 1e3 * e.transmission_j,
            "tx_compute_mj": 1e3 * e.compression_j,
            "transmitter_mj": 1e3 * e.transmitter_j,
            "receiver_mj": 1e3 * e.receiver_j,
            "net_mj": 1e3 * e.net_j,
        }
        if labels is not None:
            row.update({f"stage_{k}": v for k, v in prep.stager.score(np.stack(recon), labels).items()})
        if isinstance(codec, RawCodec):
            raw_tx = row["transmitter_mj"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if raw_tx is not None:
        df["transmitter_saving_pct"] = [savings_percent(v, raw_tx) for v in df["transmitter_mj"]]
    return df


def loss_sweep(prep, segments, labels=None, methods=("raw", "dwt", "pulselink", "adaptive"), loss_rates=(0.0, 0.02, 0.05, 0.10), **channel_kw) -> pd.DataFrame:
    from pipeline import run_stream

    rows = []
    for loss in loss_rates:
        ch = replace(prep.cfg.channel, loss_rate=loss, **channel_kw)
        for m in methods:
            s = run_stream(prep, segments, m, ch, labels=labels).summary
            rows.append({"loss_rate": loss, **s})
    return pd.DataFrame(rows)


def bandwidth_sweep(prep, segments, labels=None, methods=("raw", "dwt", "pulselink", "adaptive"), bandwidths=(800, 1600, 3200, 6400, 64000), **channel_kw) -> pd.DataFrame:
    from pipeline import run_stream

    rows = []
    for bw in bandwidths:
        ch = replace(prep.cfg.channel, bandwidth_bps=float(bw), **channel_kw)
        for m in methods:
            s = run_stream(prep, segments, m, ch, labels=labels).summary
            rows.append({"bandwidth_bps": bw, **s})
    return pd.DataFrame(rows)


def outage_experiment(prep, segments, labels=None, methods=("raw", "adaptive"), outage=(120.0, 120.0), buffers=(2_000, 256_000)) -> pd.DataFrame:
    """Disconnect the link for ``outage`` = (start, duration) seconds and
    compare a tiny buffer (data discarded) with a real buffer (data kept and
    flushed after reconnection)."""
    from pipeline import run_stream

    rows = []
    for buf in buffers:
        ch = replace(prep.cfg.channel, outages=[outage], buffer_bytes=buf)
        for m in methods:
            s = run_stream(prep, segments, m, ch, labels=labels).summary
            rows.append({"buffer_bytes": buf, **s})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
_COLORS = {"raw": "#6b7280", "dwt": "#2563eb", "pulselink": "#dc2626", "adaptive": "#16a34a"}


def plot_rate_distortion(bench: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    lossy = bench[bench["family"] != "raw"]
    for fam, g in lossy.groupby("family"):
        g = g.sort_values("compression_ratio")
        axes[0].plot(g["compression_ratio"], g["prdn"], "o-", color=_COLORS[fam], label=fam)
        axes[1].plot(g["compression_ratio"], g["psd_diff_db"], "o-", color=_COLORS[fam], label=fam)
        axes[2].plot(g["compression_ratio"], g["transmitter_mj"], "o-", color=_COLORS[fam], label=fam)
    raw = bench[bench["family"] == "raw"]
    if len(raw):
        axes[2].axhline(raw["transmitter_mj"].iloc[0], color=_COLORS["raw"], ls="--", label="raw")
    for ax, yl in zip(axes, ["PRDN (%)", "PSD difference (dB)", "Transmitter energy / segment (mJ)"]):
        ax.set_xscale("log")
        ax.set_xlabel("Compression ratio")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Rate-distortion-energy trade-off (ideal channel)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep(df: pd.DataFrame, x: str, path: Path, xlabel: str, logx: bool = False) -> None:
    metrics = [("prdn_mean_all", "PRDN incl. lost segments (%)"), ("reliable_fraction", "Reliable segments"), ("transmitter_energy_per_segment_mj", "Transmitter energy / segment (mJ)")]
    if "stage_accuracy" in df:
        metrics.append(("stage_accuracy", "Sleep-stage accuracy"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4))
    for ax, (col, label) in zip(axes, metrics):
        for m, g in df.groupby("method"):
            ax.plot(g[x], g[col], "o-", color=_COLORS.get(m, None), label=m)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        if logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_example(x: np.ndarray, recons: dict[str, np.ndarray], fs: float, path: Path, channel_names=None, seconds: float = 10.0) -> None:
    n = int(seconds * fs)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(t, x[0, :n], color="k", lw=1.2, label="original")
    for name, y in recons.items():
        fam = name.split("(")[0]
        axes[0].plot(t, y[0, :n], lw=0.9, color=_COLORS.get(fam), label=name, alpha=0.85)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(f"{(channel_names or ['ch0'])[0]} (uV)")
    axes[0].legend(loc="upper right", fontsize=8)
    f, p = psd(x[0], fs)
    axes[1].semilogy(f, p, color="k", lw=1.5, label="original")
    for name, y in recons.items():
        fam = name.split("(")[0]
        _, ph = psd(y[0], fs)
        axes[1].semilogy(f, ph, color=_COLORS.get(fam), lw=1, label=name)
    axes[1].set_xlim(0, 45)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (uV^2/Hz)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_training(history: list[dict], path: Path) -> None:
    if not history:
        return
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(h["epoch"], h["time_mse"], label="train")
    if "val_time_mse" in h:
        axes[0].plot(h["epoch"], h["val_time_mse"], label="val")
    axes[0].set_title("Time-domain MSE (normalised)")
    axes[1].plot(h["epoch"], h["freq_l1"], label="train")
    if "val_freq_l1" in h:
        axes[1].plot(h["epoch"], h["val_freq_l1"], label="val")
    axes[1].set_title("Log-PSD L1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
