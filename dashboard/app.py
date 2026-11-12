"""PulseLink Streamlit dashboard.

Run from the repository root::

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import PulseLinkConfig  # noqa: E402
from evaluation.energy_model import savings_percent  # noqa: E402
from evaluation.signal_metrics import psd  # noqa: E402
from pipeline import DEFAULT_MODEL, prepare, run_stream  # noqa: E402

st.set_page_config(page_title="PulseLink", layout="wide")
COLORS = {"raw": "#6b7280", "dwt": "#2563eb", "pulselink": "#dc2626", "adaptive": "#16a34a"}
LABELS = {"raw": "Raw transmission", "dwt": "Wavelet (DWT) baseline", "pulselink": "PulseLink autoencoder (fixed rate)", "adaptive": "PulseLink adaptive"}


@st.cache_resource(show_spinner="Loading EEG, loading/training the autoencoder, calibrating ...")
def load(source: str):
    synthetic = {"auto": None, "synthetic": True, "sleep-edf": False}[source]
    cfg = PulseLinkConfig()
    model_path = DEFAULT_MODEL
    if not model_path.exists():
        cfg.ae.epochs = 8  # first launch without a trained model: quick training
    return prepare(cfg, synthetic=synthetic, model_path=model_path)


st.title("PulseLink")
st.caption("Energy-aware, frequency-preserving EEG compression under constrained network conditions")

with st.sidebar:
    st.header("1. Recording")
    source = st.selectbox("Data source", ["auto", "synthetic", "sleep-edf"], help="auto = Sleep-EDF if downloaded, otherwise synthetic EEG")
    try:
        prep = load(source)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load data: {exc}")
        st.stop()
    data = prep.data
    rec_ids = np.unique(data["recording_ids"])
    test_rec = int(data["recording_ids"][prep.test_idx[0]])
    rec = st.selectbox("Recording", rec_ids, index=int(np.where(rec_ids == test_rec)[0][0]), format_func=lambda r: f"#{r}" + (" (held out)" if r == test_rec else " (seen in training)"))
    idx = np.flatnonzero(data["recording_ids"] == rec)
    n_seg = st.slider("Segments to stream (30 s each)", 5, min(120, len(idx)), min(30, len(idx)))
    start = st.slider("Start segment", 0, max(0, len(idx) - n_seg), 0)
    seg_idx = idx[start : start + n_seg]
    X, y = data["segments"][seg_idx], data["labels"][seg_idx]

    st.header("2. Method")
    method = st.radio("Transmission", list(LABELS), format_func=LABELS.get, index=3)

    st.header("3. Channel")
    bw = st.select_slider("Bandwidth (bit/s)", options=[400, 800, 1600, 3200, 6400, 16000, 64000, 250000], value=64000)
    loss = st.slider("Packet loss rate", 0.0, 0.3, 0.0, 0.01)
    burst = st.slider("Mean loss burst length (packets)", 1.0, 10.0, 1.0, 0.5)
    retries = st.slider("ARQ retransmissions", 0, 5, 0)
    use_outage = st.checkbox("Simulate a disconnection")
    outages = []
    if use_outage:
        dur = n_seg * prep.cfg.pre.segment_seconds
        o_start = st.slider("Outage start (s)", 0.0, dur, dur * 0.3)
        o_len = st.slider("Outage length (s)", 0.0, dur, dur * 0.2)
        outages = [(o_start, o_len)]
    buffer_kb = st.slider("Transmitter buffer (kB)", 1, 512, 256)

channel = replace(prep.cfg.channel, bandwidth_bps=float(bw), loss_rate=loss, burst_length=burst, max_retries=retries, outages=outages, buffer_bytes=buffer_kb * 1000)


@st.cache_data(show_spinner="Simulating transmission ...")
def simulate(method, channel_key, seg_key, _X, _y, _channel):
    r = run_stream(prep, _X, method, _channel, labels=_y)
    return r.summary, r.segments, r.reconstructed


ch_key = (bw, loss, burst, retries, tuple(outages), buffer_kb)
seg_key = (source, int(rec), start, n_seg)
results = {m: simulate(m, ch_key, seg_key, X, y, channel) for m in LABELS}
summary, segdf, recon = results[method]
raw_summary = results["raw"][0]

# ---- headline metrics ---------------------------------------------------- #
c = st.columns(6)
c[0].metric("Compression ratio", f"{summary['compression_ratio']:.1f}x")
c[1].metric("PRDN (incl. losses)", f"{summary['prdn_mean_all']:.1f} %")
c[2].metric("PSD difference", f"{summary['psd_diff_db']:.2f} dB")
c[3].metric("Reliable segments", f"{100 * summary['reliable_fraction']:.0f} %")
saving = savings_percent(summary["energy_transmitter_j"], raw_summary["energy_transmitter_j"])
c[4].metric("Tx energy / segment", f"{summary['transmitter_energy_per_segment_mj']:.2f} mJ", f"{-saving:+.0f} % vs raw" if method != "raw" else None, delta_color="inverse")
if "stage_accuracy" in summary:
    c[5].metric("Sleep-stage accuracy", f"{100 * summary['stage_accuracy']:.0f} %")
if data["is_synthetic"]:
    st.info("Showing synthetic EEG because Sleep-EDF is not downloaded. Run `python main.py run --download` to fetch it.")

# ---- waveform + spectrum ------------------------------------------------- #
left, right = st.columns([3, 2])
with left:
    view = st.slider("Segment to inspect", 0, n_seg - 1, min(3, n_seg - 1))
    ch = st.selectbox("Channel", range(X.shape[1]), format_func=lambda i: data["channels"][i])
    secs = st.slider("Seconds shown", 2, 30, 10)
    row = segdf.iloc[view]
    n = int(secs * prep.fs)
    t = np.arange(n) / prep.fs
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(t, X[view, ch, :n], color="k", lw=1.1, label="original")
    ax.plot(t, recon[view, ch, :n], color=COLORS[method], lw=1.0, alpha=0.85, label=f"reconstructed ({row['mode']})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("uV")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    status = "reliable" if row["reliable"] else "FLAGGED as unreliable"
    st.write(f"**Segment {view}**: status `{row['status']}`, {status}" + (f", flags: `{row['flags']}`" if row["flags"] else "") + f". True PRDN {row['prdn']:.1f} %" + (f", encoder-side estimate {row['prd_estimate']:.1f} %" if pd.notna(row.get("prd_estimate")) else ""))
    if method == "adaptive":
        st.caption(f"Controller: {row['reasons']}")
with right:
    f, p = psd(X[view, ch], prep.fs)
    _, ph = psd(recon[view, ch], prep.fs)
    fig, ax = plt.subplots(figsize=(6, 3.9))
    ax.semilogy(f, p, color="k", label="original")
    ax.semilogy(f, ph + 1e-12, color=COLORS[method], label="reconstructed")
    ax.set_xlim(0, 45)
    ax.set_ylim(max(p[(f > 0.5) & (f < 45)].min() / 30, 1e-6), p.max() * 5)
    for name, (lo, hi) in {"δ": (0.5, 4), "θ": (4, 8), "α": (8, 12), "σ": (12, 15), "β": (15, 30)}.items():
        ax.axvspan(lo, hi, alpha=0.06, color="gray")
        ax.text((lo + hi) / 2, p.max() * 2, name, ha="center", fontsize=9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV²/Hz)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---- comparison table ---------------------------------------------------- #
st.subheader("All methods under the current channel")
comp = pd.DataFrame([
    {
        "method": LABELS[m],
        "compression ratio": s["compression_ratio"],
        "PRDN incl. losses (%)": s["prdn_mean_all"],
        "PSD diff (dB)": s["psd_diff_db"],
        "band-power error": s["bp_err_mean"],
        "segments lost": s["segments_lost"],
        "reliable (%)": 100 * s["reliable_fraction"],
        "sleep-stage acc (%)": 100 * s.get("stage_accuracy", np.nan),
        "tx energy (mJ/seg)": s["transmitter_energy_per_segment_mj"],
        "energy saving vs raw (%)": savings_percent(s["energy_transmitter_j"], raw_summary["energy_transmitter_j"]),
        "max latency (s)": s["max_latency_s"],
    }
    for m, (s, _, _) in results.items()
])
st.dataframe(comp.style.format(precision=2), hide_index=True)

# ---- per-segment timeline ------------------------------------------------ #
st.subheader("Per-segment transmission log")
fig, ax = plt.subplots(figsize=(12, 2.8))
colors = ["#16a34a" if r else "#dc2626" for r in segdf["reliable"]]
ax.bar(segdf["segment"], segdf["bytes"], color=colors)
ax.set_yscale("log")
ax.set_xlabel("Segment")
ax.set_ylabel("Bytes sent")
ax.set_title("Green = reliable, red = flagged by the receiver")
for o_s, o_l in outages:
    seg_s = prep.cfg.pre.segment_seconds
    ax.axvspan(o_s / seg_s - 1, (o_s + o_l) / seg_s - 1, color="orange", alpha=0.2, label="outage")
ax.grid(alpha=0.3, axis="y")
st.pyplot(fig)
cols = ["segment", "mode", "bytes", "cr", "status", "latency_s", "prdn", "psd_diff_db", "prd_estimate", "reliable", "flags"] + (["reasons"] if method == "adaptive" else [])
st.dataframe(segdf[[c for c in cols if c in segdf]], hide_index=True)

with st.expander("Energy model assumptions"):
    st.json(prep.cfg.energy.__dict__)
    st.write("E_net = E_transmission + E_compression + E_reconstruction. Transmitter energy (radio + MCU) is the battery-critical number; receiver energy is reported separately. These are model estimates, not measurements.")
