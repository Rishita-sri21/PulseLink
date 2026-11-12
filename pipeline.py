"""End-to-end PulseLink pipeline: data -> model -> transmitter -> channel ->
receiver -> quality assessment -> metrics. Used by ``main.py``, the
dashboard and the tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from communication.channel_simulator import ChannelSimulator
from compression.adaptive import AdaptiveController, Ladder, calibrate_ladder
from compression.autoencoder import ConvAutoencoder, load_model, save_model
from compression.codecs import AutoencoderCodec, Codec, CodecRegistry, RawCodec, WaveletCodec
from compression.train import train_autoencoder
from config import MODEL_DIR, ChannelConfig, PulseLinkConfig
from evaluation.classification import SleepStager
from evaluation.energy_model import energy
from evaluation.signal_metrics import all_metrics
from preprocessing.load_eeg import load_dataset
from reconstruction.decoder import Receiver
from reconstruction.quality_check import assess_reconstruction

log = logging.getLogger(__name__)
DEFAULT_MODEL = MODEL_DIR / "pulselink_ae.pt"


# --------------------------------------------------------------------------- #
# Data + model preparation
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    cfg: PulseLinkConfig
    data: dict
    train_idx: np.ndarray
    calib_idx: np.ndarray
    test_idx: np.ndarray
    model: ConvAutoencoder
    stager: SleepStager
    history: list = field(default_factory=list)
    ladder: Ladder | None = None

    @property
    def fs(self) -> float:
        return self.data["fs"]

    def split(self, name: str):
        idx = getattr(self, f"{name}_idx")
        return self.data["segments"][idx], self.data["labels"][idx]


def split_indices(data: dict, seed: int = 0):
    """Recording-wise split when there are >= 3 recordings, else chronological."""
    rec = data["recording_ids"]
    n = len(rec)
    ids = np.unique(rec)
    if len(ids) >= 3:
        test = np.flatnonzero(rec == ids[-1])
        rest = np.flatnonzero(rec != ids[-1])
        rng = np.random.default_rng(seed)
        rest = rng.permutation(rest)
        n_cal = max(4, len(rest) // 10)
        return np.sort(rest[n_cal:]), np.sort(rest[:n_cal]), test
    a, b = int(0.7 * n), int(0.8 * n)
    return np.arange(a), np.arange(a, b), np.arange(b, n)


def prepare(
    cfg: PulseLinkConfig | None = None,
    synthetic: bool | None = None,
    subjects=(0, 1, 2, 3),
    n_synthetic_epochs: int = 300,
    model_path: Path | None = DEFAULT_MODEL,
    retrain: bool = False,
    download: bool = False,
    calibrate: bool = True,
    progress=None,
) -> Prepared:
    cfg = cfg or PulseLinkConfig()
    data = load_dataset(subjects=subjects, cfg=cfg.pre, synthetic=synthetic, download=download, n_synthetic_epochs=n_synthetic_epochs)
    log.info("data: %s segments from %s (synthetic=%s)", data["segments"].shape, data["source"], data["is_synthetic"])
    tr, ca, te = split_indices(data)
    seg = data["segments"]

    model, history = None, []
    if model_path and Path(model_path).exists() and not retrain:
        model = load_model(model_path)
        if model.n_channels != seg.shape[1]:
            log.warning("saved model has %d channels, data has %d: retraining", model.n_channels, seg.shape[1])
            model = None
        else:
            history = model.extra.get("history", [])
    if model is None:
        model, history = train_autoencoder(seg[tr], data["fs"], cfg.ae, progress=progress)
        if model_path:
            save_model(model, model_path, {"history": history, "synthetic": data["is_synthetic"], "source": data["source"]})

    stager = SleepStager(data["fs"]).fit(seg[tr], data["labels"][tr])
    prep = Prepared(cfg, data, tr, ca, te, model, stager, history)
    if calibrate:
        prep.ladder = calibrate_ladder(ladder_codecs(model), seg[ca][:80], data["fs"])
    return prep


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def fixed_codecs(model: ConvAutoencoder) -> dict[str, Codec]:
    """The three methods compared on the dashboard / in the report."""
    return {
        "raw": RawCodec(),
        "dwt": WaveletCodec(keep_ratio=0.1, bits=8),
        "pulselink": AutoencoderCodec(model, bits=4, keep=model.cfg.latent_channels // 2),
    }


def ladder_codecs(model: ConvAutoencoder) -> list[Codec]:
    lc = model.cfg.latent_channels
    widths = sorted({max(1, lc // 8), max(1, lc // 4), lc // 2, lc})
    codecs: list[Codec] = [RawCodec()]
    codecs += [WaveletCodec(k, b) for k, b in [(0.05, 6), (0.1, 8), (0.2, 8), (0.3, 8), (0.5, 10)]]
    codecs += [AutoencoderCodec(model, b, w) for w in widths for b in (3, 4, 6)]
    return codecs


# --------------------------------------------------------------------------- #
# Streaming simulation
# --------------------------------------------------------------------------- #
@dataclass
class StreamResult:
    method: str
    reconstructed: np.ndarray
    segments: pd.DataFrame
    summary: dict


def run_stream(
    prep: Prepared,
    segments: np.ndarray,
    method: str | Codec = "adaptive",
    channel: ChannelConfig | None = None,
    bandwidth_schedule: list[tuple[float, float]] | None = None,
    playout_delay_s: float | None = None,
    verify: bool = False,
    labels: np.ndarray | None = None,
    controller_kw: dict | None = None,
) -> StreamResult:
    """Stream ``segments`` through transmitter -> channel -> receiver."""
    cfg = prep.cfg
    channel = channel or cfg.channel
    fs, seg_s = prep.fs, cfg.pre.segment_seconds
    sim = ChannelSimulator(channel, bandwidth_schedule)
    registry = CodecRegistry(prep.model)
    receiver = Receiver(registry, segments.shape[1], segments.shape[2], playout_delay_s or 2 * seg_s)
    raw_bits = segments.shape[1] * segments.shape[2] * cfg.pre.raw_bits_per_sample

    controller = None
    if method == "adaptive":
        controller = AdaptiveController(
            prep.ladder, fs, seg_s, cfg.energy, channel.mtu_bytes, channel.header_bytes,
            max_prd=cfg.quality.max_prd_percent, verify=verify, retries=channel.max_retries, **(controller_kw or {}),
        )
        name = "adaptive"
    else:
        codec = fixed_codecs(prep.model)[method] if isinstance(method, str) else method
        name = method if isinstance(method, str) else codec.mode_name()

    rows, enc_ops = [], 0
    for i, x in enumerate(segments):
        t = (i + 1) * seg_s  # segment is complete at the end of its window
        if controller:
            d = controller.decide(x, sim.bandwidth_at(t), sim.backlog_bytes(t), sim.link_up(t), sim.recent_loss_rate())
            enc, mode = d.encoded, d.mode.name
            enc_ops += enc.encode_ops + d.extra_encode_ops
            info = {"budget_bytes": d.budget_bytes, "target_prd": d.target_prd, "reasons": "; ".join(d.reasons), "input_issues": ",".join(d.input_issues), "events": ",".join(d.events)}
        else:
            enc = codec.encode(x, verify=verify and not isinstance(codec, RawCodec))
            mode = codec.mode_name()
            enc_ops += enc.encode_ops
            info = {}
        results = sim.send_segment(enc.blob, i, t, priority=bool(info.get("events")))
        receiver.receive(results)
        rows.append({"segment": i, "t": t, "mode": mode, "bytes": enc.nbytes, "cr": raw_bits / (8 * enc.nbytes), "prd_estimate": enc.prd_estimate, **info})

    received = receiver.reconstruct(range(len(segments)))
    recon = np.stack([r.signal for r in received])
    dec_ops = 0
    for row, r, x in zip(rows, received, segments):
        rep = assess_reconstruction(r, fs, cfg.quality)
        m = all_metrics(x, r.signal, fs)
        dec_ops += r.decode_ops
        row.update(status=r.status, latency_s=r.latency_s, late=r.late, reliable=rep.reliable, flags=",".join(rep.flags), missing=r.missing_fraction, **m)
    df = pd.DataFrame(rows)

    st = sim.stats
    e = energy(st.bytes_on_air, st.attempts, enc_ops, dec_ops, bytes_received=sum(r.packet.wire_bytes(channel.header_bytes) for r in sim.results if r.delivered), cfg=cfg.energy)
    ok = df["status"] == "ok"
    bad_truth = df["prdn"] > cfg.quality.max_prd_percent
    flagged = ~df["reliable"]
    summary = {
        "method": name,
        "segments": len(df),
        "compression_ratio": raw_bits * len(df) / (8 * df["bytes"].sum()),
        "mean_bytes": df["bytes"].mean(),
        "prdn_mean_all": df["prdn"].mean(),
        "prdn_mean_delivered": df.loc[ok, "prdn"].mean() if ok.any() else np.nan,
        "psd_diff_db": df["psd_diff_db"].mean(),
        "bp_err_mean": df["bp_err_mean"].mean(),
        "segments_lost": int((df["status"] == "lost").sum()),
        "segments_partial": int((df["status"] == "partial").sum()),
        "segments_late": int(df["late"].sum()),
        "packet_loss_rate": st.loss_rate,
        "packets": st.packets,
        "retransmissions": st.attempts - (st.packets - st.overflow),
        "buffer_overflow": st.overflow,
        "buffered_by_outage": st.buffered_by_outage,
        "mean_latency_s": df["latency_s"].mean(),
        "max_latency_s": df["latency_s"].max(),
        "reliable_fraction": df["reliable"].mean(),
        # How good are the receiver's flags (which never see the original)?
        "flag_recall": float((flagged & bad_truth).sum() / bad_truth.sum()) if bad_truth.any() else 1.0,
        "flag_precision": float((flagged & bad_truth).sum() / flagged.sum()) if flagged.any() else 1.0,
        **{f"energy_{k}": v for k, v in e.as_dict().items()},
        "transmitter_energy_per_segment_mj": 1e3 * e.transmitter_j / len(df),
    }
    if labels is not None:
        summary.update({f"stage_{k}": v for k, v in prep.stager.score(recon, labels).items()})
    return StreamResult(name, recon, df, summary)


def channel_with(cfg: ChannelConfig, **kw) -> ChannelConfig:
    return replace(cfg, **kw)
