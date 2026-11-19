"""Task-aware fine-tuning of the PulseLink autoencoder for real-time frames.

The base model was trained on 30 s sleep segments with a time + spectrum
loss. In the real-time system it compresses 2 s frames, and what matters
at the receiver is that the *detectors still work* on the reconstruction.
The old model lost enough beta power that the focus detector mostly failed.

This script fine-tunes the model on 2 s frames with an extra **task loss**:
the error in the features the detectors use (log band powers, line length,
the theta/alpha and engagement ratios). The frames come from the synthetic
source (every brain state the demo shows) plus, if available, real CHB-MIT
EEG from a patient that is *not* used in the evaluation (chb02 by default).

    python -m compression.train_streaming            # about 1-3 min on a laptop CPU
    python -m compression.train_streaming --epochs 5 --no-real   # quick, offline

Output: ``models/pulselink_ae_stream.pt`` (used automatically by the
real-time engine and the live demo when present).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression.autoencoder import load_model, pulselink_loss, save_model, smoothed_psd
from config import MODEL_DIR
from preprocessing.segmentation import normalize_segments

log = logging.getLogger(__name__)
STREAM_MODEL = MODEL_DIR / "pulselink_ae_stream.pt"
BASE_MODEL = MODEL_DIR / "pulselink_ae.pt"
BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 15.0), (15.0, 30.0), (30.0, 40.0))


def synthetic_frames(n_minutes: float = 40.0, fs: float = 100.0, frame_s: float = 2.0, seed: int = 0) -> np.ndarray:
    """Band-passed 2 s frames covering every state of the live demo."""
    from realtime.engine import StreamingBandpass
    from realtime.sources import LiveSynthSource, random_schedule

    out = []
    per = 6.0
    for k in range(int(n_minutes / per)):
        rng = np.random.default_rng(seed + k)
        src = LiveSynthSource(fs=fs, seed=seed + k, schedule=random_schedule(rng, per * 60), base=["relaxed", "focus", "drowsy", "sleep"][k % 4])
        filt = StreamingBandpass(fs, src.c)
        n = int(frame_s * fs)
        for _ in range(int(per * 60 / frame_s)):
            out.append(filt(src.read(n).data))
    return np.stack(out).astype(np.float32)


def real_frames(patient: str = "chb02", fs: float = 100.0, frame_s: float = 2.0, max_frames: int = 3000) -> np.ndarray | None:
    """2 s frames from one CHB-MIT file of a patient not used for evaluation."""
    try:
        from realtime.engine import StreamingBandpass
        from realtime.sources import DEFAULT_CHB_CHANNELS, fetch_chbmit, load_edf

        files = fetch_chbmit(patient, n_seizure_files=0, n_normal_files=1, download=True)
        if not files:
            return None
        data, _ = load_edf(files[0][0], DEFAULT_CHB_CHANNELS, fs)
        x = StreamingBandpass(fs, data.shape[0])(data)
        n = int(frame_s * fs)
        frames = np.stack([x[:, i : i + n] for i in range(0, x.shape[1] - n, n)])
        keep = np.abs(frames).max(axis=(1, 2)) < 500  # drop saturated frames
        return frames[keep][:max_frames].astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        log.warning("no real frames (%s); synthetic only", exc)
        return None


def task_features(x: torch.Tensor, fs: float) -> torch.Tensor:
    """Differentiable version of the detectors' features, per frame and channel."""
    p = smoothed_psd(x, fs, band=(0.5, 40.0), resolution_hz=0.5)  # (B, C, F)
    freqs = torch.arange(p.shape[-1], dtype=x.dtype) * 0.5 + 0.5
    bands = torch.stack([p[..., (freqs >= lo) & (freqs < hi)].sum(-1) for lo, hi in BANDS], -1) + 1e-4
    lb = torch.log(bands)
    ll = torch.log(x.diff(dim=-1).abs().mean(-1) + 1e-4)[..., None]
    ratios = torch.stack([lb[..., 1] - lb[..., 2], torch.log(bands[..., 4]) - torch.log(bands[..., 1] + bands[..., 2])], -1)
    return torch.cat([lb, ll, ratios], -1)


def finetune(model, frames: np.ndarray, fs: float = 100.0, epochs: int = 8, lr: float = 1e-3, gamma: float = 0.5, bits: int = 4, keep: int | None = None, seed: int = 0, progress=None):
    """Fine-tune ``model`` in place. Loss = time MSE + spectrum + gamma * task."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = model.cfg
    keep = keep or cfg.latent_channels // 2
    xn, _, _ = normalize_segments(frames)
    xn = torch.as_tensor(np.clip(xn, -8, 8).astype(np.float32))
    idx = rng.permutation(len(xn))
    n_val = max(16, len(xn) // 10)
    val, tr = xn[idx[:n_val]], xn[idx[n_val:]]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(tr))
        tot = 0.0
        for i in range(0, len(perm), 32):
            xb = tr[perm[i : i + 32]]
            if rng.random() < 0.5:
                xb = -xb
            b = bits if rng.random() < 0.6 else int(rng.choice((3, 4, 5, 6)))
            k = keep if rng.random() < 0.6 else int(rng.choice((keep // 2, keep, cfg.latent_channels)))
            xh = model(xb, b, k)
            base, _, _ = pulselink_loss(xb, xh, fs, cfg.alpha, cfg.beta)
            task = F.smooth_l1_loss(task_features(xh, fs), task_features(xb, fs))
            loss = base + gamma * task
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            vh = model(val, bits, keep)
            vb, vt, vf = pulselink_loss(val, vh, fs, cfg.alpha, cfg.beta)
            vtask = F.smooth_l1_loss(task_features(vh, fs), task_features(val, fs)).item()
        rec = {"epoch": ep + 1, "train_loss": tot / len(tr), "val_time_mse": vt.item(), "val_freq_l1": vf.item(), "val_task": vtask, "seconds": time.time() - t0}
        hist.append(rec)
        log.warning("epoch %d/%d %s", ep + 1, epochs, {k: round(v, 4) for k, v in rec.items()})
        if progress:
            progress(rec)
    model.eval()
    return model, hist


def feature_error(model, frames: np.ndarray, fs: float = 100.0, bits: int = 4, keep: int | None = None) -> dict:
    """Mean absolute error of the task features (natural-log units) and PRD at the operating point."""
    from compression.codecs import AutoencoderCodec, prd

    codec = AutoencoderCodec(model, bits=bits, keep=keep or model.cfg.latent_channels // 2)
    errs, prds, nbytes = [], [], []
    for x in frames:
        enc = codec.encode(x.astype(np.float64))
        y = codec.decode(enc.blob)
        a = task_features(torch.as_tensor(x[None], dtype=torch.float32), fs)
        b = task_features(torch.as_tensor(y[None], dtype=torch.float32), fs)
        errs.append((a - b).abs().mean(dim=(0, 1)).numpy())
        prds.append(prd(x, y))
        nbytes.append(enc.nbytes)
    e = np.mean(errs, axis=0)
    names = ["delta", "theta", "alpha", "sigma", "beta", "gamma", "line_length", "theta_alpha", "engagement"]
    out = {f"err_{n}": float(v) for n, v in zip(names, e)}
    out.update(err_mean=float(e.mean()), prd=float(np.mean(prds)), bytes_per_frame=float(np.mean(nbytes)), cr=float(frames.shape[1] * frames.shape[2] * 2 / np.mean(nbytes)))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--minutes", type=float, default=48.0, help="minutes of synthetic EEG")
    ap.add_argument("--patient", default="chb02", help="CHB-MIT patient for real frames (not used in evaluation)")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--base", default=str(BASE_MODEL))
    ap.add_argument("--out", default=str(STREAM_MODEL))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    model = load_model(Path(args.base))
    syn = synthetic_frames(args.minutes)
    real = None if args.no_real else real_frames(args.patient)
    frames = syn if real is None else np.concatenate([syn, real])
    print(f"fine-tuning on {len(syn)} synthetic + {0 if real is None else len(real)} real 2 s frames, {args.epochs} epochs")
    test_syn = synthetic_frames(12.0, seed=999)
    before = feature_error(load_model(Path(args.base)), test_syn)
    model, hist = finetune(model, frames, epochs=args.epochs)
    after = feature_error(model, test_syn)
    save_model(model, Path(args.out), extra={"finetuned_from": Path(args.base).name, "task_aware": True, "history": hist, "feature_error_before": before, "feature_error_after": after})
    print(f"saved {args.out}")
    print(f"feature error on held-out synthetic frames: {before['err_mean']:.3f} -> {after['err_mean']:.3f} (beta {before['err_beta']:.3f} -> {after['err_beta']:.3f}); PRD {before['prd']:.1f} -> {after['prd']:.1f} %")


if __name__ == "__main__":
    main()
