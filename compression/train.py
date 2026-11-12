"""Training loop for the PulseLink autoencoder (CPU-friendly, Colab-ready)."""
from __future__ import annotations

import logging
import time

import numpy as np
import torch

from compression.autoencoder import TRAIN_BITS, ConvAutoencoder, pulselink_loss, train_widths
from config import AutoencoderConfig
from preprocessing.segmentation import normalize_segments

log = logging.getLogger(__name__)


def train_autoencoder(
    segments: np.ndarray,
    fs: float,
    cfg: AutoencoderConfig | None = None,
    val_fraction: float = 0.1,
    device: str | None = None,
    progress=None,
) -> tuple[ConvAutoencoder, list[dict]]:
    """Train on (S, C, T) segments in microvolts. Returns (model, history).

    ``progress`` is an optional callback ``f(epoch, n_epochs, record)``.
    """
    cfg = cfg or AutoencoderConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    xn, _, _ = normalize_segments(segments)
    xn = np.clip(xn, -8, 8).astype(np.float32)  # tame artifact spikes
    idx = rng.permutation(len(xn))
    n_val = max(1, int(len(xn) * val_fraction)) if len(xn) > 1 else 0
    val, tr = xn[idx[:n_val]], xn[idx[n_val:]]
    if len(tr) == 0:
        tr = val

    model = ConvAutoencoder(segments.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    tr_t = torch.as_tensor(tr)
    val_t = torch.as_tensor(val) if len(val) else None
    history = []
    widths = train_widths(cfg.latent_channels)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(tr_t))
        tot = lt = lf = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            xb = tr_t[perm[i : i + cfg.batch_size]].to(device)
            if rng.random() < 0.5:  # amplitude-sign augmentation
                xb = -xb
            bits = int(rng.choice(TRAIN_BITS))
            # Full width half the time so the top rate mode is well trained.
            keep = cfg.latent_channels if rng.random() < 0.4 else int(rng.choice(widths))
            x_hat = model(xb, bits, keep)
            loss, l_time, l_freq = pulselink_loss(xb, x_hat, fs, cfg.alpha, cfg.beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = len(xb)
            tot += loss.item() * n
            lt += l_time.item() * n
            lf += l_freq.item() * n
        sched.step()
        rec = {"epoch": epoch + 1, "loss": tot / len(tr_t), "time_mse": lt / len(tr_t), "freq_l1": lf / len(tr_t)}
        if val_t is not None:
            model.eval()
            with torch.no_grad():
                xv = val_t.to(device)
                vh = model(xv, 6)
                vloss, vt, vf = pulselink_loss(xv, vh, fs, cfg.alpha, cfg.beta)
            rec.update(val_loss=vloss.item(), val_time_mse=vt.item(), val_freq_l1=vf.item())
        rec["seconds"] = time.time() - t0
        history.append(rec)
        log.info("epoch %d/%d %s", epoch + 1, cfg.epochs, {k: round(v, 4) for k, v in rec.items()})
        if progress:
            progress(epoch + 1, cfg.epochs, rec)

    model.to("cpu").eval()
    return model, history
