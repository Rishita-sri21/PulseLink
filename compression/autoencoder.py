"""1D convolutional autoencoder with a frequency-aware loss.

Shapes with the default config (2 channels, 30 s at 100 Hz)::

    X  : (B, 2, 3000) -> padded to (B, 2, 3008)
    Z  : (B, 8, 188)  values in [-1, 1] (tanh), quantised to b bits
    X^ : (B, 2, 3000)

Raw size  = 2 * 3000 * 16 bit = 96 000 bit
Latent    = 8 * 188 * b bit   = 6 016 bit at b = 4  (16x before zlib)

The encoder is deliberately light (~4 M MACs per 30 s segment) because it
runs on the wearable; the decoder is larger because it runs on the receiver.

The latent is bounded by ``tanh`` so the quantiser range is fixed ([-1, 1])
and needs no side information. Training uses straight-through rounding at a
randomly chosen bit depth so one model serves every adaptive rate mode.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AutoencoderConfig

TRAIN_BITS = (3, 4, 5, 6, 8)
# Nested-dropout widths: during training only the first k latent channels are
# kept, so channels are ordered by importance and the transmitter can send a
# prefix of them. This gives one model a whole ladder of bit rates.
def train_widths(latent_channels: int) -> tuple[int, ...]:
    return tuple(sorted({max(1, latent_channels // 8), max(1, latent_channels // 4), latent_channels // 2, (3 * latent_channels) // 4, latent_channels}))


class ConvAutoencoder(nn.Module):
    def __init__(self, n_channels: int = 2, cfg: AutoencoderConfig | None = None):
        super().__init__()
        cfg = cfg or AutoencoderConfig()
        self.cfg = cfg
        self.n_channels = n_channels
        k, pad = cfg.kernel_size, cfg.kernel_size // 2
        if len(cfg.enc_hidden) != len(cfg.dec_hidden):
            raise ValueError("encoder and decoder need the same number of stages")
        self.factor = 2 ** (len(cfg.enc_hidden) + 1)

        # Encoder (wearable side): one strided conv per stage, no extra layers.
        enc, prev = [], n_channels
        for h in cfg.enc_hidden:
            enc += [nn.Conv1d(prev, h, k, stride=2, padding=pad), nn.GELU()]
            prev = h
        enc += [nn.Conv1d(prev, cfg.latent_channels, k, stride=2, padding=pad)]
        self.encoder = nn.Sequential(*enc)

        # Decoder (receiver side): upsample + two convs per stage.
        dec, prev = [], cfg.latent_channels
        for h in cfg.dec_hidden:
            dec += [nn.Upsample(scale_factor=2, mode="linear", align_corners=False), nn.Conv1d(prev, h, k, padding=pad), nn.GELU(), nn.Conv1d(h, h, k, padding=pad), nn.GELU()]
            prev = h
        dec += [nn.Upsample(scale_factor=2, mode="linear", align_corners=False), nn.Conv1d(prev, n_channels, k, padding=pad)]
        self.decoder = nn.Sequential(*dec)

    # -- shape helpers ------------------------------------------------------ #
    def padded_length(self, t: int) -> int:
        return int(math.ceil(t / self.factor) * self.factor)

    def latent_length(self, t: int) -> int:
        return self.padded_length(t) // self.factor

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        extra = self.padded_length(x.shape[-1]) - x.shape[-1]
        return F.pad(x, (0, extra), mode="reflect") if extra else x

    # -- forward ------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(self._pad(x)))

    def decode(self, z: torch.Tensor, t: int) -> torch.Tensor:
        return self.decoder(z)[..., :t]

    def forward(self, x: torch.Tensor, bits: int | None = None, keep: int | None = None) -> torch.Tensor:
        z = self.encode(x)
        if bits is not None:
            z = ste_quantize(z, bits)
        if keep is not None and keep < z.shape[1]:
            z = truncate_latent(z, keep)
        return self.decode(z, x.shape[-1])


def truncate_latent(z: torch.Tensor, keep: int) -> torch.Tensor:
    """Zero every latent channel after the first ``keep`` (not transmitted)."""
    mask = torch.zeros(1, z.shape[1], 1, dtype=z.dtype, device=z.device)
    mask[:, :keep] = 1
    return z * mask


def ste_quantize(z: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantise [-1, 1] values to ``bits`` with a straight-through gradient."""
    levels = (1 << bits) - 1
    q = torch.round((z + 1) / 2 * levels) / levels * 2 - 1
    return z + (q - z).detach()


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def psd_torch(x: torch.Tensor) -> torch.Tensor:
    """Periodogram along the last axis (Hann window)."""
    win = torch.hann_window(x.shape[-1], device=x.device, dtype=x.dtype)
    spec = torch.fft.rfft(x * win, dim=-1)
    return spec.real**2 + spec.imag**2


def smoothed_psd(x: torch.Tensor, fs: float, band=(0.5, 40.0), resolution_hz: float = 0.5):
    """Periodogram averaged into ``resolution_hz`` bins inside ``band``.

    A raw periodogram of a noise-like signal fluctuates wildly bin to bin, so
    two signals with identical spectra still have very different
    periodograms. Averaging neighbouring bins gives a stable PSD estimate.
    """
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    p = psd_torch(x)[..., sel] / x.shape[-1]
    width = max(1, int(round(resolution_hz / (fs / x.shape[-1]))))
    n = (p.shape[-1] // width) * width
    return p[..., :n].reshape(*p.shape[:-1], -1, width).mean(-1)


def frequency_loss(x: torch.Tensor, x_hat: torch.Tensor, fs: float, band=(0.5, 40.0)) -> torch.Tensor:
    """L1 distance between log-PSDs inside the EEG band.

    The roadmap's ``||PSD(X) - PSD(X^)||_1`` is computed on log power so the
    low-power alpha/sigma/beta bands are not swamped by delta.
    """
    p, p_hat = smoothed_psd(x, fs, band), smoothed_psd(x_hat, fs, band)
    return (torch.log10(p + 1e-4) - torch.log10(p_hat + 1e-4)).abs().mean()


def pulselink_loss(x, x_hat, fs: float, alpha: float, beta: float):
    """L = alpha * ||X - X^||^2 + beta * ||log PSD(X) - log PSD(X^)||_1."""
    l_time = F.mse_loss(x_hat, x)
    l_freq = frequency_loss(x, x_hat, fs)
    return alpha * l_time + beta * l_freq, l_time.detach(), l_freq.detach()


# --------------------------------------------------------------------------- #
# Complexity
# --------------------------------------------------------------------------- #
def count_macs(module: nn.Module, n_channels: int, n_samples: int, part: str = "encoder") -> int:
    """Multiply-accumulate count for the encoder or decoder of one segment."""
    model = module
    macs = 0

    def hook(m, inp, out):
        nonlocal macs
        if isinstance(m, nn.Conv1d):
            macs += out.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]

    target = model.encoder if part == "encoder" else model.decoder
    handles = [m.register_forward_hook(hook) for m in target.modules() if isinstance(m, nn.Conv1d)]
    with torch.no_grad():
        x = torch.zeros(1, n_channels, n_samples)
        z = model.encode(x)
        if part == "decoder":
            macs = 0
            model.decode(z, n_samples)
    for h in handles:
        h.remove()
    # tanh / GELU / upsampling are cheap next to the convolutions; add ~1 op
    # per activation element as a margin.
    return int(macs * 1.05)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(model: ConvAutoencoder, path: Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_channels": model.n_channels, "cfg": model.cfg.__dict__, "extra": extra or {}}, path)


def load_model(path: Path) -> ConvAutoencoder:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["enc_hidden"] = tuple(cfg_dict["enc_hidden"])
    cfg_dict["dec_hidden"] = tuple(cfg_dict["dec_hidden"])
    model = ConvAutoencoder(ckpt["n_channels"], AutoencoderConfig(**cfg_dict))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.extra = ckpt.get("extra", {})
    return model


@torch.no_grad()
def reconstruct_batch(model: ConvAutoencoder, xn: np.ndarray, bits: int | None = None) -> np.ndarray:
    """Convenience: normalised (S, C, T) numpy in -> reconstruction out."""
    model.eval()
    x = torch.as_tensor(xn, dtype=torch.float32)
    return model(x, bits).numpy().astype(np.float64)
