"""Central configuration for PulseLink.

Every tunable number in the pipeline lives here so experiments are easy to
reproduce and so the README can point to a single place for assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# EEG frequency bands (Hz) used for band-power metrics, importance detection
# and the downstream sleep-staging features.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}

# Sleep-EDF hypnogram annotation -> integer label. Stage 3 and 4 are merged
# into N3 following AASM.
SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]
ANNOTATION_TO_STAGE = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


@dataclass
class PreprocessConfig:
    channels: tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz")
    fs: float = 100.0  # Sleep-EDF EEG sampling rate
    l_freq: float = 0.5
    h_freq: float = 40.0
    filter_order: int = 4
    segment_seconds: float = 30.0  # one Sleep-EDF scoring epoch
    # Raw EDF samples are 16-bit integers: the "uncompressed" reference size.
    raw_bits_per_sample: int = 16

    @property
    def segment_samples(self) -> int:
        return int(round(self.segment_seconds * self.fs))


@dataclass
class AutoencoderConfig:
    # Asymmetric design: the encoder runs on the wearable, so it is kept small;
    # the decoder runs on the receiver (phone/PC) and can afford more capacity.
    enc_hidden: tuple[int, ...] = (16, 32, 32)
    dec_hidden: tuple[int, ...] = (64, 64, 32)
    latent_channels: int = 16  # transmitter may send only the first k of these
    kernel_size: int = 5
    # Each stage halves time, plus one stride-2 latent projection:
    # total temporal downsampling = 2 ** (len(enc_hidden) + 1) = 16.
    alpha: float = 1.0  # time-domain loss weight
    beta: float = 0.5  # frequency-domain loss weight
    epochs: int = 30
    batch_size: int = 8  # small batches = more updates per epoch on small data
    lr: float = 3e-3
    seed: int = 0


@dataclass
class ChannelConfig:
    bandwidth_bps: float = 64_000.0  # application throughput, bits/s
    mtu_bytes: int = 244  # BLE 5 max ATT payload; packet payload size
    header_bytes: int = 8  # per-packet header (seq, segment, fragment, crc)
    loss_rate: float = 0.0  # mean packet loss probability
    burst_length: float = 1.0  # mean loss burst length (1 = independent loss)
    base_latency_s: float = 0.02
    jitter_s: float = 0.01
    max_retries: int = 0  # ARQ retransmissions per packet (0 = none)
    # Outages: list of (start_s, duration_s) in stream time.
    outages: list[tuple[float, float]] = field(default_factory=list)
    buffer_bytes: int = 256_000  # transmitter buffer used during outages
    seed: int = 0


@dataclass
class EnergyConfig:
    """Transparent radio + compute energy model.

    Default numbers are order-of-magnitude figures for a BLE wearable with a
    Cortex-M4 class MCU. They are assumptions, not measurements: replace them
    with datasheet or measured values for your hardware.
    """

    tx_energy_per_bit_j: float = 50e-9  # radio energy per transmitted bit
    tx_energy_per_packet_j: float = 5e-6  # fixed cost per packet (wake-up, preamble)
    rx_energy_per_bit_j: float = 40e-9  # receiver radio energy per bit
    mcu_energy_per_op_j: float = 0.1e-9  # transmitter MCU energy per (int8 SIMD) MAC/op
    receiver_energy_per_op_j: float = 0.1e-9  # receiver (phone/PC) energy per op


@dataclass
class QualityConfig:
    max_prd_percent: float = 30.0  # flag segments above this distortion
    max_band_error: float = 0.5  # relative band-power error threshold


@dataclass
class PulseLinkConfig:
    pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    ae: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
