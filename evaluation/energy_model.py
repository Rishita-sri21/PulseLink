"""Transparent energy model for the wearable transmitter (and receiver).

    E_net = E_transmission + E_compression + E_reconstruction

* ``E_transmission`` = bits_on_air * E_bit + attempts * E_packet  (every
  retransmission and every header byte is paid for);
* ``E_compression`` = encoder ops * E_op(MCU)  (normalisation, DWT or
  autoencoder MACs, quantisation, zlib, and the closed-loop decode when
  enabled);
* ``E_reconstruction`` = decoder ops * E_op(receiver) + received bits *
  E_rx_bit. It is reported separately because the receiver (phone/PC) is
  usually not battery-critical.

The main comparison is the *transmitter* energy of each method against raw
transmission under the same channel and hardware assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from config import EnergyConfig


@dataclass
class EnergyBreakdown:
    transmission_j: float
    compression_j: float
    reconstruction_j: float
    receiver_radio_j: float

    @property
    def transmitter_j(self) -> float:
        return self.transmission_j + self.compression_j

    @property
    def receiver_j(self) -> float:
        return self.reconstruction_j + self.receiver_radio_j

    @property
    def net_j(self) -> float:
        """E_net = E_transmission + E_compression + E_reconstruction."""
        return self.transmission_j + self.compression_j + self.reconstruction_j

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(transmitter_j=self.transmitter_j, receiver_j=self.receiver_j, net_j=self.net_j)
        return d


def energy(
    bytes_on_air: float,
    attempts: float,
    encode_ops: float,
    decode_ops: float,
    bytes_received: float | None = None,
    cfg: EnergyConfig | None = None,
) -> EnergyBreakdown:
    cfg = cfg or EnergyConfig()
    bytes_received = bytes_on_air if bytes_received is None else bytes_received
    return EnergyBreakdown(
        transmission_j=bytes_on_air * 8 * cfg.tx_energy_per_bit_j + attempts * cfg.tx_energy_per_packet_j,
        compression_j=encode_ops * cfg.mcu_energy_per_op_j,
        reconstruction_j=decode_ops * cfg.receiver_energy_per_op_j,
        receiver_radio_j=bytes_received * 8 * cfg.rx_energy_per_bit_j,
    )


def segment_energy(nbytes: int, encode_ops: int, decode_ops: int, mtu: int = 244, header_bytes: int = 8, cfg: EnergyConfig | None = None) -> EnergyBreakdown:
    """Ideal-channel energy for one encoded segment (no loss, no retries)."""
    n_packets = max(1, -(-nbytes // mtu))
    on_air = nbytes + n_packets * header_bytes
    return energy(on_air, n_packets, encode_ops, decode_ops, cfg=cfg)


def savings_percent(method_j: float, raw_j: float) -> float:
    return float(100.0 * (1.0 - method_j / raw_j)) if raw_j else 0.0
