import numpy as np

from communication.channel_simulator import ChannelSimulator, packetize
from communication.packet_loss import GilbertElliottLoss, OutageSchedule
from compression.codecs import CodecRegistry, RawCodec, WaveletCodec
from config import ChannelConfig
from reconstruction.decoder import Receiver


def test_packetize_fragments_and_reassembles():
    blob = bytes(range(256)) * 3
    pk = packetize(blob, 7, 100, 0.0, 0)
    assert len(pk) == 8 and all(p.n_frags == 8 for p in pk)
    assert b"".join(p.payload for p in pk) == blob


def test_lossless_channel_delivers_everything_in_order():
    sim = ChannelSimulator(ChannelConfig(bandwidth_bps=10_000))
    res = sim.send_segment(b"x" * 1000, 0, 0.0)
    assert all(r.delivered for r in res)
    arrivals = [r.t_arrival for r in res]
    assert arrivals == sorted(arrivals)
    # 5 packets of <=244 B + 8 B header at 10 kbit/s
    assert arrivals[-1] >= (1000 + 5 * 8) * 8 / 10_000


def test_loss_rate_matches_configuration():
    sim = ChannelSimulator(ChannelConfig(loss_rate=0.1, seed=3, bandwidth_bps=1e7))
    for i in range(200):
        sim.send_segment(b"y" * 2440, i, i * 1.0)
    assert 0.07 < sim.stats.lost / sim.stats.packets < 0.13


def test_gilbert_elliott_bursty_but_same_mean():
    rng = np.random.default_rng(0)
    m = GilbertElliottLoss(0.1, 5.0, rng)
    seq = np.array([m.lost() for _ in range(50_000)])
    assert 0.08 < seq.mean() < 0.12
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, seq.astype(int), 0])))[::2]
    assert runs.mean() > 3  # bursts, not isolated losses


def test_arq_recovers_losses_at_extra_cost():
    base = dict(loss_rate=0.2, seed=1, bandwidth_bps=1e7)
    no_arq = ChannelSimulator(ChannelConfig(**base))
    arq = ChannelSimulator(ChannelConfig(**base, max_retries=4))
    for i in range(50):
        no_arq.send_segment(b"z" * 2440, i, float(i))
        arq.send_segment(b"z" * 2440, i, float(i))
    assert arq.stats.lost < no_arq.stats.lost / 10
    assert arq.stats.bytes_on_air > no_arq.stats.bytes_on_air


def test_outage_buffers_then_flushes():
    sim = ChannelSimulator(ChannelConfig(outages=[(5.0, 20.0)], bandwidth_bps=1e5))
    res = sim.send_segment(b"a" * 500, 0, 10.0)  # created during the outage
    assert all(r.delivered for r in res)
    assert min(r.t_start for r in res) >= 25.0
    assert sim.stats.buffered_by_outage == len(res)
    assert OutageSchedule([(5, 20)]).next_up(10) == 25


def test_small_buffer_overflows_during_outage():
    sim = ChannelSimulator(ChannelConfig(outages=[(0.0, 100.0)], buffer_bytes=3000, bandwidth_bps=1e5))
    for i in range(10):
        sim.send_segment(b"b" * 1000, i, float(i))
    assert sim.stats.overflow > 0


def test_receiver_conceals_lost_and_partially_recovers_raw(segment):
    reg = CodecRegistry()
    sim = ChannelSimulator(ChannelConfig(loss_rate=0.3, seed=5, bandwidth_bps=1e7))
    raw_blob = RawCodec().encode(segment).blob
    dwt_blob = WaveletCodec(0.3, 8).encode(segment).blob
    rx = Receiver(reg, *segment.shape)
    rx.receive(sim.send_segment(raw_blob, 0, 30.0))
    rx.receive(sim.send_segment(dwt_blob, 1, 60.0))
    out = rx.reconstruct([0, 1])
    assert out[0].status in ("partial", "ok", "lost")
    if out[0].status == "partial":
        assert 0 < out[0].missing_fraction < 1
        err = np.sqrt(np.mean((out[0].signal - segment) ** 2)) / segment.std()
        assert err < 1.0  # better than sending nothing
    assert out[1].status in ("ok", "lost")
    if out[1].status == "lost":
        assert np.all(out[1].signal == 0)
