import numpy as np
import pytest
import torch

from compression.autoencoder import ConvAutoencoder, count_macs, frequency_loss, ste_quantize
from compression.codecs import AutoencoderCodec, CodecRegistry, RawCodec, WaveletCodec, prd
from compression.quantization import (
    FLAG_IMPORTANT,
    SegmentHeader,
    build_segment,
    pack_bits,
    parse_segment,
    unpack_bits,
)
from config import AutoencoderConfig


@pytest.mark.parametrize("bits", [1, 3, 4, 6, 8, 12, 16])
def test_bit_packing_roundtrip(bits, rng):
    v = rng.integers(0, 1 << bits, size=1001).astype(np.uint32)
    packed = pack_bits(v, bits)
    assert len(packed) == (1001 * bits + 7) // 8
    assert np.array_equal(unpack_bits(packed, bits, 1001), v)


def test_segment_container_roundtrip():
    h = SegmentHeader(1, 2, 3000, np.array([1.5, -2.0]), np.array([20.0, 35.0]), FLAG_IMPORTANT, quality=40)
    blob = build_segment(h, b"\x00" * 500)
    h2, payload = parse_segment(blob)
    assert payload == b"\x00" * 500
    assert h2.codec_id == 1 and h2.n_samples == 3000 and h2.flags == FLAG_IMPORTANT
    assert h2.prd_estimate == 20.0
    assert len(blob) < 500  # zlib applied because it helped


def test_raw_codec_is_near_lossless(segment):
    c = RawCodec()
    enc = c.encode(segment)
    assert enc.nbytes >= segment.size * 2
    assert prd(segment, c.decode(enc.blob)) < 0.5


def test_wavelet_codec_rate_distortion(segment):
    sizes, errs = [], []
    for keep in (0.05, 0.2, 0.5):
        c = WaveletCodec(keep, 8)
        enc = c.encode(segment)
        sizes.append(enc.nbytes)
        errs.append(prd(segment, c.decode(enc.blob)))
    assert sizes == sorted(sizes)  # keeping more costs more bytes...
    assert errs == sorted(errs, reverse=True)  # ...and lowers distortion
    assert sizes[0] < segment.size * 2 / 10  # >10x compression at 5 %


def test_verify_writes_measured_prd(segment):
    c = WaveletCodec(0.1, 8)
    enc = c.encode(segment, verify=True)
    measured = prd(segment, c.decode(enc.blob))
    header, _ = parse_segment(enc.blob)
    assert abs(header.prd_estimate - measured) <= 0.5


def test_autoencoder_shapes_and_compression(segment):
    model = ConvAutoencoder(2, AutoencoderConfig())
    x = torch.randn(3, 2, 3000)
    z = model.encode(x)
    assert z.shape == (3, 16, 188) and z.abs().max() <= 1
    assert model(x, bits=4).shape == x.shape
    codec = AutoencoderCodec(model, bits=4, keep=8)
    enc = codec.encode(segment)
    assert enc.nbytes < 0.1 * segment.size * 2  # real bytes, > 10x smaller than raw
    y = codec.decode(enc.blob)
    assert y.shape == segment.shape and np.all(np.isfinite(y))
    # truncating the latent sends fewer bytes
    assert AutoencoderCodec(model, 4, 2).encode(segment).nbytes < enc.nbytes


def test_ste_quantize_levels_and_gradient():
    z = torch.linspace(-1, 1, 101, requires_grad=True)
    q = ste_quantize(z, 3)
    assert len(torch.unique(q.detach())) == 8
    q.sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))


def test_frequency_loss_zero_for_identical_and_positive_for_lowpassed():
    x = torch.randn(2, 2, 3000)
    assert frequency_loss(x, x, 100.0).item() == pytest.approx(0.0, abs=1e-6)
    smooth = torch.nn.functional.avg_pool1d(x, 9, stride=1, padding=4)
    assert frequency_loss(x, smooth, 100.0).item() > 0.1


def test_encoder_is_much_cheaper_than_decoder():
    model = ConvAutoencoder(2, AutoencoderConfig())
    enc, dec = count_macs(model, 2, 3000, "encoder"), count_macs(model, 2, 3000, "decoder")
    assert enc < 10e6 and dec > 5 * enc


def test_registry_dispatches_by_header(segment):
    model = ConvAutoencoder(2, AutoencoderConfig())
    reg = CodecRegistry(model)
    for codec in (RawCodec(), WaveletCodec(), AutoencoderCodec(model, 6)):
        y, header = reg.decode(codec.encode(segment).blob)
        assert header.codec_id == codec.codec_id and y.shape == segment.shape
