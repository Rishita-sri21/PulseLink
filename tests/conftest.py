import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AutoencoderConfig, PreprocessConfig, PulseLinkConfig  # noqa: E402
from preprocessing.load_eeg import preprocess_recording  # noqa: E402
from preprocessing.synthetic import generate_synthetic_recording  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    cfg = PreprocessConfig()
    return preprocess_recording(generate_synthetic_recording(40, cfg, seed=1), cfg)


@pytest.fixture(scope="session")
def segment(dataset):
    return dataset["segments"][3]


@pytest.fixture(scope="session")
def tiny_cfg():
    cfg = PulseLinkConfig()
    cfg.ae = AutoencoderConfig(epochs=2, batch_size=16)
    return cfg


@pytest.fixture(scope="session")
def prep(tiny_cfg, tmp_path_factory):
    from pipeline import prepare

    path = tmp_path_factory.mktemp("models") / "ae.pt"
    return prepare(tiny_cfg, synthetic=True, subjects=(0, 1, 2), n_synthetic_epochs=40, model_path=path)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
