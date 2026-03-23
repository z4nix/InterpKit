"""Shared fixtures for interpkit tests."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch


_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _load_or_skip(name: str, **kwargs):
    """Load a model or skip the test if loading fails."""
    try:
        import interpkit

        return interpkit.load(name, **kwargs)
    except Exception as e:
        pytest.skip(f"Cannot load {name}: {e}")


@pytest.fixture(scope="session")
def gpt2_model():
    """Load GPT-2 small once for the whole test session."""
    import interpkit

    return interpkit.load("gpt2", device="cpu")


@pytest.fixture(scope="session")
def resnet_model():
    """Load a small HF ResNet once for the whole test session."""
    import interpkit

    return interpkit.load("microsoft/resnet-18", device="cpu")


@pytest.fixture(scope="session")
def distilbert_model():
    return _load_or_skip("distilbert-base-uncased", device=_DEVICE)


@pytest.fixture(scope="session")
def t5_model():
    return _load_or_skip("t5-small", device=_DEVICE)


@pytest.fixture(scope="session")
def vit_model():
    return _load_or_skip("google/vit-base-patch16-224", device=_DEVICE)


@pytest.fixture(scope="session")
def recurrentgemma_model():
    return _load_or_skip(
        "google/recurrentgemma-2b-it", device=_DEVICE, dtype="float16",
    )


@pytest.fixture(scope="session")
def qwen2_model():
    return _load_or_skip("Qwen/Qwen2-0.5B", device=_DEVICE, dtype="float16")


@pytest.fixture(scope="session")
def bloom_model():
    return _load_or_skip("bigscience/bloom-560m", device=_DEVICE)


@pytest.fixture(scope="session")
def mamba_model():
    return _load_or_skip("state-spaces/mamba-130m", device=_DEVICE)


@pytest.fixture(scope="session")
def roberta_qa_model():
    return _load_or_skip("deepset/roberta-base-squad2", device=_DEVICE)


@pytest.fixture(scope="session")
def opt_model():
    return _load_or_skip("facebook/opt-350m", device=_DEVICE)


@pytest.fixture(scope="session")
def albert_model():
    return _load_or_skip("albert-base-v2", device=_DEVICE)


@pytest.fixture(scope="session")
def flan_t5_model():
    return _load_or_skip("google/flan-t5-small", device=_DEVICE)


@pytest.fixture(scope="session")
def pythia_model():
    return _load_or_skip("EleutherAI/pythia-160m", device=_DEVICE)


@pytest.fixture(scope="session")
def bart_model():
    return _load_or_skip("facebook/bart-base", device=_DEVICE)


@pytest.fixture(scope="session")
def gpt_neo_model():
    return _load_or_skip("EleutherAI/gpt-neo-125m", device=_DEVICE)


@pytest.fixture(scope="session")
def electra_model():
    return _load_or_skip("google/electra-small-discriminator", device=_DEVICE)


@pytest.fixture(scope="session")
def smollm_model():
    return _load_or_skip("HuggingFaceTB/SmolLM-135M", device=_DEVICE)


@pytest.fixture(scope="session")
def test_image_path():
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = os.path.join(tempfile.gettempdir(), "interpkit_test_img.jpg")
    img.save(path)
    return path
