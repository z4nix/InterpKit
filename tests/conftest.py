"""Shared fixtures for interpkit tests."""

from __future__ import annotations

import os
import tempfile

import pytest


def load_or_skip(model_id: str, **kwargs):
    """Load a HuggingFace model or skip the test if it cannot be downloaded.

    Handles network errors, missing models, and import failures gracefully.
    """
    import interpkit

    try:
        return interpkit.load(model_id, device="cpu", **kwargs)
    except Exception as exc:
        pytest.skip(f"Could not load {model_id}: {exc}")


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False,
        help="Run slow multi-architecture tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (multi-arch download)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def gpt2_model():
    """Load GPT-2 small once for the whole test session."""
    return load_or_skip("gpt2")


@pytest.fixture(scope="session")
def resnet_model():
    """Load a small HF ResNet once for the whole test session."""
    return load_or_skip("microsoft/resnet-18")


@pytest.fixture(scope="session")
def distilbert_model():
    """Load DistilBERT once (encoder-only)."""
    return load_or_skip("distilbert-base-uncased")


@pytest.fixture(scope="session")
def t5_model():
    """Load T5-small once (encoder-decoder)."""
    return load_or_skip("t5-small")


@pytest.fixture(scope="session")
def pythia_model():
    """Load Pythia-160m once (GQA-style decoder)."""
    return load_or_skip("EleutherAI/pythia-160m")


@pytest.fixture(scope="session")
def vit_model():
    """Load ViT once (vision transformer)."""
    return load_or_skip("google/vit-base-patch16-224")


@pytest.fixture(scope="session")
def test_image_path():
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = os.path.join(tempfile.gettempdir(), "interpkit_test_img.jpg")
    img.save(path)
    return path
