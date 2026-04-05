"""Shared fixtures for interpkit tests."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch


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
def test_image_path():
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = os.path.join(tempfile.gettempdir(), "interpkit_test_img.jpg")
    img.save(path)
    return path
