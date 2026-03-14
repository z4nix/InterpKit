"""Shared fixtures for mechkit tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def gpt2_model():
    """Load GPT-2 small once for the whole test session."""
    import mechkit

    return mechkit.load("gpt2", device="cpu")


@pytest.fixture(scope="session")
def resnet_model():
    """Load a small HF ResNet once for the whole test session."""
    import mechkit

    return mechkit.load("microsoft/resnet-18", device="cpu")
