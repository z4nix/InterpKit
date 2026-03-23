"""Tests for the activations operation."""

from __future__ import annotations

import torch


def test_activations_single_module(gpt2_model):
    act = gpt2_model.activations("The capital of France is", at="transformer.h.8.mlp")
    assert isinstance(act, torch.Tensor)
    assert act.dim() >= 2


def test_activations_multiple_modules(gpt2_model):
    acts = gpt2_model.activations(
        "The capital of France is",
        at=["transformer.h.0.mlp", "transformer.h.8.mlp"],
    )
    assert isinstance(acts, dict)
    assert "transformer.h.0.mlp" in acts
    assert "transformer.h.8.mlp" in acts
    assert isinstance(acts["transformer.h.0.mlp"], torch.Tensor)


def test_activations_resnet(resnet_model):
    from PIL import Image
    import tempfile, os

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = os.path.join(tempfile.gettempdir(), "test_act.jpg")
    img.save(path)

    act = resnet_model.activations(path, at="resnet.encoder.stages.0")
    assert isinstance(act, torch.Tensor)
