"""Tests for the logit lens operation."""

from __future__ import annotations


def test_lens_gpt2_returns_predictions(gpt2_model):
    results = gpt2_model.lens("The capital of France is")
    assert isinstance(results, list)
    assert len(results) > 0


def test_lens_gpt2_predictions_have_fields(gpt2_model):
    results = gpt2_model.lens("The capital of France is")
    for pred in results:
        assert "layer_name" in pred
        assert "top1_token" in pred
        assert "top1_prob" in pred
        assert "top5_tokens" in pred
        assert len(pred["top5_tokens"]) == 5


def test_lens_resnet_with_image(resnet_model):
    """Vision lens (Phase 3): lens on a CNN projects through the classifier head.

    Pre-1.0 returned None (silently unsupported). Now CNNs are first-class —
    lens returns per-layer class-probability evolution. Inputs must be
    images, not text.
    """
    import os
    import tempfile

    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = os.path.join(tempfile.gettempdir(), "test_lens_resnet.jpg")
    img.save(path)

    result = resnet_model.lens(path)
    assert result is not None
    assert len(result) > 0
    assert all("top1_token" in entry for entry in result)


def test_lens_resnet_text_input_raises(resnet_model):
    """Passing text to a vision model's lens raises WrongInputType (A2):
    the op is supported for the family, but a text string is the wrong input
    type. (Pre-A2 this surfaced as a generic ValueError from the tokenizer.)"""
    import pytest

    from interpkit import WrongInputType

    with pytest.raises(WrongInputType, match="vision model"):
        resnet_model.lens("hello")
