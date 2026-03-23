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


def test_lens_resnet_returns_none(resnet_model):
    result = resnet_model.lens("test")
    # ResNet is not a language model — lens should gracefully skip
    assert result is None
