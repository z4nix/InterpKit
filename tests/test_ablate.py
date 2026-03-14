"""Tests for the ablate operation."""

from __future__ import annotations


def test_ablate_zero(gpt2_model):
    result = gpt2_model.ablate("The capital of France is", at="transformer.h.8.mlp")
    assert "effect" in result
    assert 0.0 <= result["effect"] <= 1.0
    assert result["method"] == "zero"


def test_ablate_mean(gpt2_model):
    result = gpt2_model.ablate("The capital of France is", at="transformer.h.8.mlp", method="mean")
    assert "effect" in result
    assert result["method"] == "mean"


def test_ablate_different_layers(gpt2_model):
    r0 = gpt2_model.ablate("hello world", at="transformer.h.0.mlp")
    r11 = gpt2_model.ablate("hello world", at="transformer.h.11.mlp")
    assert isinstance(r0["effect"], float)
    assert isinstance(r11["effect"], float)
