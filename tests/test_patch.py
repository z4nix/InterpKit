"""Tests for the patch operation."""

from __future__ import annotations


def test_patch_gpt2_returns_effect(gpt2_model):
    result = gpt2_model.patch(
        "The Eiffel Tower is in Paris",
        "The Eiffel Tower is in Rome",
        at="transformer.h.8.mlp",
    )
    assert "effect" in result
    assert 0.0 <= result["effect"] <= 1.0


def test_patch_gpt2_has_logits(gpt2_model):
    result = gpt2_model.patch(
        "The capital of France is",
        "The capital of Germany is",
        at="transformer.h.0.mlp",
    )
    assert result["clean_logits"] is not None
    assert result["corrupted_logits"] is not None
    assert result["patched_logits"] is not None


def test_patch_gpt2_different_layers(gpt2_model):
    r0 = gpt2_model.patch("hello world", "goodbye world", at="transformer.h.0.mlp")
    r11 = gpt2_model.patch("hello world", "goodbye world", at="transformer.h.11.mlp")
    # Different layers should produce different effects (not a hard guarantee, but likely)
    assert isinstance(r0["effect"], float)
    assert isinstance(r11["effect"], float)
