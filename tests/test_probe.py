"""Tests for the probe operation."""

from __future__ import annotations


def test_probe_basic(gpt2_model):
    texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "A bird flew over the house",
        "A fish swam in the sea",
        "The cat chased the mouse",
        "The dog ate the bone",
        "A bird sang in the tree",
        "A fish jumped out of water",
    ]
    labels = [0, 0, 1, 1, 0, 0, 1, 1]  # land vs air/water

    result = gpt2_model.probe(texts=texts, labels=labels, at="transformer.h.8")
    assert "accuracy" in result
    assert "top_features" in result
    assert isinstance(result["accuracy"], float)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_probe_returns_top_features(gpt2_model):
    texts = ["good", "great", "bad", "terrible", "excellent", "awful"]
    labels = [1, 1, 0, 0, 1, 0]

    result = gpt2_model.probe(texts=texts, labels=labels, at="transformer.h.0")
    assert len(result["top_features"]) > 0
    idx, weight = result["top_features"][0]
    assert isinstance(idx, int)
    assert isinstance(weight, float)
