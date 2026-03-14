"""Tests for the diff operation."""

from __future__ import annotations

import mechkit


def test_diff_same_model(gpt2_model):
    """Diffing a model with itself should show near-zero distances."""
    results = mechkit.diff(gpt2_model, gpt2_model, "hello world")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "module" in r
        assert "distance" in r
        assert r["distance"] < 0.01, f"Self-diff should be ~0 but got {r['distance']} at {r['module']}"


def test_diff_returns_sorted(gpt2_model):
    results = mechkit.diff(gpt2_model, gpt2_model, "The capital of France")
    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i]["distance"] >= results[i + 1]["distance"]
