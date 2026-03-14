"""Tests for the diff operation."""

from __future__ import annotations

import interpkit


def test_diff_same_model(gpt2_model):
    """Diffing a model with itself should show near-zero distances."""
    output = interpkit.diff(gpt2_model, gpt2_model, "hello world")
    assert isinstance(output, dict)
    assert "results" in output
    assert "skipped_a" in output
    assert "skipped_b" in output
    results = output["results"]
    assert len(results) > 0
    for r in results:
        assert "module" in r
        assert "distance" in r
        assert r["distance"] < 0.01, f"Self-diff should be ~0 but got {r['distance']} at {r['module']}"


def test_diff_returns_sorted(gpt2_model):
    output = interpkit.diff(gpt2_model, gpt2_model, "The capital of France")
    results = output["results"]
    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i]["distance"] >= results[i + 1]["distance"]
