"""Tests for the trace operation."""

from __future__ import annotations


def test_trace_gpt2_returns_results(gpt2_model):
    results = gpt2_model.trace(
        "The Eiffel Tower is in Paris",
        "The Eiffel Tower is in Rome",
        top_k=5,
    )
    assert isinstance(results, list)
    assert len(results) > 0


def test_trace_gpt2_results_sorted(gpt2_model):
    results = gpt2_model.trace(
        "The capital of France is",
        "The capital of Germany is",
        top_k=5,
    )
    effects = [r["effect"] for r in results]
    assert effects == sorted(effects, reverse=True)


def test_trace_gpt2_results_have_fields(gpt2_model):
    results = gpt2_model.trace(
        "hello world",
        "goodbye world",
        top_k=3,
    )
    for r in results:
        assert "module" in r
        assert "effect" in r
        assert isinstance(r["effect"], float)
