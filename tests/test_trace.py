"""Tests for the trace operation.

The 1.0 trace API returns a dict with ``results`` and ``meta`` rather
than a bare list (F-015 fix). The dict shape exposes provenance per
candidate (``measurement_method``, ``atp_score``, ``atp_rank``) and a
top-level meta block describing the dispatch tier (auto/exhaustive/
approximate), candidate counts, and runtime.
"""

from __future__ import annotations


def test_trace_gpt2_returns_results(gpt2_model):
    out = gpt2_model.trace(
        "The Eiffel Tower is in Paris",
        "The Eiffel Tower is in Rome",
        top_k=5,
    )
    assert isinstance(out, dict)
    assert "results" in out and "meta" in out
    assert len(out["results"]) > 0
    assert out["meta"]["mode"] == "module"
    assert out["meta"]["algorithm"] in ("exhaustive", "approximate")


def test_trace_gpt2_results_sorted(gpt2_model):
    out = gpt2_model.trace(
        "The capital of France is",
        "The capital of Germany is",
        top_k=5,
    )
    # Results sorted by absolute effect, descending. NaN entries (degenerate
    # gaps for a given module) sort to the bottom.
    abs_effects = [abs(r["effect"]) for r in out["results"] if r["effect"] == r["effect"]]
    assert abs_effects == sorted(abs_effects, reverse=True)


def test_trace_gpt2_results_have_fields(gpt2_model):
    out = gpt2_model.trace(
        "hello world",
        "goodbye world",
        top_k=3,
    )
    for r in out["results"]:
        assert "module" in r
        assert "effect" in r
        # measurement_method is the new provenance field; full_patch for
        # exhaustive mode, atp_approximation possible for approximate-only.
        assert r["measurement_method"] in ("full_patch", "atp_approximation")


def test_trace_includes_pinned_modules(gpt2_model):
    """F-015 fix: the embedding (transformer.wte) was previously excluded
    from top-K when the activation-norm proxy ranked it low. The 1.0
    pinned-modules default ensures embed/unembed/final_norm are always
    measured, so they show up in the results when their effect is large.
    """
    out = gpt2_model.trace(
        "The capital of France is",
        "The capital of Italy is",
        top_k=10,
    )
    measured_modules = {r["module"] for r in out["results"]}
    pinned = set(out["meta"]["pinned_modules"])
    # All pinned modules should appear in the measured set.
    assert pinned <= measured_modules


def test_trace_meta_block_structure(gpt2_model):
    out = gpt2_model.trace("hello", "goodbye", top_k=3)
    meta = out["meta"]
    assert "algorithm" in meta
    assert "total_candidates" in meta
    assert "candidates_full_patched" in meta
    assert "runtime_seconds" in meta
    assert isinstance(meta["runtime_seconds"], float)
