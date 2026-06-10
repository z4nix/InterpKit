"""Tests for the public Attribution Patching op (model.atp / ops.atp)."""

from __future__ import annotations

import math

import pytest

CLEAN = "The capital of France is"
CORRUPTED = "The capital of Germany is"


def test_atp_schema_and_sorting(gpt2_model):
    result = gpt2_model.atp(CLEAN, CORRUPTED, top_k=10)
    results = result["results"]
    assert len(results) == 10
    for i, r in enumerate(results):
        assert set(r) >= {"module", "role", "score", "rank"}
        assert r["rank"] == i
    finite = [abs(r["score"]) for r in results if not math.isnan(r["score"])]
    assert finite == sorted(finite, reverse=True)
    meta = result["meta"]
    assert meta["method"] == "atp"
    assert meta["n_forward_passes"] == 2
    assert meta["n_backward_passes"] == 1
    assert "first-order" in meta["caveat"]


def test_atp_deterministic(gpt2_model):
    a = gpt2_model.atp(CLEAN, CORRUPTED, top_k=5)
    b = gpt2_model.atp(CLEAN, CORRUPTED, top_k=5)
    assert [(r["module"], r["score"]) for r in a["results"]] == [
        (r["module"], r["score"]) for r in b["results"]
    ]


def test_atp_top_k_none_returns_all(gpt2_model):
    result = gpt2_model.atp(CLEAN, CORRUPTED, top_k=None)
    assert len(result["results"]) == result["meta"]["n_modules"]


def test_atp_overlaps_with_exhaustive_trace(gpt2_model):
    """AtP's top modules should substantially overlap the true causal top
    modules from exhaustive tracing (the F-015 rationale for adopting AtP)."""
    atp_top = {
        r["module"] for r in gpt2_model.atp(CLEAN, CORRUPTED, top_k=10)["results"]
    }
    trace_result = gpt2_model.trace(CLEAN, CORRUPTED, top_k=10, method="exhaustive")
    trace_top = {r["module"] for r in trace_result["results"]}
    assert len(atp_top & trace_top) >= 3


def test_atp_unsupported_metric_nan_fallthrough(gpt2_model):
    result = gpt2_model.atp(CLEAN, CORRUPTED, top_k=5, metric="kl_div")
    assert all(math.isnan(r["score"]) for r in result["results"])
    assert "warning" in result["meta"]


def test_atp_invalid_metric_rejected(gpt2_model):
    with pytest.raises(ValueError, match="Unknown metric"):
        gpt2_model.atp(CLEAN, CORRUPTED, metric="vibes")


def test_atp_shim_import_still_works():
    from interpkit.ops._atp import compute_atp_scores as shim
    from interpkit.ops.atp import compute_atp_scores as canonical

    assert shim is canonical
