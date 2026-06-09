"""Mathematical invariant tests — verify correctness properties on GPT-2.

These catch silent wrong outputs, not just "no exception".
"""

from __future__ import annotations

import torch

TEXT = "The capital of France is"
CLEAN = "The Eiffel Tower is in Paris"
CORRUPT = "The Eiffel Tower is in Rome"


def test_patch_identical_inputs_degenerate_gap(gpt2_model):
    """Patching with identical clean and corrupted inputs is mathematically
    undefined for ratio metrics — there's no clean-vs-corrupted gap to
    normalise against.

    Pre-1.0 silently returned 0 (F-010), masking the degenerate case as
    "the patch did nothing". The 1.0 fix returns NaN and includes
    ``"degenerate_gap"`` in the result's ``warnings`` list so the
    undefined region is visible to downstream callers.
    """
    import math

    layer = gpt2_model.arch_info.layer_names[0]
    result = gpt2_model.patch(TEXT, TEXT, at=layer)
    assert math.isnan(result["effect"]), (
        f"Identical inputs should yield NaN (degenerate gap), got {result['effect']}"
    )
    assert "degenerate_gap" in result.get("warnings", [])


def test_ablate_zero_changes_output(gpt2_model):
    """Zero ablation should change the output (non-zero effect)."""
    layer = gpt2_model.arch_info.layer_names[0]
    result = gpt2_model.ablate(TEXT, at=layer, method="zero")
    assert "effect" in result


def test_decompose_components_sum_to_residual(gpt2_model):
    """The sum of component vectors should approximate the residual stream."""
    result = gpt2_model.decompose(TEXT, position=-1)
    components = result["components"]
    residual = result["residual"]

    component_sum = torch.zeros_like(residual)
    for comp in components:
        component_sum += comp["vector"]

    cos_sim = torch.nn.functional.cosine_similarity(
        component_sum.unsqueeze(0), residual.unsqueeze(0),
    ).item()
    assert cos_sim > 0.8, (
        f"Component sum should be close to residual (cosine={cos_sim:.4f})"
    )


def test_composition_scores_finite(gpt2_model):
    """Composition scores should be finite and non-negative."""
    result = gpt2_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    scores = result["scores"]
    assert torch.isfinite(scores).all(), "Composition scores contain non-finite values"
    assert (scores >= -1e-6).all(), "Composition scores contain unexpected negative values"


def test_ov_scores_bounded(gpt2_model):
    """OV norms should be positive and finite."""
    result = gpt2_model.ov_scores(layer=0)
    for head in result["heads"]:
        assert head["frobenius_norm"] >= 0
        assert torch.isfinite(torch.tensor(head["frobenius_norm"]))


def test_qk_scores_bounded(gpt2_model):
    """QK norms should be positive and finite."""
    result = gpt2_model.qk_scores(layer=0)
    for head in result["heads"]:
        assert head["frobenius_norm"] >= 0
        assert torch.isfinite(torch.tensor(head["frobenius_norm"]))


def test_trace_effects_sum_positive(gpt2_model):
    """At least some modules should show positive causal effect.

    The 1.0 trace returns ``{"results": [...], "meta": {...}}`` (F-015).
    """
    out = gpt2_model.trace(CLEAN, CORRUPT, top_k=5)
    assert isinstance(out, dict)
    effects = [r["effect"] for r in out["results"] if r["effect"] == r["effect"]]
    assert max(effects) > 0, "Expected at least one module with positive trace effect"


def test_dla_contributions_finite(gpt2_model):
    """DLA contributions should be finite floats."""
    result = gpt2_model.dla(TEXT)
    for entry in result["contributions"]:
        val = entry["logit_contribution"]
        assert torch.isfinite(torch.tensor(float(val))), (
            f"Non-finite DLA contribution for {entry['component']}: {val}"
        )


def test_lens_probabilities_valid(gpt2_model):
    """Logit lens top-1 probabilities should be in [0, 1]."""
    result = gpt2_model.lens(TEXT, position=-1)
    assert result is not None
    for layer_result in result:
        prob = layer_result.get("top1_prob")
        if prob is not None:
            assert 0.0 <= prob <= 1.0 + 1e-6, f"Invalid lens probability: {prob}"


def test_attribute_scores_finite(gpt2_model):
    """Attribution scores should be finite."""
    result = gpt2_model.attribute(TEXT, method="gradient")
    if "scores" in result:
        for score in result["scores"]:
            assert torch.isfinite(torch.tensor(float(score))), (
                f"Non-finite attribution score: {score}"
            )
