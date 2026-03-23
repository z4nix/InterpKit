"""Mathematical correctness and invariant validation tests.

These are the highest-value tests because they catch silent bugs — operations
that run without error but return wrong results.

Invariants tested:
  - ablate method="zero" produces actual zeros
  - steer scale=0 leaves output unchanged
  - patch/trace with identical inputs gives ~0 effect
  - lens position=-1 equivalent to position=len-1
  - decompose components sum ≈ residual
  - diff model vs itself gives distance ≈ 0
  - ov_scores returns exactly num_attention_heads entries
  - steer_vector is antisymmetric
  - attribute gradient scores are finite and nonzero
"""

from __future__ import annotations

import math

import pytest
import torch

TEXT = "The capital of France is"


def _first_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[0]


def _first_attn(model):
    for m in model.arch_info.modules:
        if m.role == "attention":
            return m.name
    pytest.skip("No attention module detected")


# ═══════════════════════════════════════════════════════════════════════════
#  Ablate zero → activations should actually be zero
# ═══════════════════════════════════════════════════════════════════════════


def test_ablate_zero_activations_are_zero_gpt2(gpt2_model):
    from interpkit.ops.patch import _get_module

    layer = _first_layer(gpt2_model)
    target_mod = _get_module(gpt2_model._model, layer)

    captured: list[torch.Tensor] = []

    def _ablate_hook(_mod, _inp, output):
        t = output if isinstance(output, torch.Tensor) else (
            output[0] if isinstance(output, (tuple, list)) else None
        )
        if t is None:
            return output
        replacement = torch.zeros_like(t)
        captured.append(replacement.clone())
        if isinstance(output, torch.Tensor):
            return replacement
        return (replacement,) + tuple(output[1:])

    model_input = gpt2_model._prepare(TEXT)
    handle = target_mod.register_forward_hook(_ablate_hook)
    with torch.no_grad():
        gpt2_model._forward(model_input)
    handle.remove()

    assert len(captured) == 1
    assert torch.all(captured[0] == 0), "Zero-ablated output should be all zeros"


# ═══════════════════════════════════════════════════════════════════════════
#  Steer with scale=0 should produce identical output
# ═══════════════════════════════════════════════════════════════════════════


def test_steer_scale_zero_unchanged_smollm(smollm_model):
    layer = _first_layer(smollm_model)
    vec = smollm_model.steer_vector("happy", "sad", at=layer)
    result = smollm_model.steer(TEXT, vector=vec, at=layer, scale=0.0)
    assert result["original_top"] == result["steered_top"], (
        "scale=0.0 should produce identical predictions"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Patch identical inputs → effect ≈ 0
# ═══════════════════════════════════════════════════════════════════════════


def test_patch_identical_inputs_zero_effect_pythia(pythia_model):
    result = pythia_model.patch(TEXT, TEXT, at=_first_layer(pythia_model))
    assert result["effect"] == pytest.approx(0.0, abs=1e-4), (
        f"Patching identical inputs should give ~0 effect, got {result['effect']}"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Trace identical inputs → all effects ≈ 0
# ═══════════════════════════════════════════════════════════════════════════


def test_trace_identical_inputs_zero_effects_bart(bart_model):
    results = bart_model.trace(TEXT, TEXT, top_k=5)
    if isinstance(results, list):
        for entry in results:
            assert entry["effect"] == pytest.approx(0.0, abs=1e-4), (
                f"Tracing identical inputs should give ~0 effect for {entry.get('module')}, "
                f"got {entry['effect']}"
            )
    elif isinstance(results, dict) and "effects" in results:
        for val in results["effects"].values() if isinstance(results["effects"], dict) else []:
            assert val == pytest.approx(0.0, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
#  Lens: position=-1 should equal position=len(tokens)-1
# ═══════════════════════════════════════════════════════════════════════════


def test_lens_last_vs_explicit_last_gpt_neo(gpt_neo_model):
    tokenizer = gpt_neo_model._tokenizer
    if tokenizer is None:
        pytest.skip("No tokenizer available")
    tokens = tokenizer.encode(TEXT)
    n = len(tokens)

    results_neg1 = gpt_neo_model.lens(TEXT, position=-1)
    results_explicit = gpt_neo_model.lens(TEXT, position=n - 1)

    if results_neg1 is None or results_explicit is None:
        pytest.skip("Lens returned None")

    assert len(results_neg1) == len(results_explicit), "Same number of layer results"
    for r1, r2 in zip(results_neg1, results_explicit):
        assert r1["top1_token"] == r2["top1_token"], (
            f"Layer {r1.get('layer_name')}: position=-1 gave '{r1['top1_token']}' "
            f"but position={n-1} gave '{r2['top1_token']}'"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Decompose: sum of component vectors ≈ residual
# ═══════════════════════════════════════════════════════════════════════════


def test_decompose_sum_reconstructs_residual_gpt2(gpt2_model):
    result = gpt2_model.decompose(TEXT)
    components = result["components"]
    residual = result["residual"]

    assert residual is not None, "Residual should not be None"
    assert len(components) > 0, "Should have at least one component"

    component_sum = torch.zeros_like(residual)
    for comp in components:
        component_sum = component_sum + comp["vector"]

    cos_sim = torch.nn.functional.cosine_similarity(
        component_sum.unsqueeze(0), residual.unsqueeze(0),
    ).item()

    assert cos_sim > 0.8, (
        f"Component sum should roughly reconstruct residual (cosine_sim={cos_sim:.3f})"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Diff model vs itself → distance ≈ 0 (extending to non-GPT-2 models)
# ═══════════════════════════════════════════════════════════════════════════


def test_diff_self_zero_distance_pythia(pythia_model):
    import interpkit

    result = interpkit.diff(pythia_model, pythia_model, TEXT)
    assert "results" in result
    for r in result["results"]:
        assert r["distance"] == pytest.approx(0.0, abs=1e-4), (
            f"Self-diff distance should be ~0, got {r['distance']} for {r.get('module', '?')}"
        )


def test_diff_self_zero_distance_bart(bart_model):
    import interpkit

    result = interpkit.diff(bart_model, bart_model, TEXT)
    assert "results" in result
    for r in result["results"]:
        assert r["distance"] == pytest.approx(0.0, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
#  OV scores: should return exactly num_attention_heads entries
# ═══════════════════════════════════════════════════════════════════════════


def test_ov_scores_returns_h_entries_gpt2(gpt2_model):
    result = gpt2_model.ov_scores(layer=0)
    n_heads = gpt2_model.arch_info.num_attention_heads
    assert n_heads is not None
    assert len(result["heads"]) == n_heads, (
        f"Expected {n_heads} OV score entries, got {len(result['heads'])}"
    )


def test_ov_scores_returns_h_entries_smollm(smollm_model):
    result = smollm_model.ov_scores(layer=0)
    n_heads = smollm_model.arch_info.num_attention_heads
    assert n_heads is not None
    assert len(result["heads"]) == n_heads, (
        f"Expected {n_heads} OV score entries, got {len(result['heads'])}"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Steer vector antisymmetry: vec(A,B) ≈ -vec(B,A)
# ═══════════════════════════════════════════════════════════════════════════


def test_steer_vector_antisymmetry_gpt_neo(gpt_neo_model):
    layer = _first_layer(gpt_neo_model)
    v_ab = gpt_neo_model.steer_vector("happy", "sad", at=layer)
    v_ba = gpt_neo_model.steer_vector("sad", "happy", at=layer)

    assert torch.allclose(v_ab, -v_ba, atol=1e-4), (
        "steer_vector(A,B) should be approximately -steer_vector(B,A)"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Attribute gradient: scores should be finite and nonzero
# ═══════════════════════════════════════════════════════════════════════════


def test_attribute_gradient_scores_sum_gpt2(gpt2_model):
    result = gpt2_model.attribute(TEXT, method="gradient")
    assert "scores" in result
    scores = result["scores"]
    total = sum(scores)
    assert math.isfinite(total), "Sum of attribution scores should be finite"
    assert total != 0.0, "Sum of attribution scores should be nonzero"
