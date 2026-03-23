"""Tests for untested parameter variants and position/token-specific arguments.

Covers gaps 1 and 2 from the reliability audit:
  - ablate method="resample"
  - head_activations output_proj=False
  - patch head=, positions=, metric= (kl_div, target_prob, l2_prob)
  - trace alternate metric= values
  - composition comp_type="k" and "v"
  - find_circuit method="zero", method="resample", alternate metrics
  - attention causal= override
  - decompose position=0
  - dla token= (string) and position=
  - lens arbitrary positions
"""

from __future__ import annotations

import pytest
import torch

TEXT = "The capital of France is"
TEXT_B = "The capital of Germany is"
PAIR_CLEAN = "The Eiffel Tower is in Paris"
PAIR_CORRUPT = "The Eiffel Tower is in Rome"


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


def _has_two_attn_layers(model):
    infos = model.arch_info.layer_infos
    return sum(1 for li in infos if li.attn_path is not None) >= 2


# ═══════════════════════════════════════════════════════════════════════════
#  Ablate — method="resample"
# ═══════════════════════════════════════════════════════════════════════════


def test_ablate_resample_pythia(pythia_model):
    result = pythia_model.ablate(
        TEXT, at=_first_layer(pythia_model),
        method="resample", reference=TEXT_B,
    )
    assert "effect" in result
    assert isinstance(result["effect"], float)
    assert result["method"] == "resample"


# ═══════════════════════════════════════════════════════════════════════════
#  Head activations — output_proj=False
# ═══════════════════════════════════════════════════════════════════════════


def test_head_activations_no_output_proj_bart(bart_model):
    at = _first_attn(bart_model)
    result_proj = bart_model.head_activations(TEXT, at=at, output_proj=True)
    result_raw = bart_model.head_activations(TEXT, at=at, output_proj=False)

    assert "head_acts" in result_raw
    assert result_raw["num_heads"] > 0

    proj_shape = result_proj["head_acts"].shape
    raw_shape = result_raw["head_acts"].shape
    assert raw_shape[-1] <= proj_shape[-1], (
        f"Raw head dim ({raw_shape[-1]}) should be <= projected dim ({proj_shape[-1]})"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Patch — head=, positions=, metric= variants
# ═══════════════════════════════════════════════════════════════════════════


def test_patch_with_head_gpt_neo(gpt_neo_model):
    at = _first_attn(gpt_neo_model)
    result = gpt_neo_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=at, head=0)
    assert "effect" in result
    assert isinstance(result["effect"], float)


def test_patch_with_positions_gpt_neo(gpt_neo_model):
    result = gpt_neo_model.patch(
        PAIR_CLEAN, PAIR_CORRUPT,
        at=_first_layer(gpt_neo_model),
        positions=[0, 2],
    )
    assert "effect" in result
    assert isinstance(result["effect"], float)


def test_patch_metric_kl_div_smollm(smollm_model):
    result = smollm_model.patch(
        PAIR_CLEAN, PAIR_CORRUPT,
        at=_first_layer(smollm_model),
        metric="kl_div",
    )
    assert "effect" in result
    assert isinstance(result["effect"], float)


def test_patch_metric_target_prob_smollm(smollm_model):
    result = smollm_model.patch(
        PAIR_CLEAN, PAIR_CORRUPT,
        at=_first_layer(smollm_model),
        metric="target_prob",
    )
    assert "effect" in result
    assert isinstance(result["effect"], float)


def test_patch_metric_l2_prob_gpt2(gpt2_model):
    result = gpt2_model.patch(
        PAIR_CLEAN, PAIR_CORRUPT,
        at=_first_layer(gpt2_model),
        metric="l2_prob",
    )
    assert "effect" in result
    assert isinstance(result["effect"], float)


# ═══════════════════════════════════════════════════════════════════════════
#  Trace — alternate metric= values
# ═══════════════════════════════════════════════════════════════════════════


def test_trace_metric_kl_div_pythia(pythia_model):
    results = pythia_model.trace(
        PAIR_CLEAN, PAIR_CORRUPT, top_k=3, metric="kl_div",
    )
    assert isinstance(results, list)
    assert len(results) > 0
    assert "module" in results[0]
    assert "effect" in results[0]


def test_trace_metric_target_prob_bart(bart_model):
    results = bart_model.trace(
        PAIR_CLEAN, PAIR_CORRUPT, top_k=3, metric="target_prob",
    )
    assert isinstance(results, (list, dict))


def test_trace_mode_position_electra(electra_model):
    result = electra_model.trace(PAIR_CLEAN, PAIR_CORRUPT, mode="position")
    assert isinstance(result, dict)
    assert "effects" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Composition — comp_type="k" and "v"
# ═══════════════════════════════════════════════════════════════════════════


def test_composition_k_opt(opt_model):
    if not _has_two_attn_layers(opt_model):
        pytest.skip("Not enough attention layers")
    result = opt_model.composition(src_layer=0, dst_layer=1, comp_type="k")
    assert "scores" in result


def test_composition_v_albert(albert_model):
    if not _has_two_attn_layers(albert_model):
        pytest.skip("Not enough attention layers")
    result = albert_model.composition(src_layer=0, dst_layer=1, comp_type="v")
    assert "scores" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Find circuit — method and metric variants
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.timeout(180)
def test_find_circuit_method_zero_gpt2(gpt2_model):
    result = gpt2_model.find_circuit(
        PAIR_CLEAN, PAIR_CORRUPT, threshold=0.05, method="zero",
    )
    assert "circuit" in result
    assert isinstance(result["circuit"], list)
    assert "excluded" in result


@pytest.mark.timeout(180)
def test_find_circuit_method_resample_gpt2(gpt2_model):
    result = gpt2_model.find_circuit(
        PAIR_CLEAN, PAIR_CORRUPT, threshold=0.05, method="resample",
    )
    assert "circuit" in result
    assert isinstance(result["circuit"], list)


@pytest.mark.timeout(180)
def test_find_circuit_metric_kl_div_gpt2(gpt2_model):
    result = gpt2_model.find_circuit(
        PAIR_CLEAN, PAIR_CORRUPT, threshold=0.05, metric="kl_div",
    )
    assert "circuit" in result
    assert "verification" in result


@pytest.mark.timeout(180)
def test_find_circuit_metric_target_prob_gpt2(gpt2_model):
    result = gpt2_model.find_circuit(
        PAIR_CLEAN, PAIR_CORRUPT, threshold=0.05, metric="target_prob",
    )
    assert "circuit" in result
    assert "verification" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Attention — causal= override
# ═══════════════════════════════════════════════════════════════════════════


def test_attention_causal_override_distilbert(distilbert_model):
    results = distilbert_model.attention(TEXT, causal=True)
    if results is None:
        pytest.skip("Attention extraction not supported")
    assert isinstance(results, list)
    assert len(results) > 0
    weights = results[0]["weights"]
    n = weights.shape[-1]
    if n > 1:
        assert weights[0, -1].item() < 1e-6, (
            "With causal=True, upper-triangular entries should be masked"
        )


def test_attention_causal_false_gpt2(gpt2_model):
    results = gpt2_model.attention(TEXT, causal=False)
    assert isinstance(results, list)
    assert len(results) > 0
    weights = results[0]["weights"]
    n = weights.shape[-1]
    if n > 1:
        assert weights[0, -1].item() > 0, (
            "With causal=False, upper-triangular entries should NOT be masked"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Position-specific arguments
# ═══════════════════════════════════════════════════════════════════════════


def test_decompose_position_zero_pythia(pythia_model):
    result = pythia_model.decompose(TEXT, position=0)
    assert "components" in result
    assert isinstance(result["components"], list)


def test_decompose_position_middle_flan_t5(flan_t5_model):
    result = flan_t5_model.decompose(TEXT, position=2)
    assert "components" in result
    assert isinstance(result["components"], list)


def test_dla_token_string_smollm(smollm_model):
    result = smollm_model.dla(TEXT, token="Paris")
    assert "contributions" in result
    assert isinstance(result["contributions"], list)


def test_dla_position_2_opt(opt_model):
    result = opt_model.dla(TEXT, position=2)
    assert "contributions" in result
    assert isinstance(result["contributions"], list)


def test_lens_position_zero_pythia(pythia_model):
    results = pythia_model.lens(TEXT, position=0)
    assert results is None or isinstance(results, list)
    if results is not None:
        assert len(results) > 0


def test_lens_position_middle_bart(bart_model):
    results = bart_model.lens(TEXT, position=2)
    assert results is None or isinstance(results, list)
