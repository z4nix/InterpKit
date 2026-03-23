"""Generalization tests — verify InterpKit fixes work beyond the original models.

Each test targets a specific code path that was previously broken and fixed.
If the fixes are genuine generalizations (not model-specific patches), all
tests here should pass.

Models tested:
  - facebook/opt-350m      (decoder-only, separate q_proj/k_proj/v_proj Linear)
  - albert-base-v2         (encoder-only, shared weights across layers)
  - google/flan-t5-small   (encoder-decoder, finetuned T5 variant)
  - EleutherAI/pythia-160m (decoder-only, fused query_key_value like BLOOM)

Fix code paths exercised:
  - _find_output_proj: out_lin / output.dense / deep recursive search
  - _find_proj_weight: q_lin/k_lin/v_lin, query_key_value fused QKV
  - run_attention: eager attn fallback for SDPA
  - run_ov_scores / run_qk_scores: SVD .cpu() fallback, GQA head mapping
  - run_composition: GQA shape handling
  - run_diff: .cpu() before cosine similarity
  - _load_from_hf: auto-class cascade order
  - _make_dummy_input: decoder_input_ids for enc-dec
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


# ═══════════════════════════════════════════════════════════════════════════
# OPT-350m — decoder-only with standard separate q_proj/k_proj/v_proj
# Validates: projection name search, SVD .cpu(), output_proj detection
# ═══════════════════════════════════════════════════════════════════════════


def test_discover_opt(opt_model):
    arch = opt_model.arch_info
    assert arch.is_language_model
    assert len(arch.layer_names) > 0
    assert arch.num_attention_heads is not None
    assert arch.hidden_size is not None


def test_inspect_opt(opt_model):
    opt_model.inspect()


def test_activations_opt(opt_model):
    layer = _first_layer(opt_model)
    result = opt_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)
    assert result.dim() >= 2


def test_head_activations_opt(opt_model):
    """Fix path: _find_output_proj recursive search."""
    at = _first_attn(opt_model)
    result = opt_model.head_activations(TEXT, at=at)
    assert "head_acts" in result
    assert result["num_heads"] > 0


def test_attention_opt(opt_model):
    """Fix path: eager attention fallback."""
    results = opt_model.attention(TEXT)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "layer" in results[0]


def test_ov_scores_opt(opt_model):
    """Fix path: _find_proj_weight with q_proj/v_proj naming, SVD .cpu()."""
    result = opt_model.ov_scores(layer=0)
    assert "heads" in result
    assert isinstance(result["heads"], list)
    assert len(result["heads"]) > 0


def test_qk_scores_opt(opt_model):
    """Fix path: _find_proj_weight with q_proj/k_proj naming, SVD .cpu()."""
    result = opt_model.qk_scores(layer=0)
    assert "heads" in result
    assert len(result["heads"]) > 0


def test_composition_opt(opt_model):
    """Fix path: composition with standard MHA (not GQA)."""
    result = opt_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    assert "scores" in result
    assert result["scores"].shape[0] > 0


def test_decompose_opt(opt_model):
    result = opt_model.decompose(TEXT)
    assert "components" in result
    assert isinstance(result["components"], list)


def test_lens_opt(opt_model):
    results = opt_model.lens(TEXT)
    assert isinstance(results, list)
    assert len(results) > 0


def test_dla_opt(opt_model):
    result = opt_model.dla(TEXT)
    assert "contributions" in result


def test_diff_opt_vs_gpt2(opt_model, gpt2_model):
    """Fix path: .cpu() before cosine similarity (cross-device if OPT on MPS, GPT2 on CPU)."""
    import interpkit

    result = interpkit.diff(opt_model, gpt2_model, TEXT)
    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# ALBERT — encoder-only with shared weights across all layers
# Validates: shared weight handling, query/key/value naming, output.dense
# ═══════════════════════════════════════════════════════════════════════════


def test_discover_albert(albert_model):
    arch = albert_model.arch_info
    assert arch.layer_names
    assert arch.num_attention_heads is not None
    assert arch.hidden_size is not None


def test_inspect_albert(albert_model):
    albert_model.inspect()


def test_activations_albert(albert_model):
    layer = _first_layer(albert_model)
    result = albert_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_head_activations_albert(albert_model):
    """Fix path: _find_output_proj with BERT-style attention.output.dense nesting."""
    at = _first_attn(albert_model)
    result = albert_model.head_activations(TEXT, at=at)
    assert "head_acts" in result
    assert result["num_heads"] > 0


def test_attention_albert(albert_model):
    """Fix path: eager attention fallback for SDPA."""
    results = albert_model.attention(TEXT)
    assert isinstance(results, list)
    assert len(results) > 0


def test_ov_scores_albert(albert_model):
    """Fix path: query/key/value projection naming, SVD .cpu()."""
    result = albert_model.ov_scores(layer=0)
    assert "heads" in result
    assert len(result["heads"]) > 0


def test_qk_scores_albert(albert_model):
    """Fix path: query/key/value projection naming, SVD .cpu()."""
    result = albert_model.qk_scores(layer=0)
    assert "heads" in result


def test_composition_albert(albert_model):
    """Fix path: shared weight layers — composition still valid."""
    infos = albert_model.arch_info.layer_infos
    attn_layers = [li for li in infos if li.attn_path is not None]
    if len(attn_layers) < 2:
        pytest.skip("Model has fewer than 2 attention layers for composition")
    result = albert_model.composition(
        src_layer=attn_layers[0].index,
        dst_layer=attn_layers[1].index,
        comp_type="q",
    )
    assert "scores" in result


def test_decompose_albert(albert_model):
    result = albert_model.decompose(TEXT)
    assert "components" in result


def test_ablate_albert(albert_model):
    result = albert_model.ablate(TEXT, at=_first_layer(albert_model))
    assert "effect" in result


def test_attribute_albert(albert_model):
    result = albert_model.attribute(TEXT, method="gradient")
    assert "tokens" in result


# ═══════════════════════════════════════════════════════════════════════════
# Flan-T5-small — encoder-decoder (validates T5 loading fix generalizes)
# Validates: decoder_input_ids injection, auto-class cascade, enc-dec ops
# ═══════════════════════════════════════════════════════════════════════════


def test_discover_flan_t5(flan_t5_model):
    """Fix path: _load_from_hf auto-class cascade picks Seq2SeqLM."""
    arch = flan_t5_model.arch_info
    assert arch.layer_names
    assert arch.hidden_size is not None


def test_inspect_flan_t5(flan_t5_model):
    flan_t5_model.inspect()


def test_activations_flan_t5(flan_t5_model):
    """Fix path: decoder_input_ids injection during forward pass."""
    layer = _first_layer(flan_t5_model)
    result = flan_t5_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_attention_flan_t5(flan_t5_model):
    """Fix path: eager attention fallback + enc-dec attention."""
    results = flan_t5_model.attention(TEXT)
    assert results is None or isinstance(results, list)


def test_lens_flan_t5(flan_t5_model):
    """Fix path: enc-dec lens (T5 decoder layers)."""
    results = flan_t5_model.lens(TEXT)
    assert results is None or isinstance(results, list)


def test_dla_flan_t5(flan_t5_model):
    if not flan_t5_model.arch_info.is_language_model:
        pytest.skip("Not detected as LM")
    result = flan_t5_model.dla(TEXT)
    assert "contributions" in result


def test_ablate_flan_t5(flan_t5_model):
    result = flan_t5_model.ablate(TEXT, at=_first_layer(flan_t5_model))
    assert "effect" in result


def test_decompose_flan_t5(flan_t5_model):
    result = flan_t5_model.decompose(TEXT)
    assert "components" in result


def test_scan_flan_t5(flan_t5_model):
    result = flan_t5_model.scan(TEXT)
    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Pythia-160m — decoder-only with fused query_key_value (like BLOOM)
# Validates: fused QKV splitting in _find_proj_weight, SVD .cpu()
# ═══════════════════════════════════════════════════════════════════════════


def test_discover_pythia(pythia_model):
    arch = pythia_model.arch_info
    assert arch.is_language_model
    assert len(arch.layer_names) > 0
    assert arch.num_attention_heads is not None


def test_inspect_pythia(pythia_model):
    pythia_model.inspect()


def test_activations_pythia(pythia_model):
    layer = _first_layer(pythia_model)
    result = pythia_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_head_activations_pythia(pythia_model):
    """Fix path: _find_output_proj for GPTNeoX attention."""
    at = _first_attn(pythia_model)
    result = pythia_model.head_activations(TEXT, at=at)
    assert "head_acts" in result
    assert result["num_heads"] > 0


def test_attention_pythia(pythia_model):
    """Fix path: eager attention fallback, fused QKV."""
    results = pythia_model.attention(TEXT)
    assert isinstance(results, list)
    assert len(results) > 0


def test_ov_scores_pythia(pythia_model):
    """Fix path: _find_proj_weight with query_key_value fused projection, SVD .cpu()."""
    result = pythia_model.ov_scores(layer=0)
    assert "heads" in result
    assert len(result["heads"]) > 0


def test_qk_scores_pythia(pythia_model):
    """Fix path: _find_proj_weight with query_key_value fused projection, SVD .cpu()."""
    result = pythia_model.qk_scores(layer=0)
    assert "heads" in result
    assert len(result["heads"]) > 0


def test_composition_pythia(pythia_model):
    """Fix path: composition with fused QKV projections."""
    result = pythia_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    assert "scores" in result
    assert result["scores"].shape[0] > 0


def test_decompose_pythia(pythia_model):
    result = pythia_model.decompose(TEXT)
    assert "components" in result


def test_lens_pythia(pythia_model):
    results = pythia_model.lens(TEXT)
    assert isinstance(results, list)
    assert len(results) > 0


def test_dla_pythia(pythia_model):
    result = pythia_model.dla(TEXT)
    assert "contributions" in result


def test_diff_pythia_vs_gpt2(pythia_model, gpt2_model):
    """Fix path: cross-device diff (.cpu() before cosine sim)."""
    import interpkit

    result = interpkit.diff(pythia_model, gpt2_model, TEXT)
    assert isinstance(result, dict)
