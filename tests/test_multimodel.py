"""Multi-model stress tests for InterpKit.

Tests every InterpKit operation across 10 architecturally diverse models to
surface failure modes in architecture discovery, attention extraction,
gradient attribution, circuit analysis, and more.

Models tested:
  - gpt2                        (decoder-only, Conv1D MHA)
  - distilbert-base-uncased     (encoder-only, MaskedLM, Linear Q/K/V)
  - t5-small                    (encoder-decoder, Seq2SeqLM, cross-attention)
  - google/vit-base-patch16-224 (vision transformer, image input)
  - microsoft/resnet-18         (CNN, no transformer)
  - google/recurrentgemma-2b-it (hybrid recurrent + local attention)
  - Qwen/Qwen2-0.5B            (GQA, RoPE)
  - bigscience/bloom-560m       (ALiBi, fused QKV)
  - state-spaces/mamba-130m     (pure SSM, zero attention)
  - deepset/roberta-base-squad2 (QA model, qa_outputs head)
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

slow = pytest.mark.timeout(300)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[0]


def _mid_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[len(layers) // 2]


def _first_attn(model):
    for m in model.arch_info.modules:
        if m.role == "attention":
            return m.name
    pytest.skip("No attention module detected")


def _first_mlp(model):
    for m in model.arch_info.modules:
        if m.role == "mlp":
            return m.name
    pytest.skip("No MLP module detected")


def _any_param_module(model):
    for m in model.arch_info.modules:
        if m.param_count > 0:
            return m.name
    pytest.skip("No parameterized modules")


TEXT = "The capital of France is"
TEXT_B = "The capital of Germany is"
PAIR_CLEAN = "The Eiffel Tower is in Paris"
PAIR_CORRUPT = "The Eiffel Tower is in Rome"


# ═══════════════════════════════════════════════════════════════════════════
# 1. DISCOVERY & LOADING
# ═══════════════════════════════════════════════════════════════════════════


def test_discover_gpt2(gpt2_model):
    arch = gpt2_model.arch_info
    assert arch.is_language_model
    assert len(arch.layer_names) == 12
    assert arch.num_attention_heads == 12
    assert arch.hidden_size == 768
    assert arch.unembedding_name is not None


def test_discover_distilbert(distilbert_model):
    arch = distilbert_model.arch_info
    assert arch.layer_names
    assert arch.num_attention_heads is not None
    assert arch.hidden_size is not None
    # FAILURE_MODE_CHECK: DistilBERT MaskedLM unembedding detection


def test_discover_t5(t5_model):
    arch = t5_model.arch_info
    assert arch.layer_names
    # FAILURE_MODE_CHECK: T5 uses d_model config key
    assert arch.hidden_size is not None


def test_discover_vit(vit_model):
    arch = vit_model.arch_info
    assert arch.layer_names
    assert arch.num_attention_heads is not None
    # FAILURE_MODE_CHECK: classifier should NOT be unembedding
    assert not arch.is_language_model


def test_discover_resnet(resnet_model):
    arch = resnet_model.arch_info
    assert not arch.is_language_model
    assert arch.num_attention_heads is None


@slow
def test_discover_recurrentgemma(recurrentgemma_model):
    arch = recurrentgemma_model.arch_info
    assert arch.is_language_model
    assert arch.layer_names
    # FAILURE_MODE_CHECK: mixed recurrent/attention blocks


def test_discover_qwen2(qwen2_model):
    arch = qwen2_model.arch_info
    assert arch.is_language_model
    assert arch.layer_names
    assert arch.num_attention_heads is not None
    # FAILURE_MODE_CHECK: GQA — num_key_value_heads != num_attention_heads


def test_discover_bloom(bloom_model):
    arch = bloom_model.arch_info
    assert arch.is_language_model
    assert arch.layer_names
    assert arch.num_attention_heads is not None
    # FAILURE_MODE_CHECK: ALiBi — no position embeddings


def test_discover_mamba(mamba_model):
    arch = mamba_model.arch_info
    assert arch.layer_names
    # FAILURE_MODE_CHECK: pure SSM — num_attention_heads should be None
    assert arch.is_language_model  # has lm_head


def test_discover_roberta_qa(roberta_qa_model):
    arch = roberta_qa_model.arch_info
    # FAILURE_MODE_CHECK: qa_outputs should NOT be detected as LM head
    assert not arch.is_language_model


# ═══════════════════════════════════════════════════════════════════════════
# 2. INSPECT
# ═══════════════════════════════════════════════════════════════════════════


def test_inspect_gpt2(gpt2_model):
    gpt2_model.inspect()


def test_inspect_distilbert(distilbert_model):
    distilbert_model.inspect()


def test_inspect_t5(t5_model):
    t5_model.inspect()


def test_inspect_vit(vit_model):
    vit_model.inspect()


def test_inspect_resnet(resnet_model):
    resnet_model.inspect()


@slow
def test_inspect_recurrentgemma(recurrentgemma_model):
    recurrentgemma_model.inspect()


def test_inspect_qwen2(qwen2_model):
    qwen2_model.inspect()


def test_inspect_bloom(bloom_model):
    bloom_model.inspect()


def test_inspect_mamba(mamba_model):
    mamba_model.inspect()


def test_inspect_roberta_qa(roberta_qa_model):
    roberta_qa_model.inspect()


# ═══════════════════════════════════════════════════════════════════════════
# 3. ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════


def test_activations_gpt2(gpt2_model):
    layer = _first_layer(gpt2_model)
    result = gpt2_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)
    assert result.dim() >= 2


def test_activations_distilbert(distilbert_model):
    layer = _first_layer(distilbert_model)
    result = distilbert_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_activations_t5(t5_model):
    layer = _first_layer(t5_model)
    # FAILURE_MODE_CHECK: T5 encoder/decoder layer naming
    result = t5_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_activations_vit(vit_model, test_image_path):
    layer = _first_layer(vit_model)
    result = vit_model.activations(test_image_path, at=layer)
    assert isinstance(result, torch.Tensor)


def test_activations_resnet(resnet_model, test_image_path):
    mod = _any_param_module(resnet_model)
    result = resnet_model.activations(test_image_path, at=mod)
    assert isinstance(result, torch.Tensor)


@slow
def test_activations_recurrentgemma(recurrentgemma_model):
    layer = _first_layer(recurrentgemma_model)
    result = recurrentgemma_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_activations_qwen2(qwen2_model):
    layer = _first_layer(qwen2_model)
    result = qwen2_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_activations_bloom(bloom_model):
    layer = _first_layer(bloom_model)
    result = bloom_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_activations_mamba(mamba_model):
    layer = _first_layer(mamba_model)
    # FAILURE_MODE_CHECK: Mamba SSM layers
    result = mamba_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_activations_roberta_qa(roberta_qa_model):
    layer = _first_layer(roberta_qa_model)
    result = roberta_qa_model.activations(TEXT, at=layer)
    assert isinstance(result, torch.Tensor)


def test_activations_multi_module(gpt2_model):
    layers = gpt2_model.arch_info.layer_names[:2]
    result = gpt2_model.activations(TEXT, at=layers)
    assert isinstance(result, dict)
    assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. CACHE
# ═══════════════════════════════════════════════════════════════════════════


def test_cache_gpt2(gpt2_model):
    gpt2_model.cache(TEXT)
    assert gpt2_model.cached
    gpt2_model.clear_cache()
    assert not gpt2_model.cached


def test_cache_distilbert(distilbert_model):
    distilbert_model.cache(TEXT)
    assert distilbert_model.cached
    distilbert_model.clear_cache()


def test_cache_t5(t5_model):
    t5_model.cache(TEXT)
    assert t5_model.cached
    t5_model.clear_cache()


def test_cache_vit(vit_model, test_image_path):
    vit_model.cache(test_image_path)
    assert vit_model.cached
    vit_model.clear_cache()


def test_cache_resnet(resnet_model, test_image_path):
    resnet_model.cache(test_image_path)
    assert resnet_model.cached
    resnet_model.clear_cache()


def test_cache_qwen2(qwen2_model):
    qwen2_model.cache(TEXT)
    assert qwen2_model.cached
    qwen2_model.clear_cache()


def test_cache_bloom(bloom_model):
    bloom_model.cache(TEXT)
    assert bloom_model.cached
    bloom_model.clear_cache()


def test_cache_mamba(mamba_model):
    mamba_model.cache(TEXT)
    assert mamba_model.cached
    mamba_model.clear_cache()


def test_cache_roberta_qa(roberta_qa_model):
    roberta_qa_model.cache(TEXT)
    assert roberta_qa_model.cached
    roberta_qa_model.clear_cache()


@slow
def test_cache_recurrentgemma(recurrentgemma_model):
    recurrentgemma_model.cache(TEXT)
    assert recurrentgemma_model.cached
    recurrentgemma_model.clear_cache()


# ═══════════════════════════════════════════════════════════════════════════
# 5. HEAD ACTIVATIONS (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


def test_head_activations_gpt2(gpt2_model):
    at = _first_attn(gpt2_model)
    result = gpt2_model.head_activations(TEXT, at=at)
    assert "head_acts" in result
    assert "num_heads" in result
    assert result["num_heads"] == 12


def test_head_activations_distilbert(distilbert_model):
    # FAILURE_MODE_CHECK: DistilBERT uses out_lin — _find_output_proj may miss it
    at = _first_attn(distilbert_model)
    result = distilbert_model.head_activations(TEXT, at=at)
    assert "head_acts" in result
    assert result["num_heads"] > 0


def test_head_activations_t5(t5_model):
    at = _first_attn(t5_model)
    result = t5_model.head_activations(TEXT, at=at)
    assert "head_acts" in result


def test_head_activations_vit(vit_model, test_image_path):
    # FAILURE_MODE_CHECK: ViT output.dense naming
    at = _first_attn(vit_model)
    result = vit_model.head_activations(test_image_path, at=at)
    assert "head_acts" in result


def test_head_activations_qwen2(qwen2_model):
    # FAILURE_MODE_CHECK: GQA — num_heads from config is Q heads, not KV heads
    at = _first_attn(qwen2_model)
    result = qwen2_model.head_activations(TEXT, at=at)
    assert "head_acts" in result
    assert result["num_heads"] > 0


def test_head_activations_bloom(bloom_model):
    at = _first_attn(bloom_model)
    result = bloom_model.head_activations(TEXT, at=at)
    assert "head_acts" in result


@slow
def test_head_activations_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: recurrent layers have no heads — should only work on attention layers
    at = _first_attn(recurrentgemma_model)
    result = recurrentgemma_model.head_activations(TEXT, at=at)
    assert "head_acts" in result


# ═══════════════════════════════════════════════════════════════════════════
# 6. ATTENTION
# ═══════════════════════════════════════════════════════════════════════════


def test_attention_gpt2(gpt2_model):
    results = gpt2_model.attention(TEXT)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "layer" in results[0]
    assert "head" in results[0]


def test_attention_gpt2_layer_filter(gpt2_model):
    results = gpt2_model.attention(TEXT, layer=0)
    assert all(r["layer"] == 0 for r in results)


def test_attention_gpt2_head_filter(gpt2_model):
    results = gpt2_model.attention(TEXT, layer=0, head=0)
    assert len(results) == 1


def test_attention_distilbert(distilbert_model):
    # FAILURE_MODE_CHECK: DistilBERT attention naming
    results = distilbert_model.attention(TEXT)
    assert isinstance(results, list)
    assert len(results) > 0


def test_attention_t5(t5_model):
    # FAILURE_MODE_CHECK: T5 cross-attention
    results = t5_model.attention(TEXT)
    assert results is None or isinstance(results, list)


def test_attention_vit(vit_model, test_image_path):
    results = vit_model.attention(test_image_path)
    assert results is None or isinstance(results, list)


def test_attention_resnet_none(resnet_model, test_image_path):
    result = resnet_model.attention(test_image_path)
    assert result is None


def test_attention_qwen2(qwen2_model):
    # FAILURE_MODE_CHECK: GQA — _qk_to_attention must broadcast KV heads
    results = qwen2_model.attention(TEXT)
    assert results is None or isinstance(results, list)


def test_attention_bloom(bloom_model):
    # FAILURE_MODE_CHECK: ALiBi attention bias
    results = bloom_model.attention(TEXT)
    assert results is None or isinstance(results, list)


def test_attention_mamba_none(mamba_model):
    # FAILURE_MODE_CHECK: pure SSM — should return None
    result = mamba_model.attention(TEXT)
    assert result is None


@slow
def test_attention_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: local windowed attention on some layers
    results = recurrentgemma_model.attention(TEXT)
    assert results is None or isinstance(results, list)


def test_attention_roberta_qa(roberta_qa_model):
    results = roberta_qa_model.attention(TEXT)
    assert results is None or isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════════
# 7. ABLATE
# ═══════════════════════════════════════════════════════════════════════════


def test_ablate_gpt2_zero(gpt2_model):
    result = gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model), method="zero")
    assert "effect" in result
    assert isinstance(result["effect"], float)


def test_ablate_gpt2_mean(gpt2_model):
    result = gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model), method="mean")
    assert result["method"] == "mean"


def test_ablate_distilbert(distilbert_model):
    result = distilbert_model.ablate(TEXT, at=_first_layer(distilbert_model))
    assert "effect" in result


def test_ablate_t5(t5_model):
    # FAILURE_MODE_CHECK: T5 layer naming
    result = t5_model.ablate(TEXT, at=_first_layer(t5_model))
    assert "effect" in result


def test_ablate_vit(vit_model, test_image_path):
    result = vit_model.ablate(test_image_path, at=_first_layer(vit_model))
    assert "effect" in result


def test_ablate_resnet(resnet_model, test_image_path):
    mod = _any_param_module(resnet_model)
    result = resnet_model.ablate(test_image_path, at=mod)
    assert "effect" in result


def test_ablate_qwen2(qwen2_model):
    result = qwen2_model.ablate(TEXT, at=_first_layer(qwen2_model))
    assert "effect" in result


def test_ablate_bloom(bloom_model):
    result = bloom_model.ablate(TEXT, at=_first_layer(bloom_model))
    assert "effect" in result


def test_ablate_mamba(mamba_model):
    result = mamba_model.ablate(TEXT, at=_first_layer(mamba_model))
    assert "effect" in result


def test_ablate_roberta_qa(roberta_qa_model):
    result = roberta_qa_model.ablate(TEXT, at=_first_layer(roberta_qa_model))
    assert "effect" in result


@slow
def test_ablate_recurrentgemma(recurrentgemma_model):
    result = recurrentgemma_model.ablate(TEXT, at=_first_layer(recurrentgemma_model))
    assert "effect" in result


# ═══════════════════════════════════════════════════════════════════════════
# 8. ATTRIBUTE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", ["gradient", "gradient_x_input", "integrated_gradients"])
def test_attribute_gpt2(gpt2_model, method):
    result = gpt2_model.attribute(TEXT, method=method, n_steps=5)
    assert "tokens" in result or "scores" in result or "grad" in result


def test_attribute_distilbert(distilbert_model):
    result = distilbert_model.attribute(TEXT, method="gradient")
    assert "tokens" in result


def test_attribute_distilbert_ig(distilbert_model):
    # FAILURE_MODE_CHECK: _find_embedding on DistilBERT
    result = distilbert_model.attribute(TEXT, method="integrated_gradients", n_steps=5)
    assert "tokens" in result


def test_attribute_t5(t5_model):
    # FAILURE_MODE_CHECK: T5 shared embedding for IG
    result = t5_model.attribute(TEXT, method="gradient")
    assert "tokens" in result or "scores" in result


def test_attribute_t5_ig(t5_model):
    result = t5_model.attribute(TEXT, method="integrated_gradients", n_steps=5)
    assert "tokens" in result or "scores" in result


def test_attribute_vit(vit_model, test_image_path):
    result = vit_model.attribute(test_image_path, method="gradient")
    assert "grad" in result or "scores" in result


def test_attribute_resnet(resnet_model, test_image_path):
    result = resnet_model.attribute(test_image_path, method="gradient")
    assert "grad" in result


def test_attribute_qwen2(qwen2_model):
    result = qwen2_model.attribute(TEXT, method="gradient")
    assert "tokens" in result


def test_attribute_bloom(bloom_model):
    # FAILURE_MODE_CHECK: BLOOM no position embedding for IG
    result = bloom_model.attribute(TEXT, method="gradient")
    assert "tokens" in result


def test_attribute_mamba(mamba_model):
    result = mamba_model.attribute(TEXT, method="gradient")
    assert "tokens" in result


def test_attribute_roberta_qa(roberta_qa_model):
    # FAILURE_MODE_CHECK: QA model target token semantics
    result = roberta_qa_model.attribute(TEXT, method="gradient")
    assert "tokens" in result or "scores" in result


@slow
def test_attribute_recurrentgemma(recurrentgemma_model):
    result = recurrentgemma_model.attribute(TEXT, method="gradient")
    assert "tokens" in result


# ═══════════════════════════════════════════════════════════════════════════
# 9. LENS
# ═══════════════════════════════════════════════════════════════════════════


def test_lens_gpt2(gpt2_model):
    results = gpt2_model.lens(TEXT)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "layer_name" in results[0]
    assert "top1_token" in results[0]


def test_lens_distilbert(distilbert_model):
    # FAILURE_MODE_CHECK: no lm_head — lens needs unembedding
    results = distilbert_model.lens(TEXT)
    # May be None if no unembedding detected
    assert results is None or isinstance(results, list)


def test_lens_t5(t5_model):
    # FAILURE_MODE_CHECK: T5 decoder layers, _find_final_norm
    results = t5_model.lens(TEXT)
    assert results is None or isinstance(results, list)


def test_lens_vit_none(vit_model):
    result = vit_model.lens("test")
    assert result is None


def test_lens_resnet_none(resnet_model):
    result = resnet_model.lens("test")
    assert result is None


def test_lens_qwen2(qwen2_model):
    # FAILURE_MODE_CHECK: RMSNorm naming
    results = qwen2_model.lens(TEXT)
    assert results is None or isinstance(results, list)


def test_lens_bloom(bloom_model):
    # FAILURE_MODE_CHECK: ln_f naming
    results = bloom_model.lens(TEXT)
    assert results is None or isinstance(results, list)


def test_lens_mamba(mamba_model):
    # FAILURE_MODE_CHECK: has lm_head but no attention — should still project
    results = mamba_model.lens(TEXT)
    assert results is None or isinstance(results, list)


def test_lens_roberta_qa_none(roberta_qa_model):
    # FAILURE_MODE_CHECK: not an LM — should return None
    result = roberta_qa_model.lens(TEXT)
    assert result is None


@slow
def test_lens_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: interleaved recurrent/attention layers
    results = recurrentgemma_model.lens(TEXT)
    assert results is None or isinstance(results, list)


def test_lens_gpt2_position(gpt2_model):
    results = gpt2_model.lens(TEXT, position=-1)
    assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════════
# 10. DLA (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


def test_dla_gpt2(gpt2_model):
    result = gpt2_model.dla(TEXT)
    assert "target_token" in result
    assert "contributions" in result
    assert isinstance(result["contributions"], list)


def test_dla_distilbert(distilbert_model):
    # FAILURE_MODE_CHECK: MaskedLM head structure
    if not distilbert_model.arch_info.is_language_model:
        pytest.skip("Not detected as LM")
    result = distilbert_model.dla(TEXT)
    assert "contributions" in result


def test_dla_t5(t5_model):
    # FAILURE_MODE_CHECK: encoder-decoder decomposition
    if not t5_model.arch_info.is_language_model:
        pytest.skip("Not detected as LM")
    result = t5_model.dla(TEXT)
    assert "contributions" in result


def test_dla_qwen2(qwen2_model):
    # FAILURE_MODE_CHECK: GQA head decomposition
    result = qwen2_model.dla(TEXT)
    assert "contributions" in result
    if "head_contributions" in result:
        assert isinstance(result["head_contributions"], list)


def test_dla_bloom(bloom_model):
    result = bloom_model.dla(TEXT)
    assert "contributions" in result


def test_dla_mamba(mamba_model):
    # FAILURE_MODE_CHECK: no attention submodules — attn decomposition path should be skipped
    result = mamba_model.dla(TEXT)
    assert "contributions" in result


@slow
def test_dla_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: recurrent layers have no heads
    result = recurrentgemma_model.dla(TEXT)
    assert "contributions" in result


# ═══════════════════════════════════════════════════════════════════════════
# 11. STEER
# ═══════════════════════════════════════════════════════════════════════════


def test_steer_gpt2(gpt2_model):
    layer = _first_layer(gpt2_model)
    vec = gpt2_model.steer_vector("happy", "sad", at=layer)
    assert isinstance(vec, torch.Tensor)
    result = gpt2_model.steer(TEXT, vector=vec, at=layer, scale=2.0)
    assert "original_top" in result
    assert "steered_top" in result


def test_steer_distilbert(distilbert_model):
    # FAILURE_MODE_CHECK: encoder-only steering
    layer = _first_layer(distilbert_model)
    vec = distilbert_model.steer_vector("happy", "sad", at=layer)
    assert isinstance(vec, torch.Tensor)
    result = distilbert_model.steer(TEXT, vector=vec, at=layer)
    assert "original_top" in result or "steered_top" in result


def test_steer_t5(t5_model):
    # FAILURE_MODE_CHECK: encoder-decoder mismatch
    layer = _first_layer(t5_model)
    vec = t5_model.steer_vector("happy", "sad", at=layer)
    assert isinstance(vec, torch.Tensor)


def test_steer_qwen2(qwen2_model):
    layer = _first_layer(qwen2_model)
    vec = qwen2_model.steer_vector("happy", "sad", at=layer)
    assert isinstance(vec, torch.Tensor)
    result = qwen2_model.steer(TEXT, vector=vec, at=layer)
    assert "original_top" in result


def test_steer_bloom(bloom_model):
    layer = _first_layer(bloom_model)
    vec = bloom_model.steer_vector("happy", "sad", at=layer)
    assert isinstance(vec, torch.Tensor)


def test_steer_mamba(mamba_model):
    # FAILURE_MODE_CHECK: steering a recurrent state
    layer = _first_layer(mamba_model)
    vec = mamba_model.steer_vector("happy", "sad", at=layer)
    assert isinstance(vec, torch.Tensor)


@slow
def test_steer_recurrentgemma(recurrentgemma_model):
    layer = _first_layer(recurrentgemma_model)
    vec = recurrentgemma_model.steer_vector("happy", "sad", at=layer)
    assert isinstance(vec, torch.Tensor)


# ═══════════════════════════════════════════════════════════════════════════
# 12. PATCH
# ═══════════════════════════════════════════════════════════════════════════


def test_patch_gpt2(gpt2_model):
    result = gpt2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(gpt2_model))
    assert "effect" in result
    assert isinstance(result["effect"], float)


def test_patch_distilbert(distilbert_model):
    # FAILURE_MODE_CHECK: prepare_pair padding for encoder-only
    result = distilbert_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(distilbert_model))
    assert "effect" in result


def test_patch_t5(t5_model):
    result = t5_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(t5_model))
    assert "effect" in result


def test_patch_qwen2(qwen2_model):
    # FAILURE_MODE_CHECK: GQA head patching
    result = qwen2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(qwen2_model))
    assert "effect" in result


def test_patch_bloom(bloom_model):
    result = bloom_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(bloom_model))
    assert "effect" in result


@slow
def test_patch_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: recurrent state across positions
    result = recurrentgemma_model.patch(
        PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(recurrentgemma_model),
    )
    assert "effect" in result


# ═══════════════════════════════════════════════════════════════════════════
# 13. TRACE
# ═══════════════════════════════════════════════════════════════════════════


def test_trace_gpt2_module(gpt2_model):
    results = gpt2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3, mode="module")
    assert isinstance(results, list)
    assert len(results) > 0


def test_trace_gpt2_position(gpt2_model):
    result = gpt2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, mode="position")
    assert isinstance(result, dict)
    assert "effects" in result


def test_trace_distilbert(distilbert_model):
    results = distilbert_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
    assert isinstance(results, list)


def test_trace_t5(t5_model):
    # FAILURE_MODE_CHECK: T5 double-stack
    results = t5_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
    assert isinstance(results, list) or isinstance(results, dict)


def test_trace_qwen2(qwen2_model):
    results = qwen2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
    assert isinstance(results, list) or isinstance(results, dict)


def test_trace_bloom(bloom_model):
    results = bloom_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
    assert isinstance(results, list) or isinstance(results, dict)


def test_trace_bloom_position(bloom_model):
    result = bloom_model.trace(PAIR_CLEAN, PAIR_CORRUPT, mode="position")
    assert isinstance(result, dict)


@slow
def test_trace_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: mixed layer types
    results = recurrentgemma_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
    assert isinstance(results, list) or isinstance(results, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 14. DECOMPOSE (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


def test_decompose_gpt2(gpt2_model):
    result = gpt2_model.decompose(TEXT)
    assert "components" in result
    assert isinstance(result["components"], list)
    assert "residual" in result


def test_decompose_distilbert(distilbert_model):
    result = distilbert_model.decompose(TEXT)
    assert "components" in result


def test_decompose_t5(t5_model):
    # FAILURE_MODE_CHECK: encoder+decoder layers
    result = t5_model.decompose(TEXT)
    assert "components" in result


def test_decompose_qwen2(qwen2_model):
    result = qwen2_model.decompose(TEXT)
    assert "components" in result


def test_decompose_bloom(bloom_model):
    result = bloom_model.decompose(TEXT)
    assert "components" in result


def test_decompose_mamba(mamba_model):
    # FAILURE_MODE_CHECK: no attn/MLP split
    result = mamba_model.decompose(TEXT)
    assert "components" in result


@slow
def test_decompose_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: alternating recurrent/attention blocks
    result = recurrentgemma_model.decompose(TEXT)
    assert "components" in result


# ═══════════════════════════════════════════════════════════════════════════
# 15. OV / QK SCORES (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


def test_ov_scores_gpt2(gpt2_model):
    result = gpt2_model.ov_scores(layer=0)
    assert "heads" in result
    assert isinstance(result["heads"], list)


def test_qk_scores_gpt2(gpt2_model):
    result = gpt2_model.qk_scores(layer=0)
    assert "heads" in result


def test_ov_scores_distilbert(distilbert_model):
    # FAILURE_MODE_CHECK: q_lin/k_lin/v_lin naming
    result = distilbert_model.ov_scores(layer=0)
    assert "heads" in result


def test_qk_scores_distilbert(distilbert_model):
    result = distilbert_model.qk_scores(layer=0)
    assert "heads" in result


def test_ov_scores_vit(vit_model):
    # FAILURE_MODE_CHECK: query/key/value naming
    result = vit_model.ov_scores(layer=0)
    assert "heads" in result


def test_ov_scores_qwen2(qwen2_model):
    # FAILURE_MODE_CHECK: GQA — W_V and W_O shape mismatch due to fewer KV heads
    result = qwen2_model.ov_scores(layer=0)
    assert "heads" in result


def test_qk_scores_qwen2(qwen2_model):
    result = qwen2_model.qk_scores(layer=0)
    assert "heads" in result


def test_ov_scores_bloom(bloom_model):
    # FAILURE_MODE_CHECK: fused query_key_value
    result = bloom_model.ov_scores(layer=0)
    assert "heads" in result


@slow
def test_ov_scores_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: should only work on attention layers
    result = recurrentgemma_model.ov_scores(layer=0)
    assert "heads" in result


# ═══════════════════════════════════════════════════════════════════════════
# 16. COMPOSITION (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


def test_composition_gpt2(gpt2_model):
    result = gpt2_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    assert "scores" in result


def test_composition_distilbert(distilbert_model):
    result = distilbert_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    assert "scores" in result


def test_composition_qwen2(qwen2_model):
    # FAILURE_MODE_CHECK: GQA shape mismatch in composition
    result = qwen2_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    assert "scores" in result


@slow
def test_composition_recurrentgemma(recurrentgemma_model):
    # FAILURE_MODE_CHECK: undefined if src/dst is recurrent
    layers = recurrentgemma_model.arch_info.layer_names
    if len(layers) < 2:
        pytest.skip("Not enough layers")
    result = recurrentgemma_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    assert "scores" in result


# ═══════════════════════════════════════════════════════════════════════════
# 17. FIND CIRCUIT (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.timeout(180)
def test_find_circuit_gpt2(gpt2_model):
    result = gpt2_model.find_circuit(PAIR_CLEAN, PAIR_CORRUPT, threshold=0.05)
    assert "circuit" in result
    assert "excluded" in result
    assert isinstance(result["circuit"], list)


# ═══════════════════════════════════════════════════════════════════════════
# 18. SCAN (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


def test_scan_gpt2(gpt2_model):
    result = gpt2_model.scan(TEXT)
    assert isinstance(result, dict)


def test_scan_distilbert(distilbert_model):
    # FAILURE_MODE_CHECK: graceful handling when sub-ops fail
    result = distilbert_model.scan(TEXT)
    assert isinstance(result, dict)


def test_scan_qwen2(qwen2_model):
    result = qwen2_model.scan(TEXT)
    assert isinstance(result, dict)


def test_scan_mamba(mamba_model):
    # FAILURE_MODE_CHECK: multiple sub-ops may partially fail
    result = mamba_model.scan(TEXT)
    assert isinstance(result, dict)


@slow
def test_scan_recurrentgemma(recurrentgemma_model):
    result = recurrentgemma_model.scan(TEXT)
    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 19. REPORT (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


def test_report_gpt2(gpt2_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        result = gpt2_model.report(TEXT, save=path)
        assert "html_path" in result
        assert os.path.exists(path)


def test_report_mamba(mamba_model):
    # FAILURE_MODE_CHECK: cascading partial failures
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        result = mamba_model.report(TEXT, save=path)
        assert isinstance(result, dict)


@slow
def test_report_recurrentgemma(recurrentgemma_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        result = recurrentgemma_model.report(TEXT, save=path)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 20. BATCH (previously untested)
# ═══════════════════════════════════════════════════════════════════════════


def test_batch_gpt2_attention(gpt2_model):
    dataset = [
        {"input_data": "Hello world"},
        {"input_data": "The sky is blue"},
    ]
    result = gpt2_model.batch("attention", dataset)
    assert "results" in result
    assert result["count"] == 2


def test_trace_batch_gpt2(gpt2_model):
    dataset = [
        {"clean": PAIR_CLEAN, "corrupted": PAIR_CORRUPT},
        {"clean": "Paris is in France", "corrupted": "Paris is in Germany"},
    ]
    result = gpt2_model.trace_batch(dataset, top_k=3)
    assert "results" in result
    assert result["count"] == 2


def test_dla_batch_gpt2(gpt2_model):
    texts = [TEXT, TEXT_B]
    result = gpt2_model.dla_batch(texts, top_k=3)
    assert "results" in result
    assert result["count"] == 2


def test_batch_error_handling(gpt2_model):
    dataset = [
        {"input_data": "Hello world"},
        {"input_data": "Another sentence"},
    ]
    result = gpt2_model.batch("attention", dataset)
    assert "errors" in result


# ═══════════════════════════════════════════════════════════════════════════
# 21. PROBE
# ═══════════════════════════════════════════════════════════════════════════


def test_probe_gpt2(gpt2_model):
    texts = ["I love this", "I hate this"] * 10
    labels = [1, 0] * 10
    layer = _first_layer(gpt2_model)
    result = gpt2_model.probe(texts, labels, at=layer)
    assert "accuracy" in result


def test_probe_distilbert(distilbert_model):
    texts = ["good movie", "bad movie"] * 10
    labels = [1, 0] * 10
    layer = _first_layer(distilbert_model)
    result = distilbert_model.probe(texts, labels, at=layer)
    assert "accuracy" in result


def test_probe_roberta_qa(roberta_qa_model):
    texts = ["positive review", "negative review"] * 10
    labels = [1, 0] * 10
    layer = _first_layer(roberta_qa_model)
    result = roberta_qa_model.probe(texts, labels, at=layer)
    assert "accuracy" in result


# ═══════════════════════════════════════════════════════════════════════════
# 22. DIFF
# ═══════════════════════════════════════════════════════════════════════════


def test_diff_same_model(gpt2_model):
    import interpkit

    result = interpkit.diff(gpt2_model, gpt2_model, TEXT)
    assert "results" in result
    for r in result["results"]:
        assert r["distance"] == pytest.approx(0.0, abs=1e-4)


def test_diff_cross_arch(vit_model, resnet_model, test_image_path):
    import interpkit

    # FAILURE_MODE_CHECK: mismatched module names
    result = interpkit.diff(vit_model, resnet_model, test_image_path)
    assert "results" in result or "skipped_a" in result


def test_diff_gpt2_bloom(gpt2_model, bloom_model):
    import interpkit

    # FAILURE_MODE_CHECK: same task, different architecture
    result = interpkit.diff(gpt2_model, bloom_model, TEXT)
    assert isinstance(result, dict)
