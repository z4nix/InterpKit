"""Holistic test suite — every InterpKit operation on 4 new architectures.

Tests every public operation of InterpKit across 4 models that cover
architecture families not exercised in the earlier test rounds.

Models:
  - facebook/bart-base          (encoder-decoder, BART family)
  - EleutherAI/gpt-neo-125m     (decoder-only, local/global attention)
  - google/electra-small-discriminator (encoder-only, discriminator head)
  - HuggingFaceTB/SmolLM-135M   (decoder-only, Llama architecture, GQA)
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

slow = pytest.mark.timeout(300)

TEXT = "The capital of France is"
TEXT_B = "The capital of Germany is"
PAIR_CLEAN = "The Eiffel Tower is in Paris"
PAIR_CORRUPT = "The Eiffel Tower is in Rome"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
#  BART — encoder-decoder (BART family, different from T5)
# ═══════════════════════════════════════════════════════════════════════════


class TestBART:
    """facebook/bart-base — encoder-decoder with q_proj/k_proj/v_proj/out_proj."""

    def test_discover(self, bart_model):
        arch = bart_model.arch_info
        assert arch.layer_names
        assert arch.hidden_size is not None
        assert arch.num_attention_heads is not None

    def test_inspect(self, bart_model):
        bart_model.inspect()

    def test_activations(self, bart_model):
        result = bart_model.activations(TEXT, at=_first_layer(bart_model))
        assert isinstance(result, torch.Tensor)
        assert result.dim() >= 2

    def test_cache(self, bart_model):
        bart_model.cache(TEXT)
        assert bart_model.cached
        bart_model.clear_cache()
        assert not bart_model.cached

    def test_head_activations(self, bart_model):
        at = _first_attn(bart_model)
        result = bart_model.head_activations(TEXT, at=at)
        assert "head_acts" in result
        assert result["num_heads"] > 0

    def test_attention(self, bart_model):
        results = bart_model.attention(TEXT)
        assert results is None or isinstance(results, list)

    def test_attention_layer(self, bart_model):
        results = bart_model.attention(TEXT, layer=0)
        assert results is None or isinstance(results, list)

    def test_ablate_zero(self, bart_model):
        result = bart_model.ablate(TEXT, at=_first_layer(bart_model), method="zero")
        assert "effect" in result

    def test_ablate_mean(self, bart_model):
        result = bart_model.ablate(TEXT, at=_first_layer(bart_model), method="mean")
        assert result["method"] == "mean"

    def test_attribute_gradient(self, bart_model):
        result = bart_model.attribute(TEXT, method="gradient")
        assert "tokens" in result

    def test_attribute_ig(self, bart_model):
        result = bart_model.attribute(TEXT, method="integrated_gradients", n_steps=5)
        assert "tokens" in result

    def test_lens(self, bart_model):
        results = bart_model.lens(TEXT)
        assert results is None or isinstance(results, list)

    def test_dla(self, bart_model):
        if not bart_model.arch_info.is_language_model:
            pytest.skip("Not detected as LM")
        result = bart_model.dla(TEXT)
        assert "contributions" in result

    def test_steer(self, bart_model):
        layer = _first_layer(bart_model)
        vec = bart_model.steer_vector("happy", "sad", at=layer)
        assert isinstance(vec, torch.Tensor)
        result = bart_model.steer(TEXT, vector=vec, at=layer)
        assert "original_top" in result or "steered_top" in result

    def test_patch(self, bart_model):
        result = bart_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(bart_model))
        assert "effect" in result

    def test_trace(self, bart_model):
        results = bart_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
        assert isinstance(results, (list, dict))

    def test_decompose(self, bart_model):
        result = bart_model.decompose(TEXT)
        assert "components" in result
        assert isinstance(result["components"], list)

    def test_ov_scores(self, bart_model):
        result = bart_model.ov_scores(layer=0)
        assert "heads" in result
        assert len(result["heads"]) > 0

    def test_qk_scores(self, bart_model):
        result = bart_model.qk_scores(layer=0)
        assert "heads" in result

    def test_composition(self, bart_model):
        if not _has_two_attn_layers(bart_model):
            pytest.skip("Not enough attention layers")
        result = bart_model.composition(src_layer=0, dst_layer=1, comp_type="q")
        assert "scores" in result

    def test_scan(self, bart_model):
        result = bart_model.scan(TEXT)
        assert isinstance(result, dict)

    def test_probe(self, bart_model):
        texts = ["good news", "bad news"] * 10
        labels = [1, 0] * 10
        result = bart_model.probe(texts, labels, at=_first_layer(bart_model))
        assert "accuracy" in result


# ═══════════════════════════════════════════════════════════════════════════
#  GPT-Neo — decoder-only with local/global attention
#  Maximum coverage: every single InterpKit operation is tested here.
# ═══════════════════════════════════════════════════════════════════════════


class TestGPTNeo:
    """EleutherAI/gpt-neo-125m — local/global alternating attention."""

    def test_discover(self, gpt_neo_model):
        arch = gpt_neo_model.arch_info
        assert arch.is_language_model
        assert len(arch.layer_names) > 0
        assert arch.num_attention_heads is not None
        assert arch.hidden_size is not None
        assert arch.unembedding_name is not None

    def test_inspect(self, gpt_neo_model):
        gpt_neo_model.inspect()

    def test_activations(self, gpt_neo_model):
        result = gpt_neo_model.activations(TEXT, at=_first_layer(gpt_neo_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_multi(self, gpt_neo_model):
        layers = gpt_neo_model.arch_info.layer_names[:2]
        result = gpt_neo_model.activations(TEXT, at=layers)
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_cache(self, gpt_neo_model):
        gpt_neo_model.cache(TEXT)
        assert gpt_neo_model.cached
        gpt_neo_model.clear_cache()
        assert not gpt_neo_model.cached

    def test_head_activations(self, gpt_neo_model):
        at = _first_attn(gpt_neo_model)
        result = gpt_neo_model.head_activations(TEXT, at=at)
        assert "head_acts" in result
        assert result["num_heads"] > 0

    def test_attention(self, gpt_neo_model):
        results = gpt_neo_model.attention(TEXT)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "layer" in results[0]
        assert "head" in results[0]

    def test_attention_layer_filter(self, gpt_neo_model):
        results = gpt_neo_model.attention(TEXT, layer=0)
        assert results is None or all(r["layer"] == 0 for r in results)

    def test_attention_head_filter(self, gpt_neo_model):
        results = gpt_neo_model.attention(TEXT, layer=0, head=0)
        if results is not None:
            assert len(results) == 1

    def test_ablate_zero(self, gpt_neo_model):
        result = gpt_neo_model.ablate(TEXT, at=_first_layer(gpt_neo_model), method="zero")
        assert "effect" in result
        assert isinstance(result["effect"], float)

    def test_ablate_mean(self, gpt_neo_model):
        result = gpt_neo_model.ablate(TEXT, at=_first_layer(gpt_neo_model), method="mean")
        assert result["method"] == "mean"

    def test_attribute_gradient(self, gpt_neo_model):
        result = gpt_neo_model.attribute(TEXT, method="gradient")
        assert "tokens" in result
        assert "scores" in result

    def test_attribute_ig(self, gpt_neo_model):
        result = gpt_neo_model.attribute(TEXT, method="integrated_gradients", n_steps=5)
        assert "tokens" in result

    def test_attribute_gxi(self, gpt_neo_model):
        result = gpt_neo_model.attribute(TEXT, method="gradient_x_input")
        assert "tokens" in result

    def test_lens(self, gpt_neo_model):
        results = gpt_neo_model.lens(TEXT)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "layer_name" in results[0]
        assert "top1_token" in results[0]

    def test_lens_position(self, gpt_neo_model):
        results = gpt_neo_model.lens(TEXT, position=-1)
        assert isinstance(results, list)

    def test_dla(self, gpt_neo_model):
        result = gpt_neo_model.dla(TEXT)
        assert "target_token" in result
        assert "contributions" in result
        assert isinstance(result["contributions"], list)

    def test_steer(self, gpt_neo_model):
        layer = _first_layer(gpt_neo_model)
        vec = gpt_neo_model.steer_vector("happy", "sad", at=layer)
        assert isinstance(vec, torch.Tensor)
        result = gpt_neo_model.steer(TEXT, vector=vec, at=layer, scale=2.0)
        assert "original_top" in result
        assert "steered_top" in result

    def test_patch(self, gpt_neo_model):
        result = gpt_neo_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(gpt_neo_model))
        assert "effect" in result
        assert isinstance(result["effect"], float)

    def test_trace_module(self, gpt_neo_model):
        results = gpt_neo_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3, mode="module")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_trace_position(self, gpt_neo_model):
        result = gpt_neo_model.trace(PAIR_CLEAN, PAIR_CORRUPT, mode="position")
        assert isinstance(result, dict)
        assert "effects" in result

    def test_decompose(self, gpt_neo_model):
        result = gpt_neo_model.decompose(TEXT)
        assert "components" in result
        assert "residual" in result

    def test_ov_scores(self, gpt_neo_model):
        result = gpt_neo_model.ov_scores(layer=0)
        assert "heads" in result
        assert isinstance(result["heads"], list)

    def test_qk_scores(self, gpt_neo_model):
        result = gpt_neo_model.qk_scores(layer=0)
        assert "heads" in result

    def test_composition(self, gpt_neo_model):
        if not _has_two_attn_layers(gpt_neo_model):
            pytest.skip("Not enough attention layers")
        result = gpt_neo_model.composition(src_layer=0, dst_layer=1, comp_type="q")
        assert "scores" in result
        assert result["scores"].shape[0] > 0

    @pytest.mark.timeout(180)
    def test_find_circuit(self, gpt_neo_model):
        result = gpt_neo_model.find_circuit(PAIR_CLEAN, PAIR_CORRUPT, threshold=0.05)
        assert "circuit" in result
        assert "excluded" in result
        assert isinstance(result["circuit"], list)

    def test_scan(self, gpt_neo_model):
        result = gpt_neo_model.scan(TEXT)
        assert isinstance(result, dict)

    def test_report(self, gpt_neo_model):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.html")
            result = gpt_neo_model.report(TEXT, save=path)
            assert "html_path" in result
            assert os.path.exists(path)

    def test_batch_attention(self, gpt_neo_model):
        dataset = [
            {"input_data": "Hello world"},
            {"input_data": "The sky is blue"},
        ]
        result = gpt_neo_model.batch("attention", dataset)
        assert "results" in result
        assert result["count"] == 2

    def test_trace_batch(self, gpt_neo_model):
        dataset = [
            {"clean": PAIR_CLEAN, "corrupted": PAIR_CORRUPT},
            {"clean": "Paris is in France", "corrupted": "Paris is in Germany"},
        ]
        result = gpt_neo_model.trace_batch(dataset, top_k=3)
        assert "results" in result
        assert result["count"] == 2

    def test_dla_batch(self, gpt_neo_model):
        texts = [TEXT, TEXT_B]
        result = gpt_neo_model.dla_batch(texts, top_k=3)
        assert "results" in result
        assert result["count"] == 2

    def test_probe(self, gpt_neo_model):
        texts = ["I love this", "I hate this"] * 10
        labels = [1, 0] * 10
        result = gpt_neo_model.probe(texts, labels, at=_first_layer(gpt_neo_model))
        assert "accuracy" in result

    def test_diff_vs_gpt2(self, gpt_neo_model, gpt2_model):
        import interpkit

        result = interpkit.diff(gpt_neo_model, gpt2_model, TEXT)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  ELECTRA — encoder-only discriminator (not LM, not MLM, not QA)
# ═══════════════════════════════════════════════════════════════════════════


class TestELECTRA:
    """google/electra-small-discriminator — encoder-only with discriminator head."""

    def test_discover(self, electra_model):
        arch = electra_model.arch_info
        assert arch.layer_names
        assert arch.num_attention_heads is not None
        assert arch.hidden_size is not None

    def test_discover_not_lm(self, electra_model):
        arch = electra_model.arch_info
        if arch.is_language_model:
            pytest.skip("Loaded as LM (auto-class cascade picked MaskedLM)")

    def test_inspect(self, electra_model):
        electra_model.inspect()

    def test_activations(self, electra_model):
        result = electra_model.activations(TEXT, at=_first_layer(electra_model))
        assert isinstance(result, torch.Tensor)

    def test_cache(self, electra_model):
        electra_model.cache(TEXT)
        assert electra_model.cached
        electra_model.clear_cache()

    def test_head_activations(self, electra_model):
        at = _first_attn(electra_model)
        result = electra_model.head_activations(TEXT, at=at)
        assert "head_acts" in result
        assert result["num_heads"] > 0

    def test_attention(self, electra_model):
        results = electra_model.attention(TEXT)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_ablate(self, electra_model):
        result = electra_model.ablate(TEXT, at=_first_layer(electra_model))
        assert "effect" in result

    def test_attribute_gradient(self, electra_model):
        result = electra_model.attribute(TEXT, method="gradient")
        assert "tokens" in result

    def test_lens_none_if_not_lm(self, electra_model):
        results = electra_model.lens(TEXT)
        if not electra_model.arch_info.is_language_model:
            assert results is None
        else:
            assert isinstance(results, list)

    def test_steer_vector(self, electra_model):
        layer = _first_layer(electra_model)
        vec = electra_model.steer_vector("happy", "sad", at=layer)
        assert isinstance(vec, torch.Tensor)

    def test_patch(self, electra_model):
        result = electra_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(electra_model))
        assert "effect" in result

    def test_trace(self, electra_model):
        results = electra_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
        assert isinstance(results, (list, dict))

    def test_decompose(self, electra_model):
        result = electra_model.decompose(TEXT)
        assert "components" in result

    def test_ov_scores(self, electra_model):
        result = electra_model.ov_scores(layer=0)
        assert "heads" in result

    def test_qk_scores(self, electra_model):
        result = electra_model.qk_scores(layer=0)
        assert "heads" in result

    def test_composition(self, electra_model):
        if not _has_two_attn_layers(electra_model):
            pytest.skip("Not enough attention layers")
        result = electra_model.composition(src_layer=0, dst_layer=1, comp_type="q")
        assert "scores" in result

    def test_probe(self, electra_model):
        texts = ["positive review", "negative review"] * 10
        labels = [1, 0] * 10
        result = electra_model.probe(texts, labels, at=_first_layer(electra_model))
        assert "accuracy" in result


# ═══════════════════════════════════════════════════════════════════════════
#  SmolLM — Llama architecture with GQA
# ═══════════════════════════════════════════════════════════════════════════


class TestSmolLM:
    """HuggingFaceTB/SmolLM-135M — LlamaForCausalLM with GQA."""

    def test_discover(self, smollm_model):
        arch = smollm_model.arch_info
        assert arch.is_language_model
        assert arch.layer_names
        assert arch.num_attention_heads is not None
        assert arch.hidden_size is not None

    def test_discover_gqa(self, smollm_model):
        arch = smollm_model.arch_info
        if arch.num_key_value_heads is not None:
            assert arch.num_key_value_heads <= arch.num_attention_heads

    def test_inspect(self, smollm_model):
        smollm_model.inspect()

    def test_activations(self, smollm_model):
        result = smollm_model.activations(TEXT, at=_first_layer(smollm_model))
        assert isinstance(result, torch.Tensor)

    def test_cache(self, smollm_model):
        smollm_model.cache(TEXT)
        assert smollm_model.cached
        smollm_model.clear_cache()

    def test_head_activations(self, smollm_model):
        at = _first_attn(smollm_model)
        result = smollm_model.head_activations(TEXT, at=at)
        assert "head_acts" in result
        assert result["num_heads"] > 0

    def test_attention(self, smollm_model):
        results = smollm_model.attention(TEXT)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_ablate_zero(self, smollm_model):
        result = smollm_model.ablate(TEXT, at=_first_layer(smollm_model), method="zero")
        assert "effect" in result

    def test_ablate_mean(self, smollm_model):
        result = smollm_model.ablate(TEXT, at=_first_layer(smollm_model), method="mean")
        assert result["method"] == "mean"

    def test_attribute_gradient(self, smollm_model):
        result = smollm_model.attribute(TEXT, method="gradient")
        assert "tokens" in result

    def test_attribute_ig(self, smollm_model):
        result = smollm_model.attribute(TEXT, method="integrated_gradients", n_steps=5)
        assert "tokens" in result

    def test_lens(self, smollm_model):
        results = smollm_model.lens(TEXT)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_dla(self, smollm_model):
        result = smollm_model.dla(TEXT)
        assert "contributions" in result

    def test_steer(self, smollm_model):
        layer = _first_layer(smollm_model)
        vec = smollm_model.steer_vector("happy", "sad", at=layer)
        assert isinstance(vec, torch.Tensor)
        result = smollm_model.steer(TEXT, vector=vec, at=layer)
        assert "original_top" in result

    def test_patch(self, smollm_model):
        result = smollm_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(smollm_model))
        assert "effect" in result

    def test_trace(self, smollm_model):
        results = smollm_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
        assert isinstance(results, (list, dict))

    def test_decompose(self, smollm_model):
        result = smollm_model.decompose(TEXT)
        assert "components" in result
        assert "residual" in result

    def test_ov_scores(self, smollm_model):
        result = smollm_model.ov_scores(layer=0)
        assert "heads" in result
        assert len(result["heads"]) > 0

    def test_qk_scores(self, smollm_model):
        result = smollm_model.qk_scores(layer=0)
        assert "heads" in result

    def test_composition(self, smollm_model):
        if not _has_two_attn_layers(smollm_model):
            pytest.skip("Not enough attention layers")
        result = smollm_model.composition(src_layer=0, dst_layer=1, comp_type="q")
        assert "scores" in result

    def test_scan(self, smollm_model):
        result = smollm_model.scan(TEXT)
        assert isinstance(result, dict)

    def test_probe(self, smollm_model):
        texts = ["good movie", "bad movie"] * 10
        labels = [1, 0] * 10
        result = smollm_model.probe(texts, labels, at=_first_layer(smollm_model))
        assert "accuracy" in result

    def test_diff_vs_gpt_neo(self, smollm_model, gpt_neo_model):
        import interpkit

        result = interpkit.diff(smollm_model, gpt_neo_model, TEXT)
        assert isinstance(result, dict)
