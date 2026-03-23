"""Stress tests, unicode inputs, multi-class probes, distant composition, and cache state.

Covers:
  - Long sequences (128 tokens, near-max-length)
  - Non-English / unicode inputs (French, Japanese, Chinese, mixed)
  - Probe with 3+ classes and imbalanced data
  - Composition across distant layers
  - Cache state correctness
"""

from __future__ import annotations

import pytest
import torch

TEXT = "The capital of France is"
LONG_SENTENCE = "The quick brown fox jumps over the lazy dog. "


def _first_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[0]


def _last_layer_idx(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return len(layers) - 1


def _has_two_attn_layers(model):
    infos = model.arch_info.layer_infos
    return sum(1 for li in infos if li.attn_path is not None) >= 2


def _make_long_text(approx_tokens: int = 128) -> str:
    """Repeat a sentence to produce roughly `approx_tokens` tokens."""
    repetitions = approx_tokens // 10 + 2
    return (LONG_SENTENCE * repetitions).strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Long-sequence tests (128 tokens)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.timeout(60)
def test_activations_128_tokens_gpt2(gpt2_model):
    text = _make_long_text(128)
    layer = _first_layer(gpt2_model)
    result = gpt2_model.activations(text, at=layer)
    assert isinstance(result, torch.Tensor)
    seq_dim = result.shape[-2] if result.dim() >= 2 else result.shape[0]
    assert seq_dim >= 100, f"Expected seq_len >= 100, got {seq_dim}"


@pytest.mark.timeout(60)
def test_attention_128_tokens_smollm(smollm_model):
    text = _make_long_text(128)
    results = smollm_model.attention(text)
    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0
    weights = results[0]["weights"]
    seq_len = weights.shape[-1]
    assert seq_len >= 100, f"Expected attention matrix dim >= 100, got {seq_len}"


@pytest.mark.timeout(60)
def test_lens_128_tokens_pythia(pythia_model):
    text = _make_long_text(128)
    results = pythia_model.lens(text)
    if results is None:
        pytest.skip("Lens returned None")
    n_layers = len(pythia_model.arch_info.layer_names)
    assert len(results) == n_layers, (
        f"Expected {n_layers} layer entries, got {len(results)}"
    )


@pytest.mark.timeout(120)
def test_attribute_128_tokens_bart(bart_model):
    text = _make_long_text(128)
    result = bart_model.attribute(text, method="gradient")
    assert "tokens" in result
    assert len(result["tokens"]) >= 50, (
        f"Expected many tokens, got {len(result['tokens'])}"
    )


@pytest.mark.timeout(120)
def test_long_input_truncation_gpt2(gpt2_model):
    text = _make_long_text(2200)
    layer = _first_layer(gpt2_model)
    try:
        result = gpt2_model.activations(text, at=layer)
        assert isinstance(result, torch.Tensor)
    except (ValueError, RuntimeError, IndexError):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Unicode / non-English inputs
# ═══════════════════════════════════════════════════════════════════════════


def test_activations_french_gpt2(gpt2_model):
    result = gpt2_model.activations(
        "La capitale de la France est", at=_first_layer(gpt2_model),
    )
    assert isinstance(result, torch.Tensor)
    assert result.dim() >= 2


def test_attention_japanese_smollm(smollm_model):
    results = smollm_model.attention("日本の首都は")
    assert results is None or isinstance(results, list)
    if results is not None:
        assert len(results) > 0


def test_attribute_chinese_bart(bart_model):
    result = bart_model.attribute("法国的首都是", method="gradient")
    assert "tokens" in result
    assert len(result["tokens"]) > 0


def test_lens_mixed_unicode_pythia(pythia_model):
    results = pythia_model.lens("The capital is 巴黎 (Paris)")
    assert results is None or isinstance(results, list)
    if results is not None:
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Probe: multi-class and imbalanced
# ═══════════════════════════════════════════════════════════════════════════


def test_probe_three_classes_gpt_neo(gpt_neo_model):
    texts = (
        ["great movie", "wonderful film", "loved it"] * 4
        + ["okay movie", "average film", "it was fine"] * 3
        + ["terrible movie", "hated it", "worst ever"] * 3
    )
    labels = [0] * 12 + [1] * 9 + [2] * 9
    result = gpt_neo_model.probe(texts, labels, at=_first_layer(gpt_neo_model))
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_probe_imbalanced_electra(electra_model):
    texts_pos = ["good news", "positive review", "great product"] * 5
    texts_neg = ["bad news", "negative review", "terrible product", "awful", "worst"]
    texts = texts_pos + texts_neg
    labels = [0] * 15 + [1] * 5
    result = electra_model.probe(texts, labels, at=_first_layer(electra_model))
    assert "accuracy" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Composition across distant layers
# ═══════════════════════════════════════════════════════════════════════════


def test_composition_distant_layers_gpt2(gpt2_model):
    last = _last_layer_idx(gpt2_model)
    if last < 2:
        pytest.skip("Not enough layers")
    result = gpt2_model.composition(src_layer=0, dst_layer=last, comp_type="q")
    assert "scores" in result


def test_composition_distant_layers_pythia(pythia_model):
    if not _has_two_attn_layers(pythia_model):
        pytest.skip("Not enough attention layers")
    last = _last_layer_idx(pythia_model)
    if last < 2:
        pytest.skip("Not enough layers")
    result = pythia_model.composition(src_layer=0, dst_layer=last, comp_type="q")
    assert "scores" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Cache state correctness
# ═══════════════════════════════════════════════════════════════════════════


def test_cache_then_lens_then_clear_gpt2(gpt2_model):
    gpt2_model.cache(TEXT)
    results_cached = gpt2_model.lens(TEXT)
    gpt2_model.clear_cache()
    results_uncached = gpt2_model.lens(TEXT)

    if results_cached is None or results_uncached is None:
        pytest.skip("Lens returned None")

    assert len(results_cached) == len(results_uncached)
    for r1, r2 in zip(results_cached, results_uncached):
        assert r1["top1_token"] == r2["top1_token"], (
            f"Cache should not change lens semantics: "
            f"'{r1['top1_token']}' != '{r2['top1_token']}' at {r1.get('layer_name')}"
        )


def test_cache_invalidation_on_new_input_gpt2(gpt2_model):
    text_a = "The capital of France is"
    text_b = "Machine learning is a branch of"

    gpt2_model.cache(text_a)
    layer = _first_layer(gpt2_model)
    act_b = gpt2_model.activations(text_b, at=layer)

    gpt2_model.clear_cache()
    act_b_direct = gpt2_model.activations(text_b, at=layer)

    assert torch.allclose(act_b.float(), act_b_direct.float(), atol=1e-4), (
        "Activations for text_b should be the same whether or not text_a was cached"
    )
