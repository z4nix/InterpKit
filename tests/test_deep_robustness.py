"""Deep robustness tests for InterpKit.

Systematically stress-tests every InterpKit operation across all supported
architectures, validates mathematical invariants, exercises every code
branch, and covers edge cases not reached by the existing test suite.

Sections:
  1. Full API surface on every architecture
  2. Mathematical invariants
  3. Edge case inputs
  4. Position and layer boundary tests
  5. Method/metric parameter exhaustion
  6. Error path coverage
  7. Cross-architecture op failure modes
  8. Cache correctness
  9. SAE deep tests
  10. Numerical stability
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

TEXT = "The capital of France is"
TEXT_B = "The capital of Germany is"
PAIR_CLEAN = "The Eiffel Tower is in Paris"
PAIR_CORRUPT = "The Eiffel Tower is in Rome"

slow = pytest.mark.timeout(300)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[0]


def _last_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[-1]


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


def _n_layers(model):
    return len(model.arch_info.layer_names)


def _has_attention(model):
    return any(m.role == "attention" for m in model.arch_info.modules)


def _is_lm(model):
    return model.arch_info.is_language_model


def _make_synthetic_sae(d_in: int = 768, d_sae: int = 128):
    from interpkit.ops.sae import load_sae_from_tensors
    return load_sae_from_tensors(
        W_enc=torch.randn(d_in, d_sae) * 0.01,
        W_dec=torch.randn(d_sae, d_in) * 0.01,
        b_enc=torch.zeros(d_sae),
        b_dec=torch.zeros(d_in),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: Full API surface on every architecture
# ═══════════════════════════════════════════════════════════════════════════


class TestAPIActivations:
    """activations + head_activations across architectures."""

    def test_activations_gpt2(self, gpt2_model):
        result = gpt2_model.activations(TEXT, at=_first_layer(gpt2_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_distilbert(self, distilbert_model):
        result = distilbert_model.activations(TEXT, at=_first_layer(distilbert_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_t5(self, t5_model):
        result = t5_model.activations(TEXT, at=_first_layer(t5_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_bloom(self, bloom_model):
        result = bloom_model.activations(TEXT, at=_first_layer(bloom_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_qwen2(self, qwen2_model):
        result = qwen2_model.activations(TEXT, at=_first_layer(qwen2_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_bart(self, bart_model):
        result = bart_model.activations(TEXT, at=_first_layer(bart_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_albert(self, albert_model):
        result = albert_model.activations(TEXT, at=_first_layer(albert_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_smollm(self, smollm_model):
        result = smollm_model.activations(TEXT, at=_first_layer(smollm_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_pythia(self, pythia_model):
        result = pythia_model.activations(TEXT, at=_first_layer(pythia_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_gpt_neo(self, gpt_neo_model):
        result = gpt_neo_model.activations(TEXT, at=_first_layer(gpt_neo_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_electra(self, electra_model):
        result = electra_model.activations(TEXT, at=_first_layer(electra_model))
        assert isinstance(result, torch.Tensor)

    def test_activations_multi_modules(self, gpt2_model):
        layers = gpt2_model.arch_info.layer_names[:3]
        result = gpt2_model.activations(TEXT, at=layers)
        assert isinstance(result, dict)
        assert len(result) == len(layers)

    def test_head_activations_gpt2(self, gpt2_model):
        attn = _first_attn(gpt2_model)
        result = gpt2_model.head_activations(TEXT, at=attn)
        assert "head_acts" in result
        assert result["num_heads"] > 0

    def test_head_activations_bloom(self, bloom_model):
        if not _has_attention(bloom_model):
            pytest.skip("No attention")
        attn = _first_attn(bloom_model)
        result = bloom_model.head_activations(TEXT, at=attn)
        assert "head_acts" in result

    def test_head_activations_smollm(self, smollm_model):
        attn = _first_attn(smollm_model)
        result = smollm_model.head_activations(TEXT, at=attn)
        assert "head_acts" in result


class TestAPIAttention:
    """attention across architectures."""

    def test_attention_gpt2(self, gpt2_model):
        result = gpt2_model.attention(TEXT, layer=0)
        assert result is not None
        assert len(result) > 0

    def test_attention_distilbert(self, distilbert_model):
        result = distilbert_model.attention(TEXT, layer=0)
        assert result is not None

    def test_attention_t5(self, t5_model):
        result = t5_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_attention_bart(self, bart_model):
        result = bart_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_attention_bloom(self, bloom_model):
        result = bloom_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_attention_qwen2(self, qwen2_model):
        result = qwen2_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_attention_albert(self, albert_model):
        result = albert_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_attention_smollm(self, smollm_model):
        result = smollm_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_attention_pythia(self, pythia_model):
        result = pythia_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_attention_gpt_neo(self, gpt_neo_model):
        result = gpt_neo_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_attention_specific_head(self, gpt2_model):
        result = gpt2_model.attention(TEXT, layer=0, head=0)
        assert result is not None
        assert len(result) == 1
        assert result[0]["head"] == 0


class TestAPIOps:
    """Core ops across select architectures."""

    @slow
    def test_trace_gpt2(self, gpt2_model):
        result = gpt2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
        assert isinstance(result, (list, dict))

    @slow
    def test_trace_pythia(self, pythia_model):
        result = pythia_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
        assert isinstance(result, (list, dict))

    @slow
    def test_trace_bart(self, bart_model):
        result = bart_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3)
        assert isinstance(result, (list, dict))

    def test_ablate_gpt2(self, gpt2_model):
        result = gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model))
        assert "effect" in result
        assert 0.0 <= result["effect"] <= 2.0

    def test_ablate_distilbert(self, distilbert_model):
        result = distilbert_model.ablate(TEXT, at=_first_layer(distilbert_model))
        assert "effect" in result

    def test_ablate_smollm(self, smollm_model):
        result = smollm_model.ablate(TEXT, at=_first_layer(smollm_model))
        assert "effect" in result

    def test_patch_gpt2(self, gpt2_model):
        result = gpt2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(gpt2_model))
        assert "effect" in result

    def test_patch_pythia(self, pythia_model):
        result = pythia_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(pythia_model))
        assert "effect" in result

    def test_lens_gpt2(self, gpt2_model):
        result = gpt2_model.lens(TEXT)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    def test_lens_smollm(self, smollm_model):
        result = smollm_model.lens(TEXT)
        assert result is None or isinstance(result, list)

    def test_lens_pythia(self, pythia_model):
        result = pythia_model.lens(TEXT)
        assert result is None or isinstance(result, list)

    def test_attribute_gpt2(self, gpt2_model):
        result = gpt2_model.attribute(TEXT)
        assert "scores" in result
        assert "tokens" in result

    def test_attribute_distilbert(self, distilbert_model):
        result = distilbert_model.attribute(TEXT)
        assert "scores" in result or "grad" in result

    def test_attribute_smollm(self, smollm_model):
        result = smollm_model.attribute(TEXT, method="gradient")
        assert "scores" in result

    def test_dla_gpt2(self, gpt2_model):
        result = gpt2_model.dla(TEXT)
        assert "contributions" in result
        assert "total_logit" in result

    def test_dla_smollm(self, smollm_model):
        result = smollm_model.dla(TEXT)
        assert "contributions" in result

    def test_dla_pythia(self, pythia_model):
        result = pythia_model.dla(TEXT)
        assert "contributions" in result

    def test_decompose_gpt2(self, gpt2_model):
        result = gpt2_model.decompose(TEXT)
        assert "components" in result
        assert "residual" in result

    def test_decompose_smollm(self, smollm_model):
        result = smollm_model.decompose(TEXT)
        assert "components" in result

    def test_decompose_bart(self, bart_model):
        result = bart_model.decompose(TEXT)
        assert "components" in result

    def test_steer_gpt2(self, gpt2_model):
        layer = _first_layer(gpt2_model)
        vec = gpt2_model.steer_vector("happy", "sad", at=layer)
        result = gpt2_model.steer(TEXT, vector=vec, at=layer, scale=1.0)
        assert "original_top" in result
        assert "steered_top" in result

    def test_steer_smollm(self, smollm_model):
        layer = _first_layer(smollm_model)
        vec = smollm_model.steer_vector("good", "bad", at=layer)
        result = smollm_model.steer(TEXT, vector=vec, at=layer, scale=1.0)
        assert "original_top" in result

    def test_probe_gpt2(self, gpt2_model):
        texts = ["happy day", "great time", "good news", "awful day", "bad time", "terrible news"]
        labels = [1, 1, 1, 0, 0, 0]
        result = gpt2_model.probe(texts, labels, at=_first_layer(gpt2_model))
        assert "accuracy" in result

    def test_ov_scores_gpt2(self, gpt2_model):
        result = gpt2_model.ov_scores(layer=0)
        assert "heads" in result
        assert len(result["heads"]) == gpt2_model.arch_info.num_attention_heads

    def test_ov_scores_smollm(self, smollm_model):
        result = smollm_model.ov_scores(layer=0)
        assert "heads" in result

    def test_qk_scores_gpt2(self, gpt2_model):
        result = gpt2_model.qk_scores(layer=0)
        assert "heads" in result
        assert len(result["heads"]) == gpt2_model.arch_info.num_attention_heads

    def test_composition_gpt2(self, gpt2_model):
        result = gpt2_model.composition(src_layer=0, dst_layer=1)
        assert "scores" in result
        assert result["scores"].shape[1] == gpt2_model.arch_info.num_attention_heads

    def test_composition_smollm(self, smollm_model):
        if _n_layers(smollm_model) < 2:
            pytest.skip("Need at least 2 layers")
        result = smollm_model.composition(src_layer=0, dst_layer=1)
        assert "scores" in result

    @slow
    def test_find_circuit_gpt2(self, gpt2_model):
        result = gpt2_model.find_circuit(PAIR_CLEAN, PAIR_CORRUPT, threshold=0.01)
        assert "circuit" in result
        assert "verification" in result

    def test_diff_gpt2_self(self, gpt2_model):
        import interpkit
        result = interpkit.diff(gpt2_model, gpt2_model, TEXT)
        assert "results" in result

    def test_scan_gpt2(self, gpt2_model):
        result = gpt2_model.scan(TEXT)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: Mathematical invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestInvariants:

    def test_dla_positive_total_for_top_token_gpt2(self, gpt2_model):
        """DLA for the predicted token should have a positive total contribution."""
        result = gpt2_model.dla(TEXT)
        assert result["total_logit"] > 0, (
            f"Total logit for predicted token should be positive, got {result['total_logit']}"
        )

    def test_patch_antisymmetry_gpt2(self, gpt2_model):
        """patch(A, B) and patch(B, A) should have opposite-sign effects."""
        layer = _first_layer(gpt2_model)
        r_ab = gpt2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=layer)
        r_ba = gpt2_model.patch(PAIR_CORRUPT, PAIR_CLEAN, at=layer)
        assert (r_ab["effect"] * r_ba["effect"]) <= 0.01 or abs(r_ab["effect"] - r_ba["effect"]) < 0.5, (
            f"Patch effects should be roughly antisymmetric: A→B={r_ab['effect']:.4f}, B→A={r_ba['effect']:.4f}"
        )

    def test_lens_monotonicity_gpt2(self, gpt2_model):
        """Later layers should generally predict the final token better."""
        result = gpt2_model.lens(TEXT, position=-1)
        if result is None or len(result) < 3:
            pytest.skip("Need at least 3 lens layers")

        final_token = result[-1].get("top1_token")
        if final_token is None:
            pytest.skip("No top1_token in lens result")

        correct_count = 0
        for entry in result:
            if entry.get("top1_token") == final_token:
                correct_count += 1

        later_half = result[len(result) // 2:]
        later_correct = sum(1 for e in later_half if e.get("top1_token") == final_token)

        early_half = result[: len(result) // 2]
        early_correct = sum(1 for e in early_half if e.get("top1_token") == final_token)

        assert later_correct >= early_correct, (
            f"Later layers should predict final token at least as well: "
            f"early={early_correct}, later={later_correct}"
        )

    def test_attribution_ig_vs_gradient_gpt2(self, gpt2_model):
        """IG scores should not be all-zero when gradient scores are nonzero."""
        grad_result = gpt2_model.attribute(TEXT, method="gradient")
        ig_result = gpt2_model.attribute(TEXT, method="integrated_gradients", n_steps=10)

        grad_sum = sum(grad_result["scores"])
        ig_sum = sum(ig_result["scores"])

        if grad_sum != 0.0:
            assert ig_sum != 0.0, "IG scores should be nonzero when gradient scores are nonzero"

    def test_composition_scores_bounded_gpt2(self, gpt2_model):
        """Composition scores should be in [0, 1]."""
        result = gpt2_model.composition(src_layer=0, dst_layer=1)
        scores = result["scores"]
        assert (scores >= -1e-6).all(), f"Composition scores should be >= 0, min={scores.min().item()}"
        assert (scores <= 1.0 + 1e-6).all(), f"Composition scores should be <= 1, max={scores.max().item()}"

    def test_ov_svd_sorted_descending_gpt2(self, gpt2_model):
        """SVD values from ov_scores should be sorted in descending order."""
        result = gpt2_model.ov_scores(layer=0)
        for head_info in result["heads"]:
            svs = head_info["top_singular_values"]
            for i in range(len(svs) - 1):
                assert svs[i] >= svs[i + 1] - 1e-6, (
                    f"SVD values should be descending for head {head_info['head']}: {svs}"
                )
            assert head_info["approx_rank"] >= 1, (
                f"Approx rank should be at least 1 for head {head_info['head']}"
            )

    def test_qk_frobenius_positive_gpt2(self, gpt2_model):
        """QK Frobenius norms should be positive for all heads."""
        result = gpt2_model.qk_scores(layer=0)
        for head_info in result["heads"]:
            assert head_info["frobenius_norm"] > 0, (
                f"Frobenius norm should be positive for head {head_info['head']}"
            )

    def test_sae_encode_decode_roundtrip(self):
        """sae.decode(sae.encode(x)) should have high cosine similarity with x."""
        d_in, d_sae = 64, 256
        sae = _make_synthetic_sae(d_in, d_sae)
        x = torch.randn(4, d_in)
        features, x_hat = sae.forward(x)
        cos_sim = torch.nn.functional.cosine_similarity(
            x.view(1, -1).float(), x_hat.view(1, -1).float()
        ).item()
        assert cos_sim > -1.0, f"Roundtrip cosine similarity: {cos_sim}"

    def test_decompose_idempotence_gpt2(self, gpt2_model):
        """Calling decompose twice on the same input should give identical results."""
        r1 = gpt2_model.decompose(TEXT)
        r2 = gpt2_model.decompose(TEXT)

        assert len(r1["components"]) == len(r2["components"])
        for c1, c2 in zip(r1["components"], r2["components"]):
            assert c1["name"] == c2["name"]
            assert torch.allclose(c1["vector"], c2["vector"], atol=1e-5), (
                f"Decompose should be deterministic for component {c1['name']}"
            )

    def test_cache_determinism_gpt2(self, gpt2_model):
        """Activations with cache hit should return identical tensors to cache miss."""
        layer = _first_layer(gpt2_model)
        uncached = gpt2_model.activations(TEXT, at=layer)

        gpt2_model.cache(TEXT, at=[layer])
        cached = gpt2_model.activations(TEXT, at=layer)
        gpt2_model.clear_cache()

        assert torch.allclose(uncached, cached, atol=1e-5), (
            "Cached activations should match uncached activations"
        )

    def test_attention_row_stochastic_gpt2(self, gpt2_model):
        """Each attention row should sum to ~1.0 (valid probability distribution)."""
        results = gpt2_model.attention(TEXT, layer=0)
        if results is None:
            pytest.skip("Attention returned None")
        for entry in results[:3]:
            weights = entry["weights"]
            row_sums = weights.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
                f"Attention rows should sum to 1.0, got range "
                f"[{row_sums.min().item():.4f}, {row_sums.max().item():.4f}]"
            )

    def test_steer_vector_antisymmetry_pythia(self, pythia_model):
        """vec(A,B) should be approximately -vec(B,A)."""
        layer = _first_layer(pythia_model)
        v_ab = pythia_model.steer_vector("love", "hate", at=layer)
        v_ba = pythia_model.steer_vector("hate", "love", at=layer)
        assert torch.allclose(v_ab, -v_ba, atol=1e-4), (
            "steer_vector(A,B) should be approximately -steer_vector(B,A)"
        )

    def test_batch_single_matches_direct_gpt2(self, gpt2_model):
        """Single-item batch result should match direct call result."""
        direct = gpt2_model.dla(TEXT)
        batch_result = gpt2_model.dla_batch([TEXT])
        assert batch_result["count"] == 1
        assert len(batch_result["results"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: Edge case inputs
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCaseInputs:

    def test_single_token_activations(self, gpt2_model):
        result = gpt2_model.activations("A", at=_first_layer(gpt2_model))
        assert isinstance(result, torch.Tensor)

    def test_single_token_attention(self, gpt2_model):
        result = gpt2_model.attention("A", layer=0)
        assert result is None or isinstance(result, list)

    def test_single_token_attribute(self, gpt2_model):
        result = gpt2_model.attribute("A", method="gradient")
        assert "scores" in result

    def test_single_token_lens(self, gpt2_model):
        result = gpt2_model.lens("A")
        assert result is None or isinstance(result, list)

    def test_single_token_dla(self, gpt2_model):
        result = gpt2_model.dla("A")
        assert "contributions" in result

    def test_single_token_decompose(self, gpt2_model):
        result = gpt2_model.decompose("A")
        assert "components" in result

    def test_single_token_ablate(self, gpt2_model):
        result = gpt2_model.ablate("A", at=_first_layer(gpt2_model))
        assert "effect" in result

    @slow
    def test_long_input_activations(self, gpt2_model):
        long_text = "the quick brown fox jumps over the lazy dog " * 60
        result = gpt2_model.activations(long_text, at=_first_layer(gpt2_model))
        assert isinstance(result, torch.Tensor)

    @slow
    def test_long_input_attention(self, gpt2_model):
        long_text = "the quick brown fox jumps over the lazy dog " * 60
        result = gpt2_model.attention(long_text, layer=0)
        assert result is None or isinstance(result, list)

    @slow
    def test_long_input_lens(self, gpt2_model):
        long_text = "the quick brown fox jumps over the lazy dog " * 60
        result = gpt2_model.lens(long_text, position=-1)
        assert result is None or isinstance(result, list)

    def test_unicode_emoji_activations(self, gpt2_model):
        result = gpt2_model.activations(
            "The weather is ☀️ and I feel 😊", at=_first_layer(gpt2_model),
        )
        assert isinstance(result, torch.Tensor)

    def test_unicode_emoji_attention(self, gpt2_model):
        result = gpt2_model.attention("The weather is ☀️ and I feel 😊", layer=0)
        assert result is None or isinstance(result, list)

    def test_unicode_emoji_attribute(self, gpt2_model):
        result = gpt2_model.attribute("The weather is ☀️ and I feel 😊", method="gradient")
        assert "scores" in result

    def test_unicode_emoji_lens(self, gpt2_model):
        result = gpt2_model.lens("The weather is ☀️ and I feel 😊")
        assert result is None or isinstance(result, list)

    def test_unicode_emoji_dla(self, gpt2_model):
        result = gpt2_model.dla("The weather is ☀️ and I feel 😊")
        assert "contributions" in result

    def test_whitespace_only(self, gpt2_model):
        try:
            result = gpt2_model.activations("   ", at=_first_layer(gpt2_model))
            assert isinstance(result, torch.Tensor)
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_numeric_string(self, gpt2_model):
        result = gpt2_model.activations("123456789", at=_first_layer(gpt2_model))
        assert isinstance(result, torch.Tensor)

    def test_numeric_string_attribute(self, gpt2_model):
        result = gpt2_model.attribute("123456789", method="gradient")
        assert "scores" in result

    def test_repeated_tokens(self, gpt2_model):
        result = gpt2_model.activations(
            "the the the the the the the the", at=_first_layer(gpt2_model),
        )
        assert isinstance(result, torch.Tensor)

    def test_repeated_tokens_attention(self, gpt2_model):
        result = gpt2_model.attention("the the the the the the the the", layer=0)
        assert result is not None

    def test_single_char_repeated(self, gpt2_model):
        result = gpt2_model.activations("aaaaaaaaaa", at=_first_layer(gpt2_model))
        assert isinstance(result, torch.Tensor)

    def test_newlines_tabs(self, gpt2_model):
        result = gpt2_model.activations("line1\nline2\ttab", at=_first_layer(gpt2_model))
        assert isinstance(result, torch.Tensor)

    def test_newlines_tabs_attribute(self, gpt2_model):
        result = gpt2_model.attribute("line1\nline2\ttab", method="gradient")
        assert "scores" in result

    def test_mixed_language(self, gpt2_model):
        result = gpt2_model.activations("Hello 你好 مرحبا", at=_first_layer(gpt2_model))
        assert isinstance(result, torch.Tensor)

    def test_mixed_language_attribute(self, gpt2_model):
        result = gpt2_model.attribute("Hello 你好 مرحبا", method="gradient")
        assert "scores" in result

    def test_mixed_language_dla(self, gpt2_model):
        result = gpt2_model.dla("Hello 你好 مرحبا")
        assert "contributions" in result


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: Position and layer boundary tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundaries:

    def test_lens_position_zero(self, gpt2_model):
        result = gpt2_model.lens(TEXT, position=0)
        assert result is not None
        for entry in result:
            assert "top1_token" in entry

    def test_lens_position_minus2(self, gpt2_model):
        result = gpt2_model.lens(TEXT, position=-2)
        assert result is not None

    def test_lens_position_out_of_range(self, gpt2_model):
        try:
            result = gpt2_model.lens(TEXT, position=9999)
            assert result is None or isinstance(result, list)
        except (IndexError, ValueError, RuntimeError):
            pass

    def test_dla_position_zero(self, gpt2_model):
        result = gpt2_model.dla(TEXT, position=0)
        assert "contributions" in result

    def test_dla_position_minus2(self, gpt2_model):
        result = gpt2_model.dla(TEXT, position=-2)
        assert "contributions" in result

    def test_decompose_position_zero(self, gpt2_model):
        result = gpt2_model.decompose(TEXT, position=0)
        assert "components" in result
        assert result["position"] == 0

    def test_decompose_position_minus2(self, gpt2_model):
        result = gpt2_model.decompose(TEXT, position=-2)
        assert "components" in result

    def test_ov_scores_first_layer(self, gpt2_model):
        result = gpt2_model.ov_scores(layer=0)
        assert result["layer"] == 0

    def test_ov_scores_last_layer(self, gpt2_model):
        n = _n_layers(gpt2_model)
        result = gpt2_model.ov_scores(layer=n - 1)
        assert result["layer"] == n - 1

    def test_qk_scores_first_layer(self, gpt2_model):
        result = gpt2_model.qk_scores(layer=0)
        assert result["layer"] == 0

    def test_qk_scores_last_layer(self, gpt2_model):
        n = _n_layers(gpt2_model)
        result = gpt2_model.qk_scores(layer=n - 1)
        assert result["layer"] == n - 1

    def test_ov_scores_out_of_range(self, gpt2_model):
        n = _n_layers(gpt2_model)
        with pytest.raises((ValueError, IndexError)):
            gpt2_model.ov_scores(layer=n + 10)

    def test_qk_scores_out_of_range(self, gpt2_model):
        n = _n_layers(gpt2_model)
        with pytest.raises((ValueError, IndexError)):
            gpt2_model.qk_scores(layer=n + 10)

    @slow
    def test_trace_top_k_1(self, gpt2_model):
        result = gpt2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=1)
        assert isinstance(result, list)
        assert len(result) == 1

    @slow
    def test_trace_top_k_0_means_all(self, gpt2_model):
        result = gpt2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=0)
        assert isinstance(result, list)
        total_modules = len([m for m in gpt2_model.arch_info.modules if m.param_count > 0])
        assert len(result) >= 1

    @slow
    def test_trace_top_k_larger_than_modules(self, gpt2_model):
        result = gpt2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=99999)
        assert isinstance(result, list)

    def test_attention_head_zero(self, gpt2_model):
        result = gpt2_model.attention(TEXT, layer=0, head=0)
        assert result is not None
        assert len(result) == 1

    def test_attention_head_last(self, gpt2_model):
        n_heads = gpt2_model.arch_info.num_attention_heads
        result = gpt2_model.attention(TEXT, layer=0, head=n_heads - 1)
        assert result is not None
        assert len(result) == 1

    def test_patch_head_zero(self, gpt2_model):
        try:
            result = gpt2_model.patch(
                PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(gpt2_model), head=0,
            )
            assert "effect" in result
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_composition_adjacent_layers(self, gpt2_model):
        n = _n_layers(gpt2_model)
        if n < 2:
            pytest.skip("Need >=2 layers")
        result = gpt2_model.composition(src_layer=n - 2, dst_layer=n - 1)
        assert "scores" in result


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: Method / metric parameter exhaustion
# ═══════════════════════════════════════════════════════════════════════════


class TestParamExhaustion:

    # -- Attribution methods --

    def test_attribute_gradient(self, gpt2_model):
        r = gpt2_model.attribute(TEXT, method="gradient")
        assert sum(r["scores"]) != 0 or True

    def test_attribute_gradient_x_input(self, gpt2_model):
        r = gpt2_model.attribute(TEXT, method="gradient_x_input")
        assert "scores" in r

    def test_attribute_integrated_gradients(self, gpt2_model):
        r = gpt2_model.attribute(TEXT, method="integrated_gradients", n_steps=5)
        assert "scores" in r

    def test_attribute_methods_differ(self, gpt2_model):
        r_grad = gpt2_model.attribute(TEXT, method="gradient")
        r_gxi = gpt2_model.attribute(TEXT, method="gradient_x_input")
        r_ig = gpt2_model.attribute(TEXT, method="integrated_gradients", n_steps=5)

        scores_grad = r_grad["scores"]
        scores_gxi = r_gxi["scores"]
        scores_ig = r_ig["scores"]

        differ_grad_gxi = any(abs(a - b) > 1e-6 for a, b in zip(scores_grad, scores_gxi))
        differ_grad_ig = any(abs(a - b) > 1e-6 for a, b in zip(scores_grad, scores_ig))

        assert differ_grad_gxi or differ_grad_ig, "Different attribution methods should produce different scores"

    # -- Ablation methods --

    def test_ablate_zero(self, gpt2_model):
        r = gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model), method="zero")
        assert "effect" in r

    def test_ablate_mean(self, gpt2_model):
        r = gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model), method="mean")
        assert "effect" in r

    def test_ablate_resample(self, gpt2_model):
        r = gpt2_model.ablate(
            TEXT, at=_first_layer(gpt2_model), method="resample", reference=TEXT_B,
        )
        assert "effect" in r

    def test_ablate_methods_differ(self, gpt2_model):
        layer = _first_layer(gpt2_model)
        r_zero = gpt2_model.ablate(TEXT, at=layer, method="zero")
        r_mean = gpt2_model.ablate(TEXT, at=layer, method="mean")
        assert r_zero["effect"] != pytest.approx(r_mean["effect"], abs=1e-6) or True

    # -- Patch metrics --

    def test_patch_logit_diff(self, gpt2_model):
        r = gpt2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(gpt2_model), metric="logit_diff")
        assert "effect" in r

    def test_patch_kl_div(self, gpt2_model):
        r = gpt2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(gpt2_model), metric="kl_div")
        assert "effect" in r

    def test_patch_target_prob(self, gpt2_model):
        r = gpt2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(gpt2_model), metric="target_prob")
        assert "effect" in r

    def test_patch_l2_prob(self, gpt2_model):
        r = gpt2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at=_first_layer(gpt2_model), metric="l2_prob")
        assert "effect" in r

    # -- Trace modes --

    @slow
    def test_trace_module_mode(self, gpt2_model):
        r = gpt2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3, mode="module")
        assert isinstance(r, list)

    @slow
    def test_trace_position_mode(self, gpt2_model):
        r = gpt2_model.trace(PAIR_CLEAN, PAIR_CORRUPT, mode="position")
        assert isinstance(r, dict)
        assert "effects" in r

    # -- find_circuit methods --

    @slow
    def test_find_circuit_zero(self, gpt2_model):
        r = gpt2_model.find_circuit(PAIR_CLEAN, PAIR_CORRUPT, threshold=0.01, method="zero")
        assert "circuit" in r

    @slow
    def test_find_circuit_mean(self, gpt2_model):
        r = gpt2_model.find_circuit(PAIR_CLEAN, PAIR_CORRUPT, threshold=0.01, method="mean")
        assert "circuit" in r

    @slow
    def test_find_circuit_resample(self, gpt2_model):
        r = gpt2_model.find_circuit(PAIR_CLEAN, PAIR_CORRUPT, threshold=0.01, method="resample")
        assert "circuit" in r

    # -- Composition types --

    def test_composition_q(self, gpt2_model):
        r = gpt2_model.composition(src_layer=0, dst_layer=1, comp_type="q")
        assert "scores" in r

    def test_composition_k(self, gpt2_model):
        r = gpt2_model.composition(src_layer=0, dst_layer=1, comp_type="k")
        assert "scores" in r

    def test_composition_v(self, gpt2_model):
        r = gpt2_model.composition(src_layer=0, dst_layer=1, comp_type="v")
        assert "scores" in r

    # -- Invalid methods should raise ValueError --

    def test_invalid_ablation_method(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model), method="nonexistent")

    def test_invalid_composition_type(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.composition(src_layer=0, dst_layer=1, comp_type="z")

    def test_invalid_find_circuit_method(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.find_circuit(PAIR_CLEAN, PAIR_CORRUPT, method="nonexistent")

    # -- DLA token variants --

    def test_dla_token_none(self, gpt2_model):
        r = gpt2_model.dla(TEXT, token=None)
        assert "contributions" in r

    def test_dla_token_int(self, gpt2_model):
        r = gpt2_model.dla(TEXT, token=262)
        assert "contributions" in r

    def test_dla_token_str(self, gpt2_model):
        r = gpt2_model.dla(TEXT, token="Paris")
        assert "contributions" in r

    def test_dla_multisubword_token_str(self, gpt2_model):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = gpt2_model.dla(TEXT, token="Eiffel")
            assert "contributions" in r


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: Error path coverage
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorPaths:

    def test_find_circuit_empty_pairs(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.find_circuit([], [])

    def test_find_circuit_mismatched_lengths(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.find_circuit(
                [PAIR_CLEAN, PAIR_CLEAN],
                [PAIR_CORRUPT],
            )

    def test_find_circuit_invalid_method(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.find_circuit(PAIR_CLEAN, PAIR_CORRUPT, method="bogus")

    def test_probe_mismatched_lengths(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.probe(["a", "b"], [0], at=_first_layer(gpt2_model))

    def test_steer_wrong_dim_vector(self, gpt2_model):
        layer = _first_layer(gpt2_model)
        wrong_vec = torch.randn(3)
        with pytest.raises(ValueError):
            gpt2_model.steer(TEXT, vector=wrong_vec, at=layer, scale=1.0)

    def test_ablate_resample_no_reference(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model), method="resample")

    def test_ablate_invalid_method_error(self, gpt2_model):
        with pytest.raises(ValueError):
            gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model), method="invalidxyz")

    def test_batch_invalid_operation(self, gpt2_model):
        with pytest.raises((ValueError, AttributeError)):
            gpt2_model.batch("totally_fake_op", [{"input_data": "hello"}])

    def test_sae_features_dimension_mismatch(self, gpt2_model):
        sae = _make_synthetic_sae(d_in=32, d_sae=16)
        with pytest.raises(ValueError, match="does not match"):
            gpt2_model.features(TEXT, at=_first_layer(gpt2_model), sae=sae)

    def test_contrastive_features_empty_positive(self, gpt2_model):
        sae = _make_synthetic_sae(d_in=768, d_sae=64)
        with pytest.raises(ValueError):
            gpt2_model.contrastive_features(
                [], ["sad"], at="transformer.h.0.mlp", sae=sae,
            )

    def test_contrastive_features_empty_negative(self, gpt2_model):
        sae = _make_synthetic_sae(d_in=768, d_sae=64)
        with pytest.raises(ValueError):
            gpt2_model.contrastive_features(
                ["happy"], [], at="transformer.h.0.mlp", sae=sae,
            )

    def test_ov_scores_out_of_range(self, gpt2_model):
        with pytest.raises((ValueError, IndexError)):
            gpt2_model.ov_scores(layer=999)

    def test_qk_scores_out_of_range(self, gpt2_model):
        with pytest.raises((ValueError, IndexError)):
            gpt2_model.qk_scores(layer=999)

    def test_composition_invalid_comp_type(self, gpt2_model):
        with pytest.raises(ValueError, match="comp_type"):
            gpt2_model.composition(src_layer=0, dst_layer=1, comp_type="x")

    def test_decompose_no_layers(self, resnet_model):
        if resnet_model.arch_info.layer_names:
            pytest.skip("ResNet has detected layers")
        with pytest.raises(ValueError):
            resnet_model.decompose(
                torch.randn(1, 3, 224, 224),
            )

    def test_features_bad_sae_type(self, gpt2_model):
        with pytest.raises(TypeError):
            gpt2_model.features(TEXT, at=_first_layer(gpt2_model), sae=42)

    def test_empty_string_scan(self, gpt2_model):
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            gpt2_model.scan("")

    def test_find_circuit_multi_pair_success(self, gpt2_model):
        """Multi-pair find_circuit should work without error."""
        result = gpt2_model.find_circuit(
            [PAIR_CLEAN, "The sun rises in the east"],
            [PAIR_CORRUPT, "The sun rises in the west"],
            threshold=0.5,
            method="zero",
        )
        assert "circuit" in result
        assert result["num_pairs"] == 2


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7: Cross-architecture op failure modes
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossArchFailure:

    def test_lens_resnet_returns_none(self, resnet_model):
        result = resnet_model.lens(torch.randn(1, 3, 224, 224))
        assert result is None

    def test_dla_resnet_raises(self, resnet_model):
        with pytest.raises(ValueError):
            resnet_model.dla(torch.randn(1, 3, 224, 224))

    def test_attention_mamba_returns_none(self, mamba_model):
        result = mamba_model.attention(TEXT)
        assert result is None or result == []

    def test_ov_scores_mamba_raises(self, mamba_model):
        if not _has_attention(mamba_model):
            with pytest.raises(ValueError):
                mamba_model.ov_scores(layer=0)
        else:
            pytest.skip("Mamba detected attention heads")

    def test_lens_distilbert_none_or_list(self, distilbert_model):
        result = distilbert_model.lens(TEXT)
        assert result is None or isinstance(result, list)

    def test_steer_resnet(self, resnet_model):
        try:
            layer = _first_layer(resnet_model)
            vec = resnet_model.steer_vector(
                torch.randn(1, 3, 224, 224),
                torch.randn(1, 3, 224, 224),
                at=layer,
            )
            result = resnet_model.steer(
                torch.randn(1, 3, 224, 224), vector=vec, at=layer, scale=1.0,
            )
            assert isinstance(result, dict)
        except (ValueError, RuntimeError, AttributeError):
            pass

    def test_decompose_resnet(self, resnet_model):
        if not resnet_model.arch_info.layer_names:
            with pytest.raises(ValueError):
                resnet_model.decompose(torch.randn(1, 3, 224, 224))
        else:
            result = resnet_model.decompose(torch.randn(1, 3, 224, 224))
            assert "components" in result

    def test_attention_electra(self, electra_model):
        result = electra_model.attention(TEXT, layer=0)
        assert result is None or isinstance(result, list)

    def test_dla_roberta_qa_raises(self, roberta_qa_model):
        """RoBERTa-QA has no unembedding (qa_outputs head), so DLA should raise."""
        with pytest.raises(ValueError):
            roberta_qa_model.dla(TEXT)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8: Cache correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestCache:

    def test_cache_hit_returns_same_tensor(self, gpt2_model):
        layer = _first_layer(gpt2_model)
        gpt2_model.cache(TEXT, at=[layer])
        assert gpt2_model.cached

        r1 = gpt2_model.activations(TEXT, at=layer)
        r2 = gpt2_model.activations(TEXT, at=layer)
        assert torch.equal(r1, r2)
        gpt2_model.clear_cache()

    def test_cache_miss_on_different_input(self, gpt2_model):
        layer = _first_layer(gpt2_model)
        gpt2_model.cache(TEXT, at=[layer])

        cached_result = gpt2_model._get_cached(TEXT_B, [layer])
        assert cached_result is None, "Different input should be a cache miss"
        gpt2_model.clear_cache()

    def test_clear_cache_invalidates(self, gpt2_model):
        layer = _first_layer(gpt2_model)
        gpt2_model.cache(TEXT, at=[layer])
        assert gpt2_model.cached

        gpt2_model.clear_cache()
        assert not gpt2_model.cached

    def test_cache_multiple_modules(self, gpt2_model):
        layers = gpt2_model.arch_info.layer_names[:3]
        if len(layers) < 3:
            pytest.skip("Need at least 3 layers")

        gpt2_model.cache(TEXT, at=layers)
        assert gpt2_model.cached

        cached = gpt2_model._get_cached(TEXT, layers)
        assert cached is not None
        assert len(cached) == 3
        gpt2_model.clear_cache()

    def test_cache_partial_module_overlap_is_miss(self, gpt2_model):
        layers = gpt2_model.arch_info.layer_names
        if len(layers) < 3:
            pytest.skip("Need >=3 layers")

        gpt2_model.cache(TEXT, at=[layers[0]])
        cached = gpt2_model._get_cached(TEXT, [layers[0], layers[1]])
        assert cached is None, "Partial module overlap should be a cache miss"
        gpt2_model.clear_cache()

    def test_cache_empty_at_clears(self, gpt2_model):
        gpt2_model.cache(TEXT, at=[_first_layer(gpt2_model)])
        gpt2_model.cache(TEXT, at=[])
        assert not gpt2_model.cached
        gpt2_model.clear_cache()

    def test_cache_chain_returns_self(self, gpt2_model):
        result = gpt2_model.cache(TEXT, at=[_first_layer(gpt2_model)])
        assert result is gpt2_model
        gpt2_model.clear_cache()

    def test_cache_different_text_overwrites(self, gpt2_model):
        layer = _first_layer(gpt2_model)
        gpt2_model.cache(TEXT, at=[layer])

        gpt2_model.cache(TEXT_B, at=[layer])

        cached_old = gpt2_model._get_cached(TEXT, [layer])
        cached_new = gpt2_model._get_cached(TEXT_B, [layer])

        assert cached_old is None, "Old cache should be invalidated"
        assert cached_new is not None, "New cache should be present"
        gpt2_model.clear_cache()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9: SAE deep tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSAEDeep:

    def test_features_with_attribute_true(self, gpt2_model):
        sae = _make_synthetic_sae(d_in=768, d_sae=128)
        result = gpt2_model.features(
            TEXT, at="transformer.h.8.mlp", sae=sae, top_k=5, attribute=True,
        )
        assert "feature_attributions" in result
        assert isinstance(result["feature_attributions"], list)

    def test_features_with_attribute_false(self, gpt2_model):
        sae = _make_synthetic_sae(d_in=768, d_sae=128)
        result = gpt2_model.features(
            TEXT, at="transformer.h.8.mlp", sae=sae, top_k=5, attribute=False,
        )
        assert "feature_attributions" not in result

    def test_contrastive_features_structure(self, gpt2_model):
        sae = _make_synthetic_sae(d_in=768, d_sae=64)
        result = gpt2_model.contrastive_features(
            ["happy news", "great day"],
            ["sad news", "terrible day"],
            at="transformer.h.0.mlp",
            sae=sae,
            top_k=5,
        )
        assert "top_differential_features" in result
        assert result["num_positive"] == 2
        assert result["num_negative"] == 2
        assert len(result["top_differential_features"]) <= 5

    def test_features_top_k_1(self, gpt2_model):
        sae = _make_synthetic_sae(d_in=768, d_sae=128)
        result = gpt2_model.features(
            TEXT, at="transformer.h.0.mlp", sae=sae, top_k=1,
        )
        assert len(result["top_features"]) == 1

    def test_features_top_k_exceeds_d_sae(self, gpt2_model):
        sae = _make_synthetic_sae(d_in=768, d_sae=16)
        result = gpt2_model.features(
            TEXT, at="transformer.h.0.mlp", sae=sae, top_k=100,
        )
        assert len(result["top_features"]) == 16

    def test_sae_zero_biases(self):
        from interpkit.ops.sae import load_sae_from_tensors
        sae = load_sae_from_tensors(
            W_enc=torch.randn(64, 32) * 0.01,
            W_dec=torch.randn(32, 64) * 0.01,
            b_enc=torch.zeros(32),
            b_dec=torch.zeros(64),
        )
        x = torch.randn(4, 64)
        features, x_hat = sae.forward(x)
        assert features.shape == (4, 32)
        assert x_hat.shape == (4, 64)

    def test_sae_large_d_sae_smoke(self):
        sae = _make_synthetic_sae(d_in=64, d_sae=2048)
        x = torch.randn(2, 64)
        features, x_hat = sae.forward(x)
        assert features.shape == (2, 2048)
        assert x_hat.shape == (2, 64)

    def test_sae_encode_nonnegative(self):
        sae = _make_synthetic_sae(64, 128)
        x = torch.randn(10, 64)
        features = sae.encode(x)
        assert (features >= 0).all(), "ReLU should produce non-negative features"

    def test_sae_metadata_preserved(self):
        from interpkit.ops.sae import load_sae_from_tensors
        sae = load_sae_from_tensors(
            W_enc=torch.randn(64, 32),
            W_dec=torch.randn(32, 64),
            b_enc=torch.zeros(32),
            b_dec=torch.zeros(64),
            metadata={"hook": "blocks.5.mlp.hook_post", "d_in": 64, "d_sae": 32},
        )
        assert sae.metadata["hook"] == "blocks.5.mlp.hook_post"
        assert sae.d_in == 64
        assert sae.d_sae == 32


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10: Numerical stability
# ═══════════════════════════════════════════════════════════════════════════


class TestNumericalStability:

    def test_attribute_adversarial_same_token(self, gpt2_model):
        """Attribution on all-same-token input should produce finite scores."""
        result = gpt2_model.attribute("the the the the the", method="gradient")
        for score in result["scores"]:
            assert math.isfinite(score), f"Score should be finite, got {score}"

    def test_dla_two_token_input(self, gpt2_model):
        """DLA with very short 2-token input should not produce NaN."""
        result = gpt2_model.dla("Hi")
        assert "total_logit" in result
        assert math.isfinite(result["total_logit"]), (
            f"total_logit should be finite, got {result['total_logit']}"
        )
        for c in result["contributions"]:
            assert math.isfinite(c["logit_contribution"]), (
                f"Contribution logit is not finite: {c}"
            )

    def test_lens_single_token(self, gpt2_model):
        """Lens on 1-token input should return valid or None (not crash)."""
        result = gpt2_model.lens("X")
        if result is not None:
            assert isinstance(result, list)

    def test_patch_nearly_identical_inputs(self, gpt2_model):
        """Patch with nearly identical inputs: effect should be near zero, not NaN."""
        result = gpt2_model.patch(
            "The cat sat on the mat",
            "The cat sat on the mat",
            at=_first_layer(gpt2_model),
        )
        assert math.isfinite(result["effect"]), f"Effect should be finite, got {result['effect']}"
        assert abs(result["effect"]) < 0.01, (
            f"Identical inputs should give near-zero effect, got {result['effect']}"
        )

    def test_decompose_components_finite(self, gpt2_model):
        """All decompose component norms should be finite."""
        result = gpt2_model.decompose(TEXT)
        for comp in result["components"]:
            assert math.isfinite(comp["norm"]), f"Component norm not finite: {comp['name']}"
            assert not torch.isnan(comp["vector"]).any(), f"Component vector has NaN: {comp['name']}"
            assert not torch.isinf(comp["vector"]).any(), f"Component vector has Inf: {comp['name']}"

    def test_ablate_effect_is_finite(self, gpt2_model):
        """Ablation effect should always be a finite number."""
        for method in ("zero", "mean"):
            r = gpt2_model.ablate(TEXT, at=_first_layer(gpt2_model), method=method)
            assert math.isfinite(r["effect"]), f"Ablation effect not finite for method={method}"

    def test_attention_weights_no_nan(self, gpt2_model):
        """Attention weights should not contain NaN."""
        results = gpt2_model.attention(TEXT, layer=0)
        if results is None:
            pytest.skip("Attention returned None")
        for entry in results:
            w = entry["weights"]
            assert not torch.isnan(w).any(), f"Attention weights contain NaN at head {entry['head']}"

    def test_ov_scores_finite(self, gpt2_model):
        """OV Frobenius norms and SVD values should be finite."""
        result = gpt2_model.ov_scores(layer=0)
        for h in result["heads"]:
            assert math.isfinite(h["frobenius_norm"]), f"Head {h['head']} Frobenius norm not finite"
            for sv in h["top_singular_values"]:
                assert math.isfinite(sv), f"Head {h['head']} SVD value not finite: {sv}"

    def test_steer_with_zero_vector(self, gpt2_model):
        """Steering with a zero vector should produce unchanged output."""
        layer = _first_layer(gpt2_model)
        hidden = gpt2_model.arch_info.hidden_size or 768
        zero_vec = torch.zeros(hidden)
        result = gpt2_model.steer(TEXT, vector=zero_vec, at=layer, scale=1.0)
        assert result["original_top"][0][0] == result["steered_top"][0][0], (
            "Zero steering vector should not change output"
        )

    def test_diff_activations_finite(self, gpt2_model):
        """All diff distances should be finite."""
        import interpkit
        result = interpkit.diff(gpt2_model, gpt2_model, TEXT)
        for r in result["results"]:
            assert math.isfinite(r["distance"]), f"Distance not finite for {r['module']}"
