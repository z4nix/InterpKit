"""Regression tests — one named test per historically discovered bug.

Each test encodes a specific failure mode found during rounds 1-3 of
multi-model stress testing.  They use existing conftest fixtures and
verify the exact operation+model combination that originally broke.

Numbering follows the FAILURES.md catalog:
  R0-R18  = Round 1 (original multi-model suite)
  R2_T5_* = Round 2 (T5 operations after decoder_input_ids fix)
  G1-G5   = Round 3 (generalization tests)
"""

from __future__ import annotations

import pytest
import torch

TEXT = "The capital of France is"


# ═══════════════════════════════════════════════════════════════════════════
#  Round 1 regressions (original multi-model stress test)
# ═══════════════════════════════════════════════════════════════════════════


def _first_attn(model):
    for m in model.arch_info.modules:
        if m.role == "attention":
            return m.name
    pytest.skip("No attention module")


class TestRound1:
    """Bugs R0–R18 from the initial 19-failure report."""

    def test_r0_t5_discover_has_layers(self, t5_model):
        """T5 discovery crashed without decoder_input_ids in dummy input."""
        arch = t5_model.arch_info
        assert len(arch.layer_names) > 0

    def test_r1_roberta_qa_not_causal_lm(self, roberta_qa_model):
        """RoBERTa-QA was loaded as generic CausalLM instead of QA model."""
        arch = roberta_qa_model.arch_info
        assert "QuestionAnswering" in arch.arch_family or "qa" in arch.arch_family.lower()

    def test_r2_distilbert_head_activations(self, distilbert_model):
        """DistilBERT head_activations failed — out_lin not found."""
        at = _first_attn(distilbert_model)
        result = distilbert_model.head_activations(TEXT, at=at)
        assert "head_acts" in result

    def test_r3_vit_head_activations(self, vit_model, test_image_path):
        """ViT head_activations failed — nested output.dense not found."""
        at = _first_attn(vit_model)
        result = vit_model.head_activations(test_image_path, at=at)
        assert "head_acts" in result

    def test_r4_distilbert_attention_weights(self, distilbert_model):
        """DistilBERT attention returned None — SDPA gave no weights."""
        results = distilbert_model.attention(TEXT)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_r5_roberta_qa_lens_correct(self, roberta_qa_model):
        """Lens ran on RoBERTa-QA when it shouldn't (not a language model)."""
        results = roberta_qa_model.lens(TEXT)
        assert results is None

    def test_r6_distilbert_ov_scores(self, distilbert_model):
        """DistilBERT ov_scores failed — v_lin projection not recognized."""
        result = distilbert_model.ov_scores(layer=0)
        assert "heads" in result
        assert len(result["heads"]) > 0

    def test_r7_distilbert_qk_scores(self, distilbert_model):
        """DistilBERT qk_scores failed — q_lin/k_lin not recognized."""
        result = distilbert_model.qk_scores(layer=0)
        assert "heads" in result

    def test_r8_bloom_ov_scores_fused(self, bloom_model):
        """BLOOM ov_scores failed — fused query_key_value not handled."""
        result = bloom_model.ov_scores(layer=0)
        assert "heads" in result
        assert len(result["heads"]) > 0

    def test_r9_vit_ov_scores_mps_svd(self, vit_model):
        """ViT ov_scores crashed — SVD not implemented on MPS."""
        result = vit_model.ov_scores(layer=0)
        assert "heads" in result

    def test_r10_qwen2_ov_scores_mps_svd(self, qwen2_model):
        """Qwen2 ov_scores crashed — SVD not implemented on MPS."""
        result = qwen2_model.ov_scores(layer=0)
        assert "heads" in result

    def test_r11_qwen2_qk_scores_mps_svd(self, qwen2_model):
        """Qwen2 qk_scores crashed — SVD not implemented on MPS."""
        result = qwen2_model.qk_scores(layer=0)
        assert "heads" in result

    def test_r13_distilbert_composition(self, distilbert_model):
        """DistilBERT composition failed — output proj not found."""
        result = distilbert_model.composition(src_layer=0, dst_layer=1, comp_type="q")
        assert "scores" in result

    def test_r14_qwen2_composition_gqa(self, qwen2_model):
        """Qwen2 composition crashed — GQA shape mismatch in matmul."""
        result = qwen2_model.composition(src_layer=0, dst_layer=1, comp_type="q")
        assert "scores" in result

    def test_r18_diff_cross_device(self, gpt2_model, bloom_model):
        """diff(gpt2, bloom) crashed — cross-device tensor comparison."""
        import interpkit

        result = interpkit.diff(gpt2_model, bloom_model, TEXT)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Round 2 regressions (T5 operations after decoder_input_ids fix)
# ═══════════════════════════════════════════════════════════════════════════


class TestRound2T5:
    """T5 operations that failed until _inject_decoder_ids was added."""

    def test_r2_t5_activations(self, t5_model):
        """T5 activations crashed — missing decoder_input_ids in forward."""
        layer = t5_model.arch_info.layer_names[0]
        result = t5_model.activations(TEXT, at=layer)
        assert isinstance(result, torch.Tensor)

    def test_r2_t5_lens(self, t5_model):
        """T5 lens crashed — missing decoder_input_ids."""
        results = t5_model.lens(TEXT)
        assert results is None or isinstance(results, list)

    def test_r2_t5_attention(self, t5_model):
        """T5 attention crashed — missing decoder_input_ids."""
        results = t5_model.attention(TEXT)
        assert results is None or isinstance(results, list)

    def test_r2_t5_head_activations(self, t5_model):
        """T5 head_activations failed — 'o' projection not found."""
        at = _first_attn(t5_model)
        result = t5_model.head_activations(TEXT, at=at)
        assert "head_acts" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Round 3 regressions (generalization tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestRound3Generalization:
    """Bugs G1–G5 from the generalization test round."""

    def test_g1_flan_t5_activations(self, flan_t5_model):
        """Flan-T5 activations crashed — decoder_input_ids not in forward."""
        layer = flan_t5_model.arch_info.layer_names[0]
        result = flan_t5_model.activations(TEXT, at=layer)
        assert isinstance(result, torch.Tensor)

    def test_g2_opt_lens(self, opt_model):
        """OPT lens failed — embed_dim != hidden_size shape mismatch."""
        results = opt_model.lens(TEXT)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_g3_albert_ov_scores(self, albert_model):
        """ALBERT ov_scores failed — shared-weight layers, nested attention."""
        result = albert_model.ov_scores(layer=0)
        assert "heads" in result

    def test_g4_pythia_is_language_model(self, pythia_model):
        """Pythia not detected as LM — embed_out not in unembedding patterns."""
        assert pythia_model.arch_info.is_language_model is True
        assert pythia_model.arch_info.unembedding_name is not None

    def test_g5_pythia_ov_scores_fused(self, pythia_model):
        """Pythia ov_scores failed — fused interleaved QKV not handled."""
        result = pythia_model.ov_scores(layer=0)
        assert "heads" in result
        assert len(result["heads"]) > 0
