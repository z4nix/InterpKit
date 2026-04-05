"""Smoke tests for discovery and core ops across multiple architectures.

Marked @pytest.mark.slow — run with ``pytest --slow`` to include these.
Each model fixture is session-scoped and loaded via load_or_skip.
"""

from __future__ import annotations

import pytest
import torch

TEXT = "The capital of France is"
pytestmark = pytest.mark.slow


# ── Discovery smoke tests ───────────────────────────────────────────────


class TestDiscoveryDistilBERT:
    def test_arch_info_valid(self, distilbert_model):
        arch = distilbert_model.arch_info
        assert arch.arch_family is not None
        assert arch.num_layers is not None
        assert arch.hidden_size is not None
        assert len(arch.layer_names) > 0

    def test_layers_detected(self, distilbert_model):
        arch = distilbert_model.arch_info
        assert len(arch.layer_infos) == len(arch.layer_names)

    def test_attention_resolved(self, distilbert_model):
        arch = distilbert_model.arch_info
        attn_layers = [li for li in arch.layer_infos if li.attn_path is not None]
        assert len(attn_layers) > 0

    def test_activations(self, distilbert_model):
        first_layer = distilbert_model.arch_info.layer_names[0]
        result = distilbert_model.activations(TEXT, at=first_layer)
        assert isinstance(result, torch.Tensor)
        assert result.dim() >= 2

    def test_inspect_runs(self, distilbert_model):
        distilbert_model.inspect()


class TestDiscoveryT5:
    def test_arch_info_valid(self, t5_model):
        arch = t5_model.arch_info
        assert arch.arch_family is not None
        assert arch.is_encoder_decoder is True
        assert len(arch.layer_names) > 0

    def test_layers_detected(self, t5_model):
        arch = t5_model.arch_info
        assert len(arch.layer_infos) > 0

    def test_activations(self, t5_model):
        first_layer = t5_model.arch_info.layer_names[0]
        result = t5_model.activations(TEXT, at=first_layer)
        assert isinstance(result, torch.Tensor)

    def test_inspect_runs(self, t5_model):
        t5_model.inspect()


class TestDiscoveryPythia:
    def test_arch_info_valid(self, pythia_model):
        arch = pythia_model.arch_info
        assert arch.arch_family is not None
        assert arch.num_layers is not None
        assert arch.hidden_size is not None
        assert arch.has_lm_head is True
        assert len(arch.layer_names) > 0

    def test_attention_with_qkv(self, pythia_model):
        arch = pythia_model.arch_info
        attn_layers = [li for li in arch.layer_infos if li.attn_path]
        assert len(attn_layers) > 0
        assert attn_layers[0].qkv_style != "unknown"

    def test_activations(self, pythia_model):
        first_layer = pythia_model.arch_info.layer_names[0]
        result = pythia_model.activations(TEXT, at=first_layer)
        assert isinstance(result, torch.Tensor)

    def test_trace_runs(self, pythia_model):
        result = pythia_model.trace(
            "The Eiffel Tower is in Paris",
            "The Eiffel Tower is in Rome",
            top_k=5,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_lens_runs(self, pythia_model):
        result = pythia_model.lens(TEXT)
        assert result is not None
        assert len(result) > 0

    def test_dla_runs(self, pythia_model):
        result = pythia_model.dla(TEXT)
        assert "contributions" in result
        assert len(result["contributions"]) > 0


class TestDiscoveryViT:
    def test_arch_info_valid(self, vit_model):
        arch = vit_model.arch_info
        assert arch.arch_family is not None
        assert len(arch.layer_names) > 0

    def test_not_language_model(self, vit_model):
        assert vit_model.arch_info.is_language_model is False

    def test_activations(self, vit_model, test_image_path):
        first_layer = vit_model.arch_info.layer_names[0]
        result = vit_model.activations(test_image_path, at=first_layer)
        assert isinstance(result, torch.Tensor)

    def test_attention_returns_patterns(self, vit_model, test_image_path):
        result = vit_model.attention(test_image_path)
        if result is not None:
            assert isinstance(result, list)

    def test_inspect_runs(self, vit_model):
        vit_model.inspect()


class TestDiscoverySummary:
    """Test the discovery_summary() method across models."""

    def test_gpt2_summary(self, gpt2_model):
        summary = gpt2_model.arch_info.discovery_summary()
        assert "Architecture:" in summary
        assert "Layers:" in summary
        assert "Attention resolved:" in summary

    def test_resnet_summary(self, resnet_model):
        summary = resnet_model.arch_info.discovery_summary()
        assert "Architecture:" in summary
