"""Unit tests for interpkit/core/discovery.py internals.

Uses synthetic nn.Module trees — no HuggingFace downloads, fast execution.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from interpkit.core.discovery import (
    ModelArchInfo,
    ModuleInfo,
    LayerInfo,
    _detect_layers,
    _find_unembedding,
    _parse_hf_config,
    _resolve_layer_info,
    _split_fused_weight,
    extract_proj_weight,
    _LM_HEAD_PATTERNS,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers — synthetic module builders
# ═══════════════════════════════════════════════════════════════════════════


def _mi(name: str, type_name: str = "Module") -> ModuleInfo:
    return ModuleInfo(name=name, type_name=type_name, param_count=0)


class _Linear(nn.Module):
    """Minimal linear with .weight and .out_features."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        self.out_features = out_f


class _Conv1D(nn.Module):
    """Mimics GPT-2's Conv1D: weight shape (in, out), returned transposed."""

    __qualname__ = "Conv1D"

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_f, out_f))

    @property
    def __class__(self):
        # Make type(self).__name__ == "Conv1D" for detection
        return type("Conv1D", (nn.Module,), {})


class _Attn(nn.Module):
    """Synthetic attention with separate Q/K/V/O projections."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.q_proj = _Linear(d, d)
        self.k_proj = _Linear(d, d)
        self.v_proj = _Linear(d, d)
        self.out_proj = _Linear(d, d)


class _AttnDistilBERT(nn.Module):
    """DistilBERT-style with q_lin/k_lin/v_lin/out_lin."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.q_lin = _Linear(d, d)
        self.k_lin = _Linear(d, d)
        self.v_lin = _Linear(d, d)
        self.out_lin = _Linear(d, d)


class _AttnFusedConcat(nn.Module):
    """Fused QKV (concatenated, like GPT-2 c_attn)."""

    def __init__(self, d: int = 64, heads: int = 4):
        super().__init__()
        self.c_attn = _Linear(d, 3 * d)
        self.c_proj = _Linear(d, d)


class _AttnFusedQKV(nn.Module):
    """Fused QKV named query_key_value (like BLOOM/Pythia)."""

    def __init__(self, d: int = 64, heads: int = 4):
        super().__init__()
        self.query_key_value = _Linear(d, 3 * d)
        self.dense = _Linear(d, d)


class GPTNeoXAttention(nn.Module):
    """Interleaved fused QKV (class name triggers interleaved layout)."""

    def __init__(self, d: int = 64, heads: int = 4):
        super().__init__()
        self.query_key_value = _Linear(d, 3 * d)
        self.dense = _Linear(d, d)


class _MLP(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.fc1 = _Linear(d, d * 4)
        self.fc2 = _Linear(d * 4, d)


class _Layer(nn.Module):
    """Standard transformer layer with separate Q/K/V."""

    def __init__(self, d: int = 64, attn_cls=_Attn):
        super().__init__()
        self.self_attn = attn_cls(d)
        self.mlp = _MLP(d)


class _SimpleModel(nn.Module):
    """Minimal model with N layers and optional lm_head."""

    def __init__(self, n_layers: int = 3, d: int = 64, has_head: bool = True,
                 attn_cls=_Attn, head_name: str = "lm_head", vocab: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Layer(d, attn_cls) for _ in range(n_layers)])
        if has_head:
            setattr(self, head_name, _Linear(d, vocab))
        self.config = SimpleNamespace(
            hidden_size=d,
            num_hidden_layers=n_layers,
            num_attention_heads=4,
            vocab_size=vocab,
        )



# ═══════════════════════════════════════════════════════════════════════════
#  _detect_layers tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectLayers:

    def test_standard_layers(self):
        modules = [_mi(f"model.layers.{i}") for i in range(3)]
        result = _detect_layers(modules)
        assert result == ["model.layers.0", "model.layers.1", "model.layers.2"]

    def test_gpt2_style_layers(self):
        modules = [_mi(f"transformer.h.{i}") for i in range(12)]
        result = _detect_layers(modules)
        assert len(result) == 12
        assert result[0] == "transformer.h.0"
        assert result[-1] == "transformer.h.11"

    def test_picks_largest_group(self):
        modules = [
            _mi("encoder.0"),
            _mi("encoder.1"),
            _mi("decoder.layers.0"),
            _mi("decoder.layers.1"),
            _mi("decoder.layers.2"),
        ]
        result = _detect_layers(modules)
        assert result == ["decoder.layers.0", "decoder.layers.1", "decoder.layers.2"]

    def test_single_layer(self):
        modules = [_mi("model.layers.0")]
        result = _detect_layers(modules)
        assert result == ["model.layers.0"]

    def test_no_numbered_modules(self):
        modules = [_mi("embed"), _mi("head"), _mi("norm")]
        result = _detect_layers(modules)
        assert result == []

    def test_non_contiguous_numbering(self):
        modules = [_mi("model.layers.0"), _mi("model.layers.2"), _mi("model.layers.5")]
        result = _detect_layers(modules)
        assert result == ["model.layers.0", "model.layers.2", "model.layers.5"]


# ═══════════════════════════════════════════════════════════════════════════
#  _find_unembedding tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFindUnembedding:

    def test_finds_lm_head(self):
        model = _SimpleModel(has_head=True, head_name="lm_head")
        assert _find_unembedding(model) == "lm_head"

    def test_finds_embed_out(self):
        model = _SimpleModel(has_head=True, head_name="embed_out")
        assert _find_unembedding(model) == "embed_out"

    def test_finds_output_projection(self):
        model = _SimpleModel(has_head=True, head_name="output_projection")
        assert _find_unembedding(model) == "output_projection"

    def test_relaxed_pass_with_vocab_size(self):
        model = _SimpleModel(has_head=True, head_name="classifier", vocab=500)
        model.config.vocab_size = 500
        result = _find_unembedding(model)
        assert result == "classifier"

    def test_returns_none_when_no_head(self):
        model = _SimpleModel(has_head=False)
        assert _find_unembedding(model) is None

    def test_ignores_modules_without_weight(self):
        model = _SimpleModel(has_head=False)
        model.lm_head = nn.ReLU()
        assert _find_unembedding(model) is None


# ═══════════════════════════════════════════════════════════════════════════
#  _parse_hf_config tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseHfConfig:

    def test_standard_config(self):
        model = nn.Linear(10, 10)
        model.config = SimpleNamespace(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            vocab_size=30522,
        )
        info = _parse_hf_config(model)
        assert info["hidden_size"] == 768
        assert info["num_layers"] == 12
        assert info["num_attention_heads"] == 12
        assert info["vocab_size"] == 30522

    def test_gpt2_config_aliases(self):
        model = nn.Linear(10, 10)
        model.config = SimpleNamespace(
            n_embd=768,
            n_layer=12,
            n_head=12,
            vocab_size=50257,
        )
        info = _parse_hf_config(model)
        assert info["hidden_size"] == 768
        assert info["num_layers"] == 12
        assert info["num_attention_heads"] == 12

    def test_t5_config_aliases(self):
        model = nn.Linear(10, 10)
        model.config = SimpleNamespace(
            d_model=512,
            num_layers=6,
            num_heads=8,
            vocab_size=32128,
        )
        info = _parse_hf_config(model)
        assert info["hidden_size"] == 512
        assert info["num_layers"] == 6
        assert info["num_attention_heads"] == 8

    def test_missing_config_returns_empty(self):
        model = nn.Linear(10, 10)
        info = _parse_hf_config(model)
        assert info == {}

    def test_partial_config(self):
        model = nn.Linear(10, 10)
        model.config = SimpleNamespace(hidden_size=256)
        info = _parse_hf_config(model)
        assert info["hidden_size"] == 256
        assert info.get("num_layers") is None
        assert info.get("num_attention_heads") is None

    def test_gqa_num_key_value_heads(self):
        model = nn.Linear(10, 10)
        model.config = SimpleNamespace(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            vocab_size=32000,
        )
        info = _parse_hf_config(model)
        assert info["num_key_value_heads"] == 4
        assert info["num_attention_heads"] == 12


# ═══════════════════════════════════════════════════════════════════════════
#  _resolve_layer_info / projection resolution tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveLayerInfo:

    def test_separate_qkv(self):
        model = _SimpleModel(n_layers=1, attn_cls=_Attn)
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.attn_path == "layers.0.self_attn"
        assert info.qkv_style == "separate"
        assert info.q_proj_path is not None
        assert info.k_proj_path is not None
        assert info.v_proj_path is not None
        assert info.o_proj_path == "layers.0.self_attn.out_proj"

    def test_distilbert_style_qkv(self):
        model = _SimpleModel(n_layers=1, attn_cls=_AttnDistilBERT)
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.qkv_style == "separate"
        assert "q_lin" in info.q_proj_path
        assert "k_lin" in info.k_proj_path
        assert "v_lin" in info.v_proj_path
        assert "out_lin" in info.o_proj_path

    def test_fused_c_attn_concatenated(self):
        model = _SimpleModel(n_layers=1, attn_cls=_AttnFusedConcat)
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.qkv_style == "fused"
        assert info.qkv_layout == "concatenated"
        assert "c_attn" in info.qkv_proj_path
        assert info.o_proj_path is not None
        assert "c_proj" in info.o_proj_path

    def test_fused_query_key_value(self):
        model = _SimpleModel(n_layers=1, attn_cls=_AttnFusedQKV)
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.qkv_style == "fused"
        assert "query_key_value" in info.qkv_proj_path

    def test_gptneox_interleaved_layout(self):
        class _LayerNeoX(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.attention = GPTNeoXAttention(d)
                self.mlp = _MLP(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_LayerNeoX()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.qkv_layout == "interleaved"

    def test_o_proj_various_names(self):
        for o_name in ("out_proj", "c_proj", "o_proj", "out_lin", "o"):
            class _A(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = _Linear(64, 64)
                    self.k_proj = _Linear(64, 64)
                    self.v_proj = _Linear(64, 64)
                    setattr(self, o_name, _Linear(64, 64))

            class _L(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.self_attn = _A()

            model = nn.Module()
            model.layers = nn.ModuleList([_L()])
            info = _resolve_layer_info(model, "layers.0", 0)
            assert info.o_proj_path is not None, f"Failed for o_proj name: {o_name}"
            assert o_name in info.o_proj_path

    def test_missing_attention_submodule(self):
        class _NoAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = _Linear(64, 64)

        model = nn.Module()
        model.layers = nn.ModuleList([_NoAttn()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.attn_path is None

    def test_mlp_detected(self):
        model = _SimpleModel(n_layers=1)
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.mlp_path == "layers.0.mlp"


# ═══════════════════════════════════════════════════════════════════════════
#  extract_proj_weight / _split_fused_weight tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractProjWeight:

    def test_separate_returns_weight(self):
        model = _SimpleModel(n_layers=1, d=64, attn_cls=_Attn)
        info = _resolve_layer_info(model, "layers.0", 0)
        w = extract_proj_weight(model, info, "q", num_heads=4)
        assert w is not None
        assert w.shape == (64, 64)

    def test_fused_concat_splits_correctly(self):
        d, heads = 64, 4
        model = _SimpleModel(n_layers=1, d=d, attn_cls=_AttnFusedConcat)
        info = _resolve_layer_info(model, "layers.0", 0)
        for proj in ("q", "k", "v"):
            w = extract_proj_weight(model, info, proj, num_heads=heads)
            assert w is not None
            assert w.shape == (d, d), f"Wrong shape for {proj}: {w.shape}"

    def test_fused_concat_gqa(self):
        d, heads, kv_heads = 64, 8, 2
        head_dim = d // heads

        class _GQA(nn.Module):
            def __init__(self):
                super().__init__()
                total = heads * head_dim + 2 * kv_heads * head_dim
                self.query_key_value = _Linear(d, total)
                self.dense = _Linear(d, d)

        model = nn.Module()

        class _L(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _GQA()
                self.mlp = _MLP(d)

        model.layers = nn.ModuleList([_L()])
        info = _resolve_layer_info(model, "layers.0", 0)
        q = extract_proj_weight(model, info, "q", num_heads=heads, num_kv_heads=kv_heads)
        k = extract_proj_weight(model, info, "k", num_heads=heads, num_kv_heads=kv_heads)
        v = extract_proj_weight(model, info, "v", num_heads=heads, num_kv_heads=kv_heads)
        assert q.shape == (heads * head_dim, d)
        assert k.shape == (kv_heads * head_dim, d)
        assert v.shape == (kv_heads * head_dim, d)

    def test_split_fused_interleaved(self):
        d, heads = 64, 4
        head_dim = d // heads
        w = torch.randn(3 * d, d)
        q = _split_fused_weight(w, "q", heads, heads, is_conv1d=False, interleaved=True)
        k = _split_fused_weight(w, "k", heads, heads, is_conv1d=False, interleaved=True)
        v = _split_fused_weight(w, "v", heads, heads, is_conv1d=False, interleaved=True)
        assert q.shape == (d, d)
        assert k.shape == (d, d)
        assert v.shape == (d, d)
        reconstructed = torch.zeros_like(w)
        grouped_orig = w.view(heads, 3, head_dim, d)
        q_check = grouped_orig[:, 0, :, :].reshape(-1, d)
        assert torch.allclose(q, q_check)

    def test_unknown_style_returns_none(self):
        info = LayerInfo(name="layers.0", index=0, qkv_style="unknown")
        model = _SimpleModel(n_layers=1)
        w = extract_proj_weight(model, info, "q", num_heads=4)
        assert w is None


# ═══════════════════════════════════════════════════════════════════════════
#  Regex pattern spot-checks
# ═══════════════════════════════════════════════════════════════════════════


class TestRegexPatterns:

    def test_lm_head_patterns(self):
        for name in ("lm_head", "embed_out", "output_projection"):
            assert _LM_HEAD_PATTERNS.search(name), f"Should match: {name}"
        assert not _LM_HEAD_PATTERNS.search("mlp")
        assert not _LM_HEAD_PATTERNS.search("classifier")


# ═══════════════════════════════════════════════════════════════════════════
#  ModelArchInfo property
# ═══════════════════════════════════════════════════════════════════════════


class TestModelArchInfoProperty:

    def test_is_language_model_true(self):
        info = ModelArchInfo(has_lm_head=True, unembedding_name="lm_head")
        assert info.is_language_model is True

    def test_is_language_model_false_no_head(self):
        info = ModelArchInfo(has_lm_head=False, unembedding_name=None)
        assert info.is_language_model is False

    def test_is_language_model_false_head_no_unembed(self):
        info = ModelArchInfo(has_lm_head=True, unembedding_name=None)
        assert info.is_language_model is False
