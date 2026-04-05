"""Unit tests for interpkit/core/discovery.py internals.

Uses synthetic nn.Module trees — no HuggingFace downloads, fast execution.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from interpkit.core.discovery import (
    _LM_HEAD_PATTERNS,
    LayerInfo,
    ModelArchInfo,
    ModuleInfo,
    _detect_layers,
    _find_unembedding,
    _parse_hf_config,
    _resolve_layer_info,
    _split_fused_weight,
    extract_proj_weight,
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
                    setattr(self, o_name, _Linear(64, 64))  # noqa: B023

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
        torch.zeros_like(w)
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

    def test_attention_layer_indices(self):
        infos = [
            LayerInfo(name="l.0", index=0, attn_path=None, layer_type="recurrent"),
            LayerInfo(name="l.1", index=1, attn_path=None, layer_type="recurrent"),
            LayerInfo(name="l.2", index=2, attn_path="l.2.attn", layer_type="standard"),
            LayerInfo(name="l.3", index=3, attn_path=None, layer_type="recurrent"),
            LayerInfo(name="l.4", index=4, attn_path=None, layer_type="recurrent"),
            LayerInfo(name="l.5", index=5, attn_path="l.5.attn", layer_type="standard"),
        ]
        arch = ModelArchInfo(layer_infos=infos)
        assert arch.attention_layer_indices == [2, 5]

    def test_attention_layer_infos(self):
        infos = [
            LayerInfo(name="l.0", index=0, attn_path="l.0.attn"),
            LayerInfo(name="l.1", index=1, attn_path=None),
        ]
        arch = ModelArchInfo(layer_infos=infos)
        result = arch.attention_layer_infos
        assert len(result) == 1
        assert result[0].index == 0

    def test_is_hybrid_true(self):
        infos = [
            LayerInfo(name="l.0", index=0, layer_type="recurrent"),
            LayerInfo(name="l.1", index=1, layer_type="standard"),
        ]
        arch = ModelArchInfo(layer_infos=infos)
        assert arch.is_hybrid is True

    def test_is_hybrid_false(self):
        infos = [
            LayerInfo(name="l.0", index=0, layer_type="standard"),
            LayerInfo(name="l.1", index=1, layer_type="standard"),
        ]
        arch = ModelArchInfo(layer_infos=infos)
        assert arch.is_hybrid is False

    def test_is_hybrid_empty(self):
        arch = ModelArchInfo(layer_infos=[])
        assert arch.is_hybrid is False


# ═══════════════════════════════════════════════════════════════════════════
#  Bottom-up Q/K/V probe tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBottomUpProbe:
    """Verify attention detection works when the container has a non-standard
    name that _ATTN_RE won't match."""

    def test_nonstandard_name_with_qkv_inside(self):
        """temporal_block (RecurrentGemma-style) containing q/k/v projections."""
        class _WeirdAttn(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.q_proj = _Linear(d, d)
                self.k_proj = _Linear(d, d)
                self.v_proj = _Linear(d, d)
                self.o_proj = _Linear(d, d)

        class _WeirdLayer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.temporal_block = _WeirdAttn(d)
                self.mlp_block = _MLP(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_WeirdLayer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.attn_path == "layers.0.temporal_block"
        assert info.qkv_style == "separate"
        assert info.q_proj_path is not None
        assert info.k_proj_path is not None
        assert info.v_proj_path is not None
        assert info.o_proj_path is not None

    def test_nonstandard_name_with_fused_qkv(self):
        """Container with fused c_attn but non-standard container name."""
        class _FusedBlock(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.c_attn = _Linear(d, 3 * d)
                self.c_proj = _Linear(d, d)

        class _Layer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.compute_block = _FusedBlock(d)
                self.feed = _MLP(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.attn_path == "layers.0.compute_block"
        assert info.qkv_style == "fused"

    def test_no_qkv_projections_returns_none(self):
        """A child with linear modules but no Q/K/V — probe should NOT match."""
        class _Recurrent(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.gate = _Linear(d, d)
                self.state = _Linear(d, d)

        class _Layer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.temporal_block = _Recurrent(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.attn_path is None

    def test_standard_name_still_takes_fast_path(self):
        """When the name matches _ATTN_RE, probe is never needed."""
        model = _SimpleModel(n_layers=1, attn_cls=_Attn)
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.attn_path == "layers.0.self_attn"
        assert info.qkv_style == "separate"

    def test_nested_qkv_projections(self):
        """Q/K/V projections nested two levels deep — probe drills through
        the wrapper to the actual attention module."""
        class _Inner(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.q_proj = _Linear(d, d)
                self.k_proj = _Linear(d, d)
                self.v_proj = _Linear(d, d)
                self.out_proj = _Linear(d, d)

        class _Outer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.core = _Inner(d)

        class _Layer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.block_x = _Outer(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.attn_path == "layers.0.block_x.core"
        assert info.q_proj_path is not None


# ═══════════════════════════════════════════════════════════════════════════
#  MLP-by-elimination probe tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMLPByElimination:
    """Verify MLP detection works when _MLP_RE doesn't match the name."""

    def test_nonstandard_mlp_name(self):
        """mlp_block (RecurrentGemma-style) detected by elimination."""
        class _WeirdAttn(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.q_proj = _Linear(d, d)
                self.k_proj = _Linear(d, d)
                self.v_proj = _Linear(d, d)
                self.o_proj = _Linear(d, d)

        class _WeirdMLP(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.up = _Linear(d, d * 4)
                self.down = _Linear(d * 4, d)

        class _Layer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.temporal_block = _WeirdAttn(d)
                self.mlp_block = _WeirdMLP(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.mlp_path == "layers.0.mlp_block"

    def test_standard_mlp_name_still_works(self):
        model = _SimpleModel(n_layers=1)
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.mlp_path == "layers.0.mlp"

    def test_single_linear_not_mlp(self):
        """A child with only one Linear should not be tagged as MLP."""
        class _Layer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.proj = _Linear(d, d)

        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.mlp_path is None

    def test_norm_child_not_tagged_as_mlp(self):
        """Norm layers should be excluded even if they technically have weight."""
        class _Layer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.self_attn = _Attn(d)
                self.norm = nn.LayerNorm(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.mlp_path is None


# ═══════════════════════════════════════════════════════════════════════════
#  layer_type classification tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLayerType:

    def test_standard_layer(self):
        model = _SimpleModel(n_layers=1)
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.layer_type == "standard"

    def test_attention_only_layer(self):
        class _AttnOnlyLayer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.self_attn = _Attn(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_AttnOnlyLayer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.layer_type == "attention_only"

    def test_mlp_only_layer(self):
        class _MLPOnlyLayer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.mlp = _MLP(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_MLPOnlyLayer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.layer_type == "mlp_only"

    def test_recurrent_layer(self):
        class _RecurrentLayer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.gate = _Linear(d, d)

        model = nn.Module()
        model.layers = nn.ModuleList([_RecurrentLayer()])
        info = _resolve_layer_info(model, "layers.0", 0)
        assert info.layer_type == "recurrent"

    def test_block_types_override_recurrent(self):
        """Config-declared recurrent skips the attention probe entirely."""
        class _Layer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.self_attn = _Attn(d)
                self.mlp = _MLP(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        info = _resolve_layer_info(
            model, "layers.0", 0, block_types=["recurrent"],
        )
        assert info.layer_type == "recurrent"
        assert info.attn_path is None

    def test_block_types_attention_layer_proceeds(self):
        """Config-declared attention layer proceeds normally."""
        class _WeirdAttn(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.q_proj = _Linear(d, d)
                self.k_proj = _Linear(d, d)
                self.v_proj = _Linear(d, d)
                self.o_proj = _Linear(d, d)

        class _Layer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.temporal_block = _WeirdAttn(d)

        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        info = _resolve_layer_info(
            model, "layers.0", 0, block_types=["attention"],
        )
        assert info.attn_path == "layers.0.temporal_block"
        assert info.layer_type == "attention_only"

    def test_hybrid_model_mixed_types(self):
        """Simulate a hybrid model with alternating layer types."""
        class _AttnLayer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.temporal_block = _Attn(d)
                self.mlp_block = _MLP(d)

        class _RecurrentLayer(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.gate = _Linear(d, d)
                self.state = _Linear(d, d)
                self.mlp_block = _MLP(d)

        model = nn.Module()
        model.layers = nn.ModuleList([
            _RecurrentLayer(),  # 0
            _RecurrentLayer(),  # 1
            _AttnLayer(),       # 2
            _RecurrentLayer(),  # 3
            _RecurrentLayer(),  # 4
            _AttnLayer(),       # 5
        ])
        block_types = ["recurrent", "recurrent", "attention",
                       "recurrent", "recurrent", "attention"]

        infos = [
            _resolve_layer_info(model, f"layers.{i}", i, block_types=block_types)
            for i in range(6)
        ]

        assert infos[0].layer_type == "recurrent"
        assert infos[0].attn_path is None
        assert infos[1].layer_type == "recurrent"
        assert infos[2].layer_type == "standard"
        assert infos[2].attn_path is not None
        assert infos[3].layer_type == "recurrent"
        assert infos[5].layer_type == "standard"
        assert infos[5].attn_path is not None

        arch = ModelArchInfo(layer_infos=infos)
        assert arch.is_hybrid is True
        assert arch.attention_layer_indices == [2, 5]


# ═══════════════════════════════════════════════════════════════════════════
#  HF config block_types parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestBlockTypesParsing:

    def test_block_types_parsed(self):
        model = nn.Linear(10, 10)
        model.config = SimpleNamespace(
            hidden_size=256,
            num_hidden_layers=3,
            num_attention_heads=4,
            vocab_size=1000,
            block_types=("recurrent", "recurrent", "attention"),
        )
        info = _parse_hf_config(model)
        assert info["block_types"] == ["recurrent", "recurrent", "attention"]

    def test_layers_block_type_alias(self):
        model = nn.Linear(10, 10)
        model.config = SimpleNamespace(
            hidden_size=256,
            layers_block_type=["attention", "recurrent"],
        )
        info = _parse_hf_config(model)
        assert info["block_types"] == ["attention", "recurrent"]

    def test_no_block_types(self):
        model = nn.Linear(10, 10)
        model.config = SimpleNamespace(hidden_size=256)
        info = _parse_hf_config(model)
        assert "block_types" not in info
