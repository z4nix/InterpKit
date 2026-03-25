"""Broad architecture diversity test for layer detection.

Loads a large set of models spanning different architecture families and
validates that discovery correctly identifies attention paths, MLP paths,
projection styles, and layer types — including models whose naming
conventions would break the old regex-only detection.

Each model is tested in isolation so a download failure skips that model
without affecting the rest.  Models are cached by HuggingFace after the
first download.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import interpkit

_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_DTYPE = "float16"


# ---------------------------------------------------------------------------
# Model catalogue — (hub_id, load_kwargs, expected properties)
#
# Guidelines for adding models:
#   - Keep total parameter count ≤ ~600M per model so downloads and CPU
#     loading finish within reasonable timeouts.
#   - For architecture families that only ship large checkpoints, mark
#     them @pytest.mark.slow (see _SLOW below).
# ---------------------------------------------------------------------------

_MODELS_FAST: list[tuple[str, dict, dict]] = [
    # ── Standard decoder-only (separate QKV) ──────────────────────────
    ("gpt2", dict(device=_DEVICE),
     dict(qkv_style="fused", is_hybrid=False)),

    ("EleutherAI/gpt-neo-125m", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("EleutherAI/pythia-160m", dict(device=_DEVICE),
     dict(qkv_style="fused", is_hybrid=False)),

    ("facebook/opt-125m", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("HuggingFaceTB/SmolLM-135M", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    # ── GQA / modern decoder-only ─────────────────────────────────────
    ("Qwen/Qwen2-0.5B", dict(device=_DEVICE, dtype=_DTYPE),
     dict(qkv_style="separate", is_hybrid=False)),

    # ── Fused QKV (ALiBi) ─────────────────────────────────────────────
    ("bigscience/bloom-560m", dict(device=_DEVICE),
     dict(qkv_style="fused", is_hybrid=False)),

    # ── Encoder-only ──────────────────────────────────────────────────
    ("distilbert-base-uncased", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("bert-base-uncased", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("albert-base-v2", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("google/electra-small-discriminator", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("deepset/roberta-base-squad2", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    # ── Encoder-decoder ───────────────────────────────────────────────
    ("t5-small", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("google/flan-t5-small", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("facebook/bart-base", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),

    # ── Vision transformers ───────────────────────────────────────────
    ("google/vit-base-patch16-224", dict(device=_DEVICE),
     dict(qkv_style="separate", is_hybrid=False)),
]

_MODELS_SLOW: list[tuple[str, dict, dict]] = [
    # ── Llama-family (1B+) ────────────────────────────────────────────
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", dict(device=_DEVICE, dtype=_DTYPE),
     dict(qkv_style="separate", is_hybrid=False)),

    ("microsoft/phi-1_5", dict(device=_DEVICE, dtype=_DTYPE),
     dict(qkv_style="separate", is_hybrid=False)),

    # ── Hybrid recurrent + attention ──────────────────────────────────
    ("google/recurrentgemma-2b-it", dict(device=_DEVICE, dtype=_DTYPE),
     dict(is_hybrid=True)),

    # ── Gemma ─────────────────────────────────────────────────────────
    ("google/gemma-2b", dict(device=_DEVICE, dtype=_DTYPE),
     dict(qkv_style="separate", is_hybrid=False)),
]

_ALL_MODELS = _MODELS_FAST + [
    pytest.param(m, marks=pytest.mark.slow, id=m[0]) for m in _MODELS_SLOW
]


def _load(hub_id: str, kwargs: dict):
    """Load model or skip on failure."""
    try:
        return interpkit.load(hub_id, **kwargs)
    except Exception as e:
        pytest.skip(f"Cannot load {hub_id}: {e}")


# ---------------------------------------------------------------------------
# Parametrised tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=_ALL_MODELS, scope="module")
def model_and_expected(request):
    hub_id, kwargs, expected = request.param
    model = _load(hub_id, kwargs)
    yield hub_id, model, expected
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


class TestLayerDetection:
    """Every model must have layers detected with valid attn/mlp paths."""

    def test_layers_detected(self, model_and_expected):
        hub_id, model, _ = model_and_expected
        arch = model.arch_info
        assert arch.layer_names, f"{hub_id}: no layers detected"
        assert len(arch.layer_infos) == len(arch.layer_names)

    def test_attention_path_resolved(self, model_and_expected):
        """At least one layer should have attn_path resolved."""
        hub_id, model, _ = model_and_expected
        arch = model.arch_info
        attn_layers = [li for li in arch.layer_infos if li.attn_path]
        assert attn_layers, f"{hub_id}: no attention paths found in any layer"

    def test_projections_resolved(self, model_and_expected):
        """Every attention layer should have Q/K/V projections or fused QKV."""
        hub_id, model, _ = model_and_expected
        arch = model.arch_info
        for li in arch.layer_infos:
            if li.attn_path is None:
                continue
            has_separate = (li.q_proj_path and li.k_proj_path and li.v_proj_path)
            has_fused = li.qkv_proj_path is not None
            assert has_separate or has_fused, (
                f"{hub_id} layer {li.index}: attn_path={li.attn_path} "
                f"but no Q/K/V projections found"
            )

    def test_output_proj_resolved(self, model_and_expected):
        """Every attention layer should have an output projection."""
        hub_id, model, _ = model_and_expected
        arch = model.arch_info
        for li in arch.layer_infos:
            if li.attn_path is None:
                continue
            assert li.o_proj_path is not None, (
                f"{hub_id} layer {li.index}: no output projection found"
            )

    def test_qkv_style_known(self, model_and_expected):
        """Attention layers should have a known QKV style, not 'unknown'."""
        hub_id, model, _ = model_and_expected
        arch = model.arch_info
        for li in arch.layer_infos:
            if li.attn_path is None:
                continue
            assert li.qkv_style in ("separate", "fused"), (
                f"{hub_id} layer {li.index}: qkv_style={li.qkv_style}"
            )


class TestLayerType:
    """layer_type must be set and consistent with resolved paths."""

    def test_layer_type_set(self, model_and_expected):
        hub_id, model, _ = model_and_expected
        valid = {"standard", "attention_only", "mlp_only", "recurrent"}
        for li in model.arch_info.layer_infos:
            assert li.layer_type in valid, (
                f"{hub_id} layer {li.index}: layer_type={li.layer_type}"
            )

    def test_layer_type_consistent_with_paths(self, model_and_expected):
        hub_id, model, _ = model_and_expected
        for li in model.arch_info.layer_infos:
            if li.layer_type == "standard":
                assert li.attn_path and li.mlp_path, (
                    f"{hub_id} L{li.index}: standard but missing path"
                )
            elif li.layer_type == "attention_only":
                assert li.attn_path and not li.mlp_path, (
                    f"{hub_id} L{li.index}: attention_only inconsistent"
                )
            elif li.layer_type == "mlp_only":
                assert not li.attn_path and li.mlp_path, (
                    f"{hub_id} L{li.index}: mlp_only inconsistent"
                )
            elif li.layer_type == "recurrent":
                assert not li.attn_path, (
                    f"{hub_id} L{li.index}: recurrent but has attn_path"
                )

    def test_expected_hybrid(self, model_and_expected):
        hub_id, model, expected = model_and_expected
        if "is_hybrid" in expected:
            assert model.arch_info.is_hybrid == expected["is_hybrid"], (
                f"{hub_id}: is_hybrid={model.arch_info.is_hybrid}, "
                f"expected {expected['is_hybrid']}"
            )


class TestExpectedProperties:
    """Validate architecture-specific expectations."""

    def test_qkv_style(self, model_and_expected):
        hub_id, model, expected = model_and_expected
        if "qkv_style" not in expected:
            return
        attn_layers = [li for li in model.arch_info.layer_infos if li.attn_path]
        if not attn_layers:
            pytest.skip(f"{hub_id}: no attention layers")
        actual = attn_layers[0].qkv_style
        assert actual == expected["qkv_style"], (
            f"{hub_id}: qkv_style={actual}, expected {expected['qkv_style']}"
        )


class TestConvenienceProperties:
    """attention_layer_indices, attention_layer_infos, is_hybrid."""

    def test_indices_match_infos(self, model_and_expected):
        _, model, _ = model_and_expected
        arch = model.arch_info
        assert arch.attention_layer_indices == [
            li.index for li in arch.attention_layer_infos
        ]

    def test_indices_are_sorted(self, model_and_expected):
        _, model, _ = model_and_expected
        indices = model.arch_info.attention_layer_indices
        assert indices == sorted(indices)


class TestOpsOnAttentionLayers:
    """OV/QK/composition should work on valid attention layers."""

    def test_ov_scores(self, model_and_expected):
        hub_id, model, _ = model_and_expected
        arch = model.arch_info
        if not arch.attention_layer_indices or not arch.num_attention_heads:
            pytest.skip(f"{hub_id}: no attention layers or heads")
        layer = arch.attention_layer_indices[0]
        result = model.ov_scores(layer=layer)
        assert "heads" in result
        assert result["layer"] == layer

    def test_qk_scores(self, model_and_expected):
        hub_id, model, _ = model_and_expected
        arch = model.arch_info
        if not arch.attention_layer_indices or not arch.num_attention_heads:
            pytest.skip(f"{hub_id}: no attention layers or heads")
        layer = arch.attention_layer_indices[0]
        result = model.qk_scores(layer=layer)
        assert "heads" in result
        assert result["layer"] == layer

    def test_composition(self, model_and_expected):
        hub_id, model, _ = model_and_expected
        arch = model.arch_info
        if len(arch.attention_layer_indices) < 2 or not arch.num_attention_heads:
            pytest.skip(f"{hub_id}: need >=2 attention layers")
        src = arch.attention_layer_indices[0]
        dst = arch.attention_layer_indices[1]
        result = model.composition(src_layer=src, dst_layer=dst, comp_type="q")
        assert "scores" in result


class TestAutoRedirect:
    """Ops should auto-redirect when called on non-attention layers."""

    def test_ov_scores_redirect(self, model_and_expected):
        hub_id, model, expected = model_and_expected
        if not expected.get("is_hybrid"):
            pytest.skip("Not a hybrid model")
        arch = model.arch_info
        non_attn = [li for li in arch.layer_infos if li.attn_path is None]
        if not non_attn or not arch.attention_layer_indices:
            pytest.skip("No non-attention layers or no attention layers")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.ov_scores(layer=non_attn[0].index)
        assert "heads" in result
        assert result["layer"] in arch.attention_layer_indices
        assert any("Redirecting" in str(x.message) for x in w)

    def test_composition_redirect(self, model_and_expected):
        hub_id, model, expected = model_and_expected
        if not expected.get("is_hybrid"):
            pytest.skip("Not a hybrid model")
        arch = model.arch_info
        non_attn = [li for li in arch.layer_infos if li.attn_path is None]
        if len(non_attn) < 2 or len(arch.attention_layer_indices) < 1:
            pytest.skip("Need >=2 non-attention layers and >=1 attention layer")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.composition(
                src_layer=non_attn[0].index,
                dst_layer=non_attn[1].index,
                comp_type="q",
            )
        assert "scores" in result
        assert any("Redirecting" in str(x.message) for x in w)
