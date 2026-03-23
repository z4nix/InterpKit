"""Tests for render internals — console output, edge cases, helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

import pytest
import torch
from rich.console import Console

from interpkit.core.render import (
    _format_params,
    render_ablate,
    render_activations,
    render_attention,
    render_attribution_heatmap,
    render_attribution_tokens,
    render_decompose,
    render_diff,
    render_dla,
    render_features,
    render_inspect,
    render_lens,
    render_patch,
    render_position_trace,
    render_probe,
    render_steer,
    render_trace,
)


# ── Helpers ──────────────────────────────────────────────────────


def _capture(func, *args, **kwargs) -> str:
    """Capture console output from a render function by temporarily
    replacing the module-level console."""
    import interpkit.core.render as render_mod

    buf = StringIO()
    old_console = render_mod.console
    render_mod.console = Console(file=buf, width=200, force_terminal=True)
    try:
        func(*args, **kwargs)
    finally:
        render_mod.console = old_console
    return buf.getvalue()


@dataclass
class _FakeModuleInfo:
    name: str = "transformer.h.0"
    type_name: str = "Linear"
    param_count: int = 1000
    output_shape: tuple | None = (1, 768)
    role: str | None = None


@dataclass
class _FakeArchInfo:
    arch_family: str | None = "GPT2"
    num_layers: int | None = 12
    hidden_size: int | None = 768
    vocab_size: int | None = 50257
    num_attention_heads: int | None = 12
    num_key_value_heads: int | None = None
    has_lm_head: bool = True
    output_head_name: str | None = "lm_head"
    unembedding_name: str | None = "lm_head"
    modules: list = field(default_factory=list)
    layer_names: list = field(default_factory=list)
    layer_infos: list = field(default_factory=list)
    is_tl_model: bool = False
    is_encoder_decoder: bool = False
    project_out_path: str | None = None


# ══════════════════════════════════════════════════════════════════
# _format_params
# ══════════════════════════════════════════════════════════════════


def test_format_params_zero():
    assert _format_params(0) == "0"


def test_format_params_small():
    assert _format_params(999) == "999"


def test_format_params_thousands():
    assert _format_params(1_500) == "1.5K"


def test_format_params_millions():
    assert _format_params(1_500_000) == "1.5M"


def test_format_params_billions():
    assert _format_params(2_300_000_000) == "2.3B"


def test_format_params_exact_thousand():
    assert _format_params(1_000) == "1.0K"


def test_format_params_exact_million():
    assert _format_params(1_000_000) == "1.0M"


def test_format_params_exact_billion():
    assert _format_params(1_000_000_000) == "1.0B"


# ══════════════════════════════════════════════════════════════════
# render_inspect
# ══════════════════════════════════════════════════════════════════


def test_render_inspect_basic():
    arch = _FakeArchInfo(
        modules=[
            _FakeModuleInfo(name="embed", role="embed", param_count=500),
            _FakeModuleInfo(name="transformer.h.0.attn", role="attention", param_count=300),
            _FakeModuleInfo(name="transformer.h.0.mlp", role="mlp", param_count=200),
        ]
    )
    out = _capture(render_inspect, arch)
    assert "GPT2" in out
    assert "12 layers" in out
    assert "768" in out
    assert "embed" in out
    assert "transformer.h.0.attn" in out
    assert "transformer.h.0.mlp" in out
    assert "Module" in out
    assert "Role" in out


def test_render_inspect_with_nn_model():
    arch = _FakeArchInfo(modules=[_FakeModuleInfo(param_count=100)])
    model = torch.nn.Linear(10, 10)
    out = _capture(render_inspect, arch, nn_model=model)
    assert "params total" in out


def test_render_inspect_no_arch_family():
    arch = _FakeArchInfo(arch_family=None, num_layers=None, hidden_size=None, vocab_size=None)
    arch.modules = [_FakeModuleInfo()]
    out = _capture(render_inspect, arch)
    assert "params total" in out


# ══════════════════════════════════════════════════════════════════
# render_trace
# ══════════════════════════════════════════════════════════════════


def test_render_trace_basic():
    results = [
        {"module": "transformer.h.8.mlp", "effect": 0.5, "role": "mlp"},
        {"module": "transformer.h.4.attn", "effect": 0.3, "role": "attention"},
    ]
    out = _capture(render_trace, results, "gpt2", 100, top_k=20)
    assert "transformer.h.8.mlp" in out
    assert "0.500" in out or "0.5" in out
    assert "█" in out
    assert "Top component" in out


def test_render_trace_empty():
    out = _capture(render_trace, [], "gpt2", 100, None)
    assert "No significant causal effects" in out


def test_render_trace_shows_scan_hint():
    results = [{"module": "x", "effect": 0.1}]
    out = _capture(render_trace, results, "test", 50, top_k=10)
    assert "top-k 0" in out


def test_render_trace_zero_effect():
    results = [{"module": "x", "effect": 0.0}]
    out = _capture(render_trace, results, "test", 10, None)
    assert "0.000" in out


# ══════════════════════════════════════════════════════════════════
# render_position_trace
# ══════════════════════════════════════════════════════════════════


def test_render_position_trace_basic():
    result = {
        "effects": torch.rand(4, 6),
        "layer_names": ["layer.0", "layer.1", "layer.2", "layer.3"],
        "tokens": ["The", "cat", "sat", "on", "the", "mat"],
    }
    out = _capture(render_position_trace, result)
    assert "Position-Aware" in out
    assert "Layer" in out
    assert "4 layers" in out


def test_render_position_trace_list_effects():
    result = {
        "effects": [[0.1, 0.2], [0.3, 0.4]],
        "layer_names": ["L0", "L1"],
    }
    out = _capture(render_position_trace, result)
    assert "2 layers" in out


def test_render_position_trace_missing_tokens():
    result = {
        "effects": torch.rand(2, 3),
        "layer_names": ["L0", "L1"],
    }
    out = _capture(render_position_trace, result)
    assert "Position-Aware" in out


# ══════════════════════════════════════════════════════════════════
# render_lens
# ══════════════════════════════════════════════════════════════════


def test_render_lens_basic():
    predictions = [
        {
            "layer_name": "layer.0",
            "top1_token": "the",
            "top1_prob": 0.3,
            "top5_tokens": ["the", "a", "an", "this", "that"],
            "top5_probs": [0.3, 0.2, 0.15, 0.1, 0.05],
        },
        {
            "layer_name": "layer.11",
            "top1_token": "Paris",
            "top1_prob": 0.9,
            "top5_tokens": ["Paris", "London", "Berlin", "Rome", "Madrid"],
            "top5_probs": [0.9, 0.03, 0.02, 0.01, 0.01],
        },
    ]
    out = _capture(render_lens, predictions, "gpt2")
    assert "Logit Lens" in out
    assert "Paris" in out
    assert "layer.0" in out or "Layer" in out


# ══════════════════════════════════════════════════════════════════
# render_attribution_tokens
# ══════════════════════════════════════════════════════════════════


def test_render_attribution_tokens_basic():
    tokens = ["The", "cat", "sat"]
    scores = [0.1, 0.8, 0.3]
    out = _capture(render_attribution_tokens, tokens, scores)
    assert "Attribution" in out
    assert "cat" in out
    assert "█" in out


def test_render_attribution_tokens_empty():
    out = _capture(render_attribution_tokens, [], [])
    assert "No attribution scores" in out


def test_render_attribution_tokens_all_zero():
    tokens = ["a", "b"]
    scores = [0.0, 0.0]
    out = _capture(render_attribution_tokens, tokens, scores)
    assert "Attribution" in out


# ══════════════════════════════════════════════════════════════════
# render_attribution_heatmap
# ══════════════════════════════════════════════════════════════════


def test_render_attribution_heatmap_2d():
    path = os.path.join(tempfile.gettempdir(), "test_render_hm_2d.png")
    try:
        attr = torch.rand(8, 8)
        _capture(render_attribution_heatmap, attr, path)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_render_attribution_heatmap_3d():
    path = os.path.join(tempfile.gettempdir(), "test_render_hm_3d.png")
    try:
        attr = torch.rand(3, 8, 8)
        _capture(render_attribution_heatmap, attr, path)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_render_attribution_heatmap_4d():
    path = os.path.join(tempfile.gettempdir(), "test_render_hm_4d.png")
    try:
        attr = torch.rand(1, 3, 8, 8)
        _capture(render_attribution_heatmap, attr, path)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════
# render_patch
# ══════════════════════════════════════════════════════════════════


def test_render_patch():
    result = {"module": "transformer.h.8.mlp", "effect": 0.1234}
    out = _capture(render_patch, result)
    assert "transformer.h.8.mlp" in out
    assert "0.1234" in out


# ══════════════════════════════════════════════════════════════════
# render_activations
# ══════════════════════════════════════════════════════════════════


def test_render_activations_basic():
    cache = {
        "transformer.h.0": torch.randn(1, 5, 768),
        "transformer.h.11": torch.randn(1, 5, 768),
    }
    out = _capture(render_activations, cache)
    assert "Activations" in out
    assert "transformer.h.0" in out
    assert "Norm" in out
    assert "Mean" in out


def test_render_activations_single():
    cache = {"module": torch.tensor([1.0, 2.0, 3.0])}
    out = _capture(render_activations, cache)
    assert "module" in out


# ══════════════════════════════════════════════════════════════════
# render_ablate
# ══════════════════════════════════════════════════════════════════


def test_render_ablate_zero():
    result = {"module": "transformer.h.8.mlp", "effect": 0.5, "method": "zero"}
    out = _capture(render_ablate, result)
    assert "zero" in out
    assert "transformer.h.8.mlp" in out
    assert "0.5000" in out


def test_render_ablate_mean():
    result = {"module": "layer.0", "effect": 0.3, "method": "mean"}
    out = _capture(render_ablate, result)
    assert "mean" in out


def test_render_ablate_no_method():
    result = {"module": "layer.0", "effect": 0.2}
    out = _capture(render_ablate, result)
    assert "zero" in out


# ══════════════════════════════════════════════════════════════════
# render_attention
# ══════════════════════════════════════════════════════════════════


def test_render_attention_basic():
    data = [
        {
            "layer": 0,
            "head": 0,
            "top_pairs": [(0, 1, 0.95), (1, 0, 0.7), (2, 1, 0.5)],
            "entropy": 1.23,
        },
    ]
    tokens = ["The", "cat", "sat"]
    out = _capture(render_attention, data, tokens, "gpt2")
    assert "Attention Patterns" in out
    assert "Entropy" in out
    assert "1.23" in out


def test_render_attention_empty():
    out = _capture(render_attention, [], None, "gpt2")
    assert "No attention data" in out


def test_render_attention_out_of_bounds_tokens():
    data = [
        {
            "layer": 0,
            "head": 0,
            "top_pairs": [(0, 100, 0.8)],
            "entropy": 0.5,
        },
    ]
    tokens = ["a", "b"]
    out = _capture(render_attention, data, tokens, "test")
    assert "100" in out


# ══════════════════════════════════════════════════════════════════
# render_steer
# ══════════════════════════════════════════════════════════════════


def test_render_steer_basic():
    orig = [("happy", 0.3), ("good", 0.2)]
    steered = [("sad", 0.4), ("bad", 0.25)]
    out = _capture(render_steer, orig, steered, "layer.8", 2.0)
    assert "Steering" in out
    assert "happy" in out
    assert "sad" in out
    assert "scale=2.0" in out


def test_render_steer_different_lengths():
    orig = [("a", 0.5)]
    steered = [("x", 0.3), ("y", 0.2), ("z", 0.1)]
    out = _capture(render_steer, orig, steered, "layer.0", 1.0)
    assert "Steering" in out


def test_render_steer_more_than_10_capped():
    orig = [(f"t{i}", 0.1) for i in range(15)]
    steered = [(f"s{i}", 0.1) for i in range(15)]
    out = _capture(render_steer, orig, steered, "m", 1.0)
    assert "t9" in out
    assert "t10" not in out


# ══════════════════════════════════════════════════════════════════
# render_probe
# ══════════════════════════════════════════════════════════════════


def test_render_probe_holdout():
    result = {
        "module": "layer.8",
        "accuracy": 0.85,
        "eval_method": "holdout",
        "cv_accuracy": 0.83,
        "train_accuracy": 0.95,
        "top_features": [(42, 1.5), (13, 0.9), (7, 0.3)],
    }
    out = _capture(render_probe, result)
    assert "0.85" in out or "0.850" in out
    assert "holdout" in out.lower() or "Test accuracy" in out
    assert "dim" in out
    assert "█" in out


def test_render_probe_cv_only():
    result = {
        "module": "layer.0",
        "accuracy": 0.75,
        "eval_method": "cv_only",
    }
    out = _capture(render_probe, result)
    assert "0.75" in out or "0.750" in out
    assert "few samples" in out.lower() or "CV" in out


def test_render_probe_train_only():
    result = {
        "module": "layer.0",
        "accuracy": 0.90,
        "eval_method": "train_only",
    }
    out = _capture(render_probe, result)
    assert "0.90" in out or "0.900" in out


def test_render_probe_no_top_features():
    result = {
        "module": "layer.0",
        "accuracy": 0.5,
        "eval_method": "holdout",
    }
    out = _capture(render_probe, result)
    assert "0.5" in out or "0.500" in out


def test_render_probe_top_features_first_weight_zero():
    """When first feature weight is 0, bar_len should be 0 (no crash)."""
    result = {
        "module": "layer.0",
        "accuracy": 0.5,
        "eval_method": "holdout",
        "top_features": [(0, 0.0), (1, 0.0)],
    }
    out = _capture(render_probe, result)
    assert "dim" in out


# ══════════════════════════════════════════════════════════════════
# render_diff
# ══════════════════════════════════════════════════════════════════


def test_render_diff_basic():
    results = [
        {"module": "layer.0", "distance": 0.5},
        {"module": "layer.1", "distance": 0.2},
    ]
    out = _capture(render_diff, results, "modelA", "modelB")
    assert "Model Diff" in out
    assert "modelA" in out
    assert "modelB" in out
    assert "█" in out
    assert "Most changed" in out


def test_render_diff_empty():
    out = _capture(render_diff, [], "A", "B")
    assert "No differences" in out


def test_render_diff_zero_distances():
    results = [{"module": "x", "distance": 0.0}]
    out = _capture(render_diff, results, "A", "B")
    assert "0.0000" in out


# ══════════════════════════════════════════════════════════════════
# render_features
# ══════════════════════════════════════════════════════════════════


def test_render_features_basic():
    result = {
        "module": "layer.8",
        "num_active_features": 50,
        "total_features": 1000,
        "sparsity": 0.95,
        "reconstruction_error": 0.01,
        "top_features": [(42, 3.5), (99, 2.1), (7, 1.0)],
    }
    out = _capture(render_features, result)
    assert "SAE Features" in out
    assert "50" in out
    assert "1000" in out
    assert "█" in out


def test_render_features_no_active():
    result = {
        "module": "layer.0",
        "num_active_features": 0,
        "total_features": 100,
        "sparsity": 1.0,
        "reconstruction_error": 0.0,
        "top_features": [],
    }
    out = _capture(render_features, result)
    assert "No active features" in out


# ══════════════════════════════════════════════════════════════════
# render_decompose
# ══════════════════════════════════════════════════════════════════


def test_render_decompose_basic():
    result = {
        "components": [
            {"name": "layer.0.attn", "type": "attention", "norm": 5.0},
            {"name": "layer.0.mlp", "type": "mlp", "norm": 3.0},
        ],
        "position": -1,
        "residual": torch.randn(768),
    }
    out = _capture(render_decompose, result)
    assert "Decomposition" in out
    assert "layer.0.attn" in out
    assert "█" in out
    assert "residual norm" in out.lower() or "Final residual" in out


def test_render_decompose_no_residual():
    result = {
        "components": [{"name": "c", "type": "t", "norm": 1.0}],
        "position": 0,
    }
    out = _capture(render_decompose, result)
    assert "Decomposition" in out


def test_render_decompose_all_zero_norms():
    result = {
        "components": [
            {"name": "a", "type": "t", "norm": 0.0},
            {"name": "b", "type": "t", "norm": 0.0},
        ],
        "position": 0,
    }
    out = _capture(render_decompose, result)
    assert "0.000" in out


# ══════════════════════════════════════════════════════════════════
# render_dla
# ══════════════════════════════════════════════════════════════════


def test_render_dla_basic():
    result = {
        "target_token": "Paris",
        "total_logit": 5.5,
        "contributions": [
            {"component": "layer.0.attn", "type": "attention", "logit_contribution": 2.0},
            {"component": "layer.0.mlp", "type": "mlp", "logit_contribution": -1.0},
        ],
    }
    out = _capture(render_dla, result)
    assert "Paris" in out
    assert "5.5" in out or "5.500" in out
    assert "█" in out


def test_render_dla_with_heads():
    result = {
        "target_token": "x",
        "total_logit": 1.0,
        "contributions": [
            {"component": "c", "type": "t", "logit_contribution": 0.5},
        ],
        "head_contributions": [
            {"component": "L0H0", "logit_contribution": 0.3},
            {"component": "L0H1", "logit_contribution": 0.2},
        ],
    }
    out = _capture(render_dla, result)
    assert "Per-Head" in out
    assert "L0H0" in out


def test_render_dla_many_heads_deduplication():
    heads = [{"component": f"L0H{i}", "logit_contribution": 0.1 * i} for i in range(30)]
    result = {
        "target_token": "tok",
        "total_logit": 3.0,
        "contributions": [{"component": "c", "type": "t", "logit_contribution": 3.0}],
        "head_contributions": heads,
    }
    out = _capture(render_dla, result, top_k=5)
    assert "L0H0" in out


def test_render_dla_negative_contributions():
    result = {
        "target_token": "neg",
        "total_logit": -1.0,
        "contributions": [
            {"component": "c1", "type": "t", "logit_contribution": -2.0},
            {"component": "c2", "type": "t", "logit_contribution": 1.0},
        ],
    }
    out = _capture(render_dla, result)
    assert "-2.0000" in out
    assert "+1.0000" in out
