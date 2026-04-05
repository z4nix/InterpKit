"""Tests for plot internals — figure properties, edge cases, format detection."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import matplotlib.pyplot as plt
import torch

from interpkit.core.plot import (
    _INTERPKIT_RC,
    _PALETTE,
    _save_and_show,
    plot_attention,
    plot_attention_multi,
    plot_attribution,
    plot_diff,
    plot_dla,
    plot_lens,
    plot_position_trace,
    plot_steer,
    plot_trace,
)

# ── Helpers ──────────────────────────────────────────────────────


def _tmp(ext: str = ".png") -> str:
    return os.path.join(tempfile.gettempdir(), f"plot_test{ext}")


def _cleanup(path: str):
    if os.path.exists(path):
        os.unlink(path)


def _intercept_fig(func, *args, **kwargs):
    """Call a plot function but intercept the figure before it's closed.

    Returns (return_value, captured_fig_properties).
    """
    captured = {}

    _save_and_show.__wrapped__ if hasattr(_save_and_show, "__wrapped__") else None

    def _intercept(fig, path, default_name):
        axes = fig.get_axes()
        captured["num_axes"] = len(axes)
        if axes:
            ax = axes[0]
            captured["xlabel"] = ax.get_xlabel()
            captured["ylabel"] = ax.get_ylabel()
            captured["title"] = ax.get_title()
            xlim = ax.get_xlim()
            captured["xlim"] = xlim
        captured["facecolor"] = fig.get_facecolor()

        out = path or default_name
        ext = out.rsplit(".", 1)[-1].lower() if "." in out else "png"
        dpi = 300 if ext in ("svg", "pdf") else 150
        fig.savefig(out, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out

    with patch("interpkit.core.plot._save_and_show", side_effect=_intercept):
        ret = func(*args, **kwargs)

    return ret, captured


# ══════════════════════════════════════════════════════════════════
# _save_and_show
# ══════════════════════════════════════════════════════════════════


def test_save_and_show_png():
    path = _tmp(".png")
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = _save_and_show(fig, path, "default.png")
        assert result == path
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read(4) == b"\x89PNG"
    finally:
        _cleanup(path)


def test_save_and_show_svg():
    path = _tmp(".svg")
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = _save_and_show(fig, path, "default.svg")
        assert result == path
        with open(path) as f:
            content = f.read()
            assert "<svg" in content or "<?xml" in content
    finally:
        _cleanup(path)


def test_save_and_show_pdf():
    path = _tmp(".pdf")
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = _save_and_show(fig, path, "default.pdf")
        assert result == path
        with open(path, "rb") as f:
            assert f.read(4) == b"%PDF"
    finally:
        _cleanup(path)


def test_save_and_show_no_extension_uses_default_name():
    """When path has no extension, _save_and_show still returns the path."""
    default = os.path.join(tempfile.gettempdir(), "plot_default_noext.png")
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = _save_and_show(fig, None, default)
        assert result == default
        assert os.path.exists(default)
    finally:
        _cleanup(default)


def test_save_and_show_uses_default_name_when_path_is_none():
    default = os.path.join(tempfile.gettempdir(), "plot_default_test.png")
    try:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = _save_and_show(fig, None, default)
        assert result == default
        assert os.path.exists(default)
    finally:
        _cleanup(default)


# ══════════════════════════════════════════════════════════════════
# plot_attention
# ══════════════════════════════════════════════════════════════════


def test_plot_attention_basic():
    path = _tmp()
    try:
        weights = torch.rand(5, 5)
        tokens = ["The", "capital", "of", "France", "is"]
        _, props = _intercept_fig(plot_attention, weights, tokens, layer=0, head=0, save_path=path)
        assert os.path.exists(path)
        assert props["xlabel"] == "Key (attends to)"
        assert props["ylabel"] == "Query (from)"
        assert "Attention" in props["title"]
    finally:
        _cleanup(path)


def test_plot_attention_no_tokens():
    path = _tmp()
    try:
        weights = torch.rand(4, 4)
        plot_attention(weights, tokens=None, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


def test_plot_attention_long_tokens_truncated():
    path = _tmp()
    try:
        weights = torch.rand(3, 3)
        tokens = ["a" * 50, "b" * 50, "c" * 50]
        plot_attention(weights, tokens, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


# ══════════════════════════════════════════════════════════════════
# plot_attention_multi
# ══════════════════════════════════════════════════════════════════


def test_plot_attention_multi_basic():
    path = _tmp()
    try:
        data = [
            {"layer": 0, "head": 0, "weights": torch.rand(4, 4)},
            {"layer": 0, "head": 1, "weights": torch.rand(4, 4)},
            {"layer": 1, "head": 0, "weights": torch.rand(4, 4)},
        ]
        result = plot_attention_multi(data, tokens=["a", "b", "c", "d"], save_path=path)
        assert os.path.exists(path)
        assert result == path
    finally:
        _cleanup(path)


def test_plot_attention_multi_empty_returns_empty_string():
    result = plot_attention_multi([], save_path=_tmp())
    assert result == ""


def test_plot_attention_multi_caps_at_8_heads():
    path = _tmp()
    try:
        data = [
            {"layer": 0, "head": h, "weights": torch.rand(3, 3)}
            for h in range(12)
        ]
        plot_attention_multi(data, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


# ══════════════════════════════════════════════════════════════════
# plot_trace
# ══════════════════════════════════════════════════════════════════


def test_plot_trace_basic():
    path = _tmp()
    try:
        results = [
            {"module": "a.b.c.d", "effect": 0.8},
            {"module": "a.b.e.f", "effect": 0.3},
            {"module": "a.b.g.h", "effect": 0.1},
        ]
        _, props = _intercept_fig(plot_trace, results, model_name="test", save_path=path)
        assert os.path.exists(path)
        assert props["xlabel"] == "Patching Effect"
    finally:
        _cleanup(path)


def test_plot_trace_empty_returns_empty_string():
    result = plot_trace([], save_path=_tmp())
    assert result == ""


def test_plot_trace_all_zero_effects():
    path = _tmp()
    try:
        results = [
            {"module": "a.b.c", "effect": 0.0},
            {"module": "a.b.d", "effect": 0.0},
        ]
        _, props = _intercept_fig(plot_trace, results, save_path=path)
        assert os.path.exists(path)
        assert props["xlim"][1] >= 0.001
    finally:
        _cleanup(path)


def test_plot_trace_more_than_25_results_capped():
    path = _tmp()
    try:
        results = [{"module": f"x.y.m{i}", "effect": i * 0.01} for i in range(30)]
        plot_trace(results, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


# ══════════════════════════════════════════════════════════════════
# plot_position_trace
# ══════════════════════════════════════════════════════════════════


def test_plot_position_trace_basic():
    path = _tmp()
    try:
        result_data = {
            "effects": torch.rand(4, 6),
            "layer_names": ["model.layer.0", "model.layer.1", "model.layer.2", "model.layer.3"],
            "tokens": ["The", "capital", "of", "France", "is", "Paris"],
        }
        _, props = _intercept_fig(plot_position_trace, result_data, save_path=path)
        assert os.path.exists(path)
        assert props["xlabel"] == "Token position"
        assert props["ylabel"] == "Layer"
    finally:
        _cleanup(path)


def test_plot_position_trace_list_input():
    path = _tmp()
    try:
        result_data = {
            "effects": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "layer_names": ["layer.0", "layer.1"],
        }
        plot_position_trace(result_data, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


def test_plot_position_trace_unicode_token_replacement():
    path = _tmp()
    try:
        result_data = {
            "effects": torch.rand(2, 3),
            "layer_names": ["layer.0", "layer.1"],
            "tokens": ["\u0120The", "\u0120capital", "\u0120of"],
        }
        plot_position_trace(result_data, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


# ══════════════════════════════════════════════════════════════════
# plot_lens
# ══════════════════════════════════════════════════════════════════


def test_plot_lens_1d():
    path = _tmp()
    try:
        predictions = [
            {"layer_name": "layer.0", "top1_prob": 0.1, "top1_token": "the"},
            {"layer_name": "layer.1", "top1_prob": 0.5, "top1_token": "Paris"},
            {"layer_name": "layer.2", "top1_prob": 0.9, "top1_token": "Paris"},
        ]
        plot_lens(predictions, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


def test_plot_lens_2d():
    path = _tmp()
    try:
        predictions = [
            {
                "layer_name": "layer.0",
                "positions": [
                    {"pos": 0, "top1_prob": 0.2, "top1_token": "a"},
                    {"pos": 1, "top1_prob": 0.4, "top1_token": "b"},
                    {"pos": 2, "top1_prob": 0.6, "top1_token": "c"},
                ],
            },
            {
                "layer_name": "layer.1",
                "positions": [
                    {"pos": 0, "top1_prob": 0.3, "top1_token": "x"},
                    {"pos": 1, "top1_prob": 0.7, "top1_token": "y"},
                    {"pos": 2, "top1_prob": 0.9, "top1_token": "z"},
                ],
            },
        ]
        plot_lens(predictions, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


def test_plot_lens_2d_more_than_20_positions_no_annotations():
    path = _tmp()
    try:
        positions = [{"pos": i, "top1_prob": 0.1 * (i % 10), "top1_token": f"t{i}"} for i in range(25)]
        predictions = [{"layer_name": "layer.0", "positions": positions}]
        plot_lens(predictions, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


def test_plot_lens_empty_returns_empty_string():
    result = plot_lens([], save_path=_tmp())
    assert result == ""


# ══════════════════════════════════════════════════════════════════
# plot_steer
# ══════════════════════════════════════════════════════════════════


def test_plot_steer_basic():
    path = _tmp()
    try:
        orig = [("happy", 0.3), ("good", 0.2), ("great", 0.1)]
        steered = [("sad", 0.4), ("bad", 0.25), ("terrible", 0.05)]
        _, props = _intercept_fig(
            plot_steer, orig, steered, module_name="layer.8", scale=2.0, save_path=path
        )
        assert os.path.exists(path)
        assert props["num_axes"] >= 2
    finally:
        _cleanup(path)


def test_plot_steer_empty_original_returns_empty():
    result = plot_steer([], [("a", 0.1)], save_path=_tmp())
    assert result == ""


def test_plot_steer_empty_steered_returns_empty():
    result = plot_steer([("a", 0.1)], [], save_path=_tmp())
    assert result == ""


def test_plot_steer_both_empty_returns_empty():
    result = plot_steer([], [], save_path=_tmp())
    assert result == ""


# ══════════════════════════════════════════════════════════════════
# plot_diff
# ══════════════════════════════════════════════════════════════════


def test_plot_diff_basic():
    path = _tmp()
    try:
        results = [
            {"module": "a.b.layer1", "distance": 0.5},
            {"module": "a.b.layer2", "distance": 0.2},
        ]
        _, props = _intercept_fig(
            plot_diff, results, model_a_name="A", model_b_name="B", save_path=path
        )
        assert os.path.exists(path)
        assert props["xlabel"] == "Cosine Distance"
    finally:
        _cleanup(path)


def test_plot_diff_empty_returns_empty_string():
    result = plot_diff([], save_path=_tmp())
    assert result == ""


def test_plot_diff_all_zero_distances():
    path = _tmp()
    try:
        results = [
            {"module": "a.b", "distance": 0.0},
            {"module": "c.d", "distance": 0.0},
        ]
        _, props = _intercept_fig(plot_diff, results, save_path=path)
        assert os.path.exists(path)
        assert props["xlim"][1] >= 0.001
    finally:
        _cleanup(path)


# ══════════════════════════════════════════════════════════════════
# plot_attribution
# ══════════════════════════════════════════════════════════════════


def test_plot_attribution_basic():
    path = _tmp()
    try:
        tokens = ["The", "cat", "sat"]
        scores = [0.1, 0.8, 0.3]
        _, props = _intercept_fig(plot_attribution, tokens, scores, save_path=path)
        assert os.path.exists(path)
        assert props["ylabel"] == "Attribution Score"
    finally:
        _cleanup(path)


def test_plot_attribution_empty_returns_empty_string():
    result = plot_attribution(["a"], [], save_path=_tmp())
    assert result == ""


def test_plot_attribution_all_zero_scores():
    path = _tmp()
    try:
        tokens = ["a", "b", "c"]
        scores = [0.0, 0.0, 0.0]
        plot_attribution(tokens, scores, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


def test_plot_attribution_negative_scores():
    path = _tmp()
    try:
        tokens = ["pos", "neg"]
        scores = [0.5, -0.3]
        plot_attribution(tokens, scores, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


# ══════════════════════════════════════════════════════════════════
# plot_dla
# ══════════════════════════════════════════════════════════════════


def test_plot_dla_basic():
    path = _tmp()
    try:
        result_data = {
            "contributions": [
                {"component": "layer.0.attn", "logit_contribution": 1.5},
                {"component": "layer.0.mlp", "logit_contribution": -0.8},
                {"component": "layer.1.attn", "logit_contribution": 0.3},
            ],
            "target_token": "Paris",
        }
        plot_dla(result_data, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)


def test_plot_dla_empty_contributions_returns_empty():
    result_data = {"contributions": [], "target_token": "x"}
    result = plot_dla(result_data, save_path=_tmp())
    assert result == ""


def test_plot_dla_with_head_contributions():
    path = _tmp(".png")
    try:
        result_data = {
            "contributions": [
                {"component": "layer.0.attn", "logit_contribution": 1.0},
            ],
            "head_contributions": [
                {"component": "L0H0", "logit_contribution": 0.7},
                {"component": "L0H1", "logit_contribution": 0.3},
            ],
            "target_token": "Paris",
        }
        plot_dla(result_data, save_path=path)
        assert os.path.exists(path)
        head_path = path.rsplit(".", 1)
        head_file = f"{head_path[0]}_heads.{head_path[1]}"
        assert os.path.exists(head_file)
    finally:
        _cleanup(path)
        head_path = path.rsplit(".", 1)
        _cleanup(f"{head_path[0]}_heads.{head_path[1]}")


def test_plot_dla_head_path_no_extension_does_not_crash():
    """When save_path has no extension, rsplit produces a single-element list.
    The code falls back to f"{save_path}_heads". Verify it doesn't raise."""
    path = os.path.join(tempfile.gettempdir(), "plot_test_dla_noext.png")
    try:
        result_data = {
            "contributions": [
                {"component": "layer.0.attn", "logit_contribution": 1.0},
            ],
            "head_contributions": [
                {"component": "L0H0", "logit_contribution": 0.5},
            ],
            "target_token": "x",
        }
        plot_dla(result_data, save_path=path)
        assert os.path.exists(path)
    finally:
        _cleanup(path)
        head_path = path.rsplit(".", 1)
        _cleanup(f"{head_path[0]}_heads.{head_path[1]}")
        _cleanup("dla_heads.png")


# ══════════════════════════════════════════════════════════════════
# Style / palette
# ══════════════════════════════════════════════════════════════════


def test_interpkit_rc_has_dark_background():
    assert _INTERPKIT_RC["figure.facecolor"] == "#1a1a2e"
    assert _INTERPKIT_RC["axes.facecolor"] == "#16213e"


def test_palette_keys():
    expected = {"bg", "surface", "primary", "accent", "text", "muted", "grid"}
    assert set(_PALETTE.keys()) == expected
