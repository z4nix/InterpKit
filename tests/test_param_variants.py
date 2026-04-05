"""Branch coverage for rarely-tested operation kwargs on GPT-2.

Ensures that non-default parameter combinations don't crash.
"""

from __future__ import annotations

import os
import tempfile

import pytest

TEXT = "The capital of France is"
CLEAN = "The Eiffel Tower is in Paris"
CORRUPT = "The Eiffel Tower is in Rome"


# ── ablate method variants ──────────────────────────────────────────────

def test_ablate_mean(gpt2_model):
    layer = gpt2_model.arch_info.layer_names[0]
    result = gpt2_model.ablate(TEXT, at=layer, method="mean")
    assert "effect" in result


def test_ablate_resample(gpt2_model):
    layer = gpt2_model.arch_info.layer_names[0]
    result = gpt2_model.ablate(TEXT, at=layer, method="resample", reference="The capital of Spain is")
    assert "effect" in result


# ── patch metric variants ──────────────────────────────────────────────

@pytest.mark.parametrize("metric", ["logit_diff", "kl_div", "target_prob", "l2_prob"])
def test_patch_metrics(gpt2_model, metric):
    layer = gpt2_model.arch_info.layer_names[0]
    result = gpt2_model.patch(CLEAN, CORRUPT, at=layer, metric=metric)
    assert "effect" in result


# ── trace metric and mode variants ─────────────────────────────────────

@pytest.mark.parametrize("metric", ["logit_diff", "kl_div"])
def test_trace_metrics(gpt2_model, metric):
    result = gpt2_model.trace(CLEAN, CORRUPT, top_k=3, metric=metric)
    assert isinstance(result, list)


def test_trace_position_mode(gpt2_model):
    result = gpt2_model.trace(CLEAN, CORRUPT, mode="position")
    assert isinstance(result, dict)
    assert "effects" in result
    assert "layer_names" in result


# ── attribute method variants ──────────────────────────────────────────

@pytest.mark.parametrize("method", ["gradient", "gradient_x_input", "integrated_gradients"])
def test_attribute_methods(gpt2_model, method):
    kwargs = {}
    if method == "integrated_gradients":
        kwargs["n_steps"] = 5
    result = gpt2_model.attribute(TEXT, method=method, **kwargs)
    assert "scores" in result or "grad" in result


# ── batch operation variants ───────────────────────────────────────────

def test_batch_attribute(gpt2_model):
    dataset = [{"input_data": TEXT}, {"input_data": "Hello world"}]
    result = gpt2_model.batch("attribute", dataset, op_kwargs={"method": "gradient"})
    assert result["count"] == 2


def test_batch_ablate(gpt2_model):
    layer = gpt2_model.arch_info.layer_names[0]
    dataset = [{"input_data": TEXT}, {"input_data": "Hello world"}]
    result = gpt2_model.batch("ablate", dataset, op_kwargs={"at": layer})
    assert result["count"] == 2


def test_batch_no_aggregate(gpt2_model):
    dataset = [{"input_data": TEXT}]
    result = gpt2_model.batch("attention", dataset, aggregate=False)
    assert "results" in result


# ── lens position variants ─────────────────────────────────────────────

def test_lens_position_first(gpt2_model):
    result = gpt2_model.lens(TEXT, position=0)
    assert result is not None


def test_lens_position_last(gpt2_model):
    result = gpt2_model.lens(TEXT, position=-1)
    assert result is not None


# ── composition type variants ──────────────────────────────────────────

@pytest.mark.parametrize("comp_type", ["q", "k", "v"])
def test_composition_types(gpt2_model, comp_type):
    result = gpt2_model.composition(src_layer=0, dst_layer=1, comp_type=comp_type)
    assert "scores" in result
    assert result["comp_type"] == comp_type


# ── dla position/token variants ───────────────────────────────────────

def test_dla_explicit_position(gpt2_model):
    result = gpt2_model.dla(TEXT, position=0)
    assert "contributions" in result


def test_dla_explicit_token_id(gpt2_model):
    result = gpt2_model.dla(TEXT, token=50256)
    assert "target_id" in result


# ── save output ────────────────────────────────────────────────────────

def test_trace_save_html(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "interpkit_test_trace.html")
    try:
        gpt2_model.trace(CLEAN, CORRUPT, top_k=3, html=path)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_attention_save_html(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "interpkit_test_attn.html")
    try:
        gpt2_model.attention(TEXT, html=path)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
