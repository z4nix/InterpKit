"""Tests for operations not covered by individual op test files.

Covers: head_activations, dla, decompose, ov_scores, qk_scores,
composition, find_circuit, scan, report, batch (happy path).
All tests use only the session-scoped GPT-2 fixture.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

TEXT = "The capital of France is"
CLEAN = "The Eiffel Tower is in Paris"
CORRUPT = "The Eiffel Tower is in Rome"


# ── head_activations ────────────────────────────────────────────────────

def test_head_activations_returns_dict(gpt2_model):
    result = gpt2_model.head_activations(TEXT, at="transformer.h.0.attn")
    assert "head_acts" in result
    assert "num_heads" in result
    assert "head_dim" in result
    assert isinstance(result["head_acts"], torch.Tensor)
    assert result["head_acts"].shape[0] == result["num_heads"]


def test_head_activations_no_output_proj(gpt2_model):
    result = gpt2_model.head_activations(
        TEXT, at="transformer.h.0.attn", output_proj=False,
    )
    assert result["head_acts"].shape[0] == result["num_heads"]


# ── dla ──────────────────────────────────────────────────────────────────

def test_dla_returns_contributions(gpt2_model):
    result = gpt2_model.dla(TEXT)
    assert "contributions" in result
    assert "target_token" in result
    assert "total_logit" in result
    assert len(result["contributions"]) > 0
    for entry in result["contributions"]:
        assert "component" in entry
        assert "logit_contribution" in entry


def test_dla_with_token(gpt2_model):
    result = gpt2_model.dla(TEXT, token="Paris")
    assert result["target_token"] == "Paris"


# ── decompose ────────────────────────────────────────────────────────────

def test_decompose_returns_components(gpt2_model):
    result = gpt2_model.decompose(TEXT)
    assert "components" in result
    assert "residual" in result
    assert isinstance(result["residual"], torch.Tensor)
    assert len(result["components"]) > 0
    for comp in result["components"]:
        assert "name" in comp
        assert "vector" in comp
        assert "norm" in comp


def test_decompose_specific_position(gpt2_model):
    result = gpt2_model.decompose(TEXT, position=0)
    assert result["position"] == 0


# ── ov_scores / qk_scores ───────────────────────────────────────────────

def test_ov_scores_returns_per_head(gpt2_model):
    result = gpt2_model.ov_scores(layer=0)
    assert "heads" in result
    assert len(result["heads"]) > 0
    for head in result["heads"]:
        assert "head" in head
        assert "frobenius_norm" in head


def test_qk_scores_returns_per_head(gpt2_model):
    result = gpt2_model.qk_scores(layer=0)
    assert "heads" in result
    assert len(result["heads"]) > 0
    for head in result["heads"]:
        assert "head" in head
        assert "frobenius_norm" in head


# ── composition ──────────────────────────────────────────────────────────

def test_composition_q(gpt2_model):
    result = gpt2_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    assert "scores" in result
    assert isinstance(result["scores"], torch.Tensor)
    assert result["comp_type"] == "q"


# ── find_circuit ─────────────────────────────────────────────────────────

@pytest.mark.timeout(180)
def test_find_circuit_returns_circuit(gpt2_model):
    result = gpt2_model.find_circuit(CLEAN, CORRUPT, threshold=0.01)
    assert "circuit" in result
    assert "excluded" in result
    assert isinstance(result["circuit"], list)


# ── scan ─────────────────────────────────────────────────────────────────

def test_scan_returns_sections(gpt2_model):
    result = gpt2_model.scan(TEXT)
    assert isinstance(result, dict)
    assert len(result) > 0


# ── report ───────────────────────────────────────────────────────────────

def test_report_generates_html(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "interpkit_test_report.html")
    try:
        result = gpt2_model.report(TEXT, save=path)
        assert isinstance(result, dict)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── batch (happy path) ──────────────────────────────────────────────────

def test_batch_attention(gpt2_model):
    dataset = [
        {"input_data": "Hello world"},
        {"input_data": "Goodbye world"},
    ]
    result = gpt2_model.batch("attention", dataset)
    assert result["count"] == 2
    assert len(result["results"]) == 2


def test_trace_batch(gpt2_model):
    dataset = [
        {"clean": CLEAN, "corrupted": CORRUPT},
    ]
    result = gpt2_model.trace_batch(dataset)
    assert result["count"] == 1


def test_dla_batch(gpt2_model):
    result = gpt2_model.dla_batch(["Hello world", TEXT])
    assert result["count"] == 2
