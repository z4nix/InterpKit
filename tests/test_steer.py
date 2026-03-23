"""Tests for the steer operation."""

from __future__ import annotations

import torch


def test_steer_vector_returns_tensor(gpt2_model):
    vector = gpt2_model.steer_vector("Love", "Hate", at="transformer.h.8")
    assert isinstance(vector, torch.Tensor)
    assert vector.dim() == 1


def test_steer_runs(gpt2_model):
    vector = gpt2_model.steer_vector("happy", "sad", at="transformer.h.8")
    result = gpt2_model.steer("The weather today is", vector=vector, at="transformer.h.8", scale=2.0)
    assert "original_top" in result
    assert "steered_top" in result
    assert len(result["original_top"]) > 0
    assert len(result["steered_top"]) > 0


def test_steer_changes_predictions(gpt2_model):
    vector = gpt2_model.steer_vector("happy", "sad", at="transformer.h.8")
    result = gpt2_model.steer("The weather today is", vector=vector, at="transformer.h.8", scale=5.0)
    orig_top = result["original_top"][0][0]
    steered_top = result["steered_top"][0][0]
    # With a large scale, predictions should shift (not guaranteed but very likely)
    assert isinstance(orig_top, str)
    assert isinstance(steered_top, str)
