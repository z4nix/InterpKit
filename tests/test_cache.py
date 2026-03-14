"""Tests for the activation cache feature."""

from __future__ import annotations

import torch


def test_cache_populates(gpt2_model):
    gpt2_model.clear_cache()
    assert not gpt2_model.cached

    gpt2_model.cache("The capital of France is", at=["transformer.h.0.mlp"])
    assert gpt2_model.cached
    assert "transformer.h.0.mlp" in gpt2_model._cache

    gpt2_model.clear_cache()
    assert not gpt2_model.cached


def test_cache_returns_self_for_chaining(gpt2_model):
    gpt2_model.clear_cache()
    result = gpt2_model.cache("Hello world", at=["transformer.h.0.mlp"])
    assert result is gpt2_model
    gpt2_model.clear_cache()


def test_cache_auto_invalidates_on_new_input(gpt2_model):
    gpt2_model.clear_cache()
    gpt2_model.cache("Input A", at=["transformer.h.0.mlp"])
    hash_a = gpt2_model._cache_input_hash

    gpt2_model.cache("Input B", at=["transformer.h.0.mlp"])
    hash_b = gpt2_model._cache_input_hash

    assert hash_a != hash_b
    gpt2_model.clear_cache()


def test_cache_used_by_activations(gpt2_model):
    gpt2_model.clear_cache()
    gpt2_model.cache("The capital of France is", at=["transformer.h.8.mlp"])

    act = gpt2_model.activations("The capital of France is", at="transformer.h.8.mlp")
    assert isinstance(act, torch.Tensor)
    assert act.dim() >= 2

    gpt2_model.clear_cache()


def test_cache_with_default_at(gpt2_model):
    gpt2_model.clear_cache()
    gpt2_model.cache("Hello world")
    assert gpt2_model.cached
    assert len(gpt2_model._cache) > 0
    gpt2_model.clear_cache()


def test_clear_cache_resets_everything(gpt2_model):
    gpt2_model.cache("test", at=["transformer.h.0.mlp"])
    assert gpt2_model.cached
    assert gpt2_model._cache_input_hash is not None

    gpt2_model.clear_cache()
    assert not gpt2_model.cached
    assert gpt2_model._cache_input_hash is None
    assert len(gpt2_model._cache) == 0
