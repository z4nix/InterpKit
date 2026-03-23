"""Tests for plot/save functionality."""

from __future__ import annotations

import os
import tempfile


def test_attention_save(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "test_attn_plot.png")
    gpt2_model.attention("hello world", layer=0, head=0, save=path)
    assert os.path.exists(path)
    os.remove(path)


def test_attention_grid_save(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "test_attn_grid.png")
    gpt2_model.attention("hello world", layer=0, save=path)
    assert os.path.exists(path)
    os.remove(path)


def test_trace_save(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "test_trace.png")
    gpt2_model.trace("hello world", "goodbye world", top_k=5, save=path)
    assert os.path.exists(path)
    os.remove(path)


def test_lens_save(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "test_lens.png")
    gpt2_model.lens("The capital of France is", save=path)
    assert os.path.exists(path)
    os.remove(path)


def test_steer_save(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "test_steer.png")
    vector = gpt2_model.steer_vector("happy", "sad", at="transformer.h.8")
    gpt2_model.steer("The weather is", vector=vector, at="transformer.h.8", save=path)
    assert os.path.exists(path)
    os.remove(path)


def test_attribute_save(gpt2_model):
    path = os.path.join(tempfile.gettempdir(), "test_attr.png")
    gpt2_model.attribute("The capital of France is", save=path)
    assert os.path.exists(path)
    os.remove(path)


def test_diff_save(gpt2_model):
    import interpkit

    path = os.path.join(tempfile.gettempdir(), "test_diff.png")
    interpkit.diff(gpt2_model, gpt2_model, "hello world", save=path)
    assert os.path.exists(path)
    os.remove(path)
