"""Tests for the auto-discovery module."""

from __future__ import annotations

import torch
import torch.nn as nn

from interpkit.core.discovery import discover


class _Attn(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

class _MLP(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)

class _Layer(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d)

class ToyModel(nn.Module):
    def __init__(self, n_layers=2, d=32, vocab=100):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Layer(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer.self_attn.out_proj(x)
        return self.lm_head(x)


def test_discover_toy_model():
    model = ToyModel()
    info = discover(model)
    assert len(info.modules) > 0
    names = {m.name for m in info.modules}
    assert "embed" in names
    assert "lm_head" in names
    assert "layers.0" in names
    assert "layers.0.self_attn" in names
    assert "layers.0.mlp" in names


def test_discover_roles():
    model = ToyModel()
    info = discover(model)
    role_map = {m.name: m.role for m in info.modules}
    assert role_map["layers.0.self_attn"] == "attention"
    assert role_map["layers.0.mlp"] == "mlp"
    assert role_map["lm_head"] == "head"
    assert role_map["embed"] == "embed"


def test_discover_with_dummy_input():
    model = ToyModel()
    dummy = torch.randint(0, 100, (1, 5))
    info = discover(model, dummy_input=dummy)
    shapes = {m.name: m.output_shape for m in info.modules if m.output_shape}
    assert "lm_head" in shapes
    assert shapes["lm_head"] == (1, 5, 100)


def test_discover_param_counts():
    model = ToyModel()
    info = discover(model)
    param_map = {m.name: m.param_count for m in info.modules}
    assert param_map["embed"] == 100 * 32
