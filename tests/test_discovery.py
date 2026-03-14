"""Tests for the auto-discovery module."""

from __future__ import annotations

import torch
import torch.nn as nn

from interpkit.core.discovery import discover


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.attention = nn.Linear(32, 32)
        self.mlp = nn.Linear(32, 32)
        self.head = nn.Linear(32, 100)

    def forward(self, x):
        x = self.embed(x)
        x = self.attention(x)
        x = self.mlp(x)
        return self.head(x)


def test_discover_toy_model():
    model = ToyModel()
    info = discover(model)
    assert len(info.modules) > 0
    names = {m.name for m in info.modules}
    assert "embed" in names
    assert "attention" in names
    assert "mlp" in names
    assert "head" in names


def test_discover_roles():
    model = ToyModel()
    info = discover(model)
    role_map = {m.name: m.role for m in info.modules}
    assert role_map["attention"] == "attention"
    assert role_map["mlp"] == "mlp"
    assert role_map["head"] == "head"
    assert role_map["embed"] == "embed"


def test_discover_with_dummy_input():
    model = ToyModel()
    dummy = torch.randint(0, 100, (1, 5))
    info = discover(model, dummy_input=dummy)
    shapes = {m.name: m.output_shape for m in info.modules if m.output_shape}
    assert "head" in shapes
    assert shapes["head"] == (1, 5, 100)


def test_discover_param_counts():
    model = ToyModel()
    info = discover(model)
    param_map = {m.name: m.param_count for m in info.modules}
    assert param_map["embed"] == 100 * 32
    assert param_map["attention"] == 32 * 32 + 32  # weight + bias
