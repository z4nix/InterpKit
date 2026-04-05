"""Tests for the manual registration API."""

from __future__ import annotations

import torch.nn as nn

from interpkit.core.registry import get_registration, register


def test_register_stores_layers():
    model = nn.Linear(10, 10)
    register(model, layers=["layer1", "layer2"])
    reg = get_registration(model)
    assert reg is not None
    assert reg.layers == ["layer1", "layer2"]


def test_register_stores_output_head():
    model = nn.Linear(10, 10)
    register(model, output_head="my_head")
    reg = get_registration(model)
    assert reg is not None
    assert reg.output_head == "my_head"


def test_register_no_registration():
    model = nn.Linear(10, 10)
    reg = get_registration(model)
    assert reg is None
