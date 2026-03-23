"""Tests for the inspect operation."""

from __future__ import annotations


def test_inspect_gpt2_runs(gpt2_model, capsys):
    gpt2_model.inspect()
    captured = capsys.readouterr()
    assert "GPT2" in captured.out or "gpt2" in captured.out.lower()


def test_inspect_gpt2_has_modules(gpt2_model):
    arch = gpt2_model.arch_info
    assert len(arch.modules) > 0


def test_inspect_gpt2_detects_architecture(gpt2_model):
    arch = gpt2_model.arch_info
    assert arch.arch_family is not None
    assert arch.num_layers is not None
    assert arch.hidden_size is not None


def test_inspect_gpt2_detects_roles(gpt2_model):
    arch = gpt2_model.arch_info
    roles = {m.role for m in arch.modules if m.role is not None}
    assert "attention" in roles or "mlp" in roles


def test_inspect_gpt2_detects_lm_head(gpt2_model):
    arch = gpt2_model.arch_info
    assert arch.has_lm_head
    assert arch.unembedding_name is not None


def test_inspect_gpt2_detects_layers(gpt2_model):
    arch = gpt2_model.arch_info
    assert len(arch.layer_names) > 0


def test_inspect_resnet_runs(resnet_model, capsys):
    resnet_model.inspect()
    captured = capsys.readouterr()
    assert "resnet" in captured.out.lower() or "ResNet" in captured.out


def test_inspect_resnet_no_lm_head(resnet_model):
    arch = resnet_model.arch_info
    # ResNet has a classifier head, but it should not be detected as a language model
    assert not arch.is_language_model
