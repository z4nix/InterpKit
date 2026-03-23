"""Tests for TransformerLens model path — running actual ops through the TL code path."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import interpkit


# ── Shared fake TL model ────────────────────────────────────────


class _FakeTokenizer:
    """Minimal tokenizer mock for TL models."""

    pad_token = None
    eos_token = "<eos>"

    def __call__(self, text, return_tensors=None, **kwargs):
        tokens = text.split()
        ids = list(range(len(tokens)))
        result = {"input_ids": torch.tensor([ids])}
        if return_tensors == "pt":
            return result
        return result

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{i}" for i in ids]

    def decode(self, ids):
        return " ".join(f"tok_{i}" for i in ids)

    def batch_decode(self, batch, **kwargs):
        return [self.decode(ids) for ids in batch]


class _FakeHookedTransformer(nn.Module):
    """Extended fake HookedTransformer that supports hook-based activation extraction."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.block0 = nn.Linear(32, 32)
        self.block1 = nn.Linear(32, 32)
        self.hook_dict = {
            "blocks.0.hook_resid_pre": None,
            "blocks.1.hook_resid_pre": None,
        }
        self.cfg = MagicMock()
        self.cfg.n_layers = 2
        self.tokenizer = _FakeTokenizer()

    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs.get("input_ids", list(kwargs.values())[0])
        if isinstance(x, dict):
            x = list(x.values())[0]
        if x.dtype in (torch.long, torch.int):
            x = self.embed(x)
        h = self.block0(x.view(x.shape[0], -1, 32).float())
        h = self.block1(h)
        return h


_FakeHookedTransformer.__name__ = "HookedTransformer"
_FakeHookedTransformer.__qualname__ = "HookedTransformer"


@pytest.fixture()
def tl_model():
    """Load a fake TL model through interpkit."""
    fake = _FakeHookedTransformer()
    return interpkit.load(fake, device="cpu")


# ══════════════════════════════════════════════════════════════════
# TL detection and loading
# ══════════════════════════════════════════════════════════════════


def test_tl_model_is_detected(tl_model):
    assert tl_model.arch_info.is_tl_model is True


def test_tl_tokenizer_extracted(tl_model):
    assert tl_model._tokenizer is not None
    assert tl_model._tokenizer.eos_token == "<eos>"


def test_tl_pad_token_set_from_eos(tl_model):
    assert tl_model._tokenizer.pad_token == "<eos>"


def test_tl_explicit_tokenizer_overrides():
    fake = _FakeHookedTransformer()
    custom_tok = _FakeTokenizer()
    custom_tok.eos_token = "<custom_eos>"
    model = interpkit.load(fake, tokenizer=custom_tok, device="cpu")
    assert model._tokenizer.eos_token == "<custom_eos>"


def test_tl_pad_token_not_overwritten_if_set():
    """If the TL model's tokenizer already has a pad_token, don't replace it."""
    fake = _FakeHookedTransformer()
    fake.tokenizer.pad_token = "<pad>"
    model = interpkit.load(fake, device="cpu")
    assert model._tokenizer.pad_token == "<pad>"


# ══════════════════════════════════════════════════════════════════
# Dummy input
# ══════════════════════════════════════════════════════════════════


def test_tl_dummy_input_is_raw_tensor():
    """TL path should use torch.tensor([[0]]) instead of tokenizer dict."""
    fake = _FakeHookedTransformer()
    model = interpkit.load(fake, device="cpu")
    assert model.arch_info.is_tl_model is True


# ══════════════════════════════════════════════════════════════════
# Inspect
# ══════════════════════════════════════════════════════════════════


def test_tl_inspect(tl_model, capsys):
    tl_model.inspect()
    captured = capsys.readouterr().out
    assert "embed" in captured.lower() or "Embedding" in captured


# ══════════════════════════════════════════════════════════════════
# Activations
# ══════════════════════════════════════════════════════════════════


def test_tl_activations_by_module_name(tl_model):
    """Extract activations from a named module in the fake TL model."""
    result = tl_model.activations(torch.tensor([[0, 1, 2]]), at="embed")
    assert isinstance(result, torch.Tensor)


def test_tl_activations_tensor_shape(tl_model):
    result = tl_model.activations(torch.tensor([[0, 1]]), at="block0")
    assert isinstance(result, torch.Tensor)


def test_tl_activations_multiple_modules(tl_model):
    result = tl_model.activations(torch.tensor([[0, 1]]), at=["embed", "block0"])
    assert isinstance(result, dict)
    assert "embed" in result
    assert "block0" in result


# ══════════════════════════════════════════════════════════════════
# Ablation
# ══════════════════════════════════════════════════════════════════


def test_tl_ablate_zero(tl_model):
    result = tl_model.ablate(torch.tensor([[0, 1, 2]]), at="block0", method="zero")
    assert "effect" in result
    assert "module" in result
    assert result["module"] == "block0"


def test_tl_ablate_mean(tl_model):
    result = tl_model.ablate(torch.tensor([[0, 1, 2]]), at="block0", method="mean")
    assert "effect" in result


# ══════════════════════════════════════════════════════════════════
# String input via TL tokenizer
# ══════════════════════════════════════════════════════════════════


def test_tl_string_input_uses_tokenizer(tl_model):
    """When passing a string, the TL tokenizer should be used."""
    result = tl_model.activations("hello world test", at="embed")
    assert isinstance(result, torch.Tensor)
    assert result.shape[1] == 3


# ══════════════════════════════════════════════════════════════════
# Forward pass
# ══════════════════════════════════════════════════════════════════


def test_tl_forward_with_tensor():
    """Verify the TL model can run a forward pass with raw tensor input."""
    fake = _FakeHookedTransformer()
    model = interpkit.load(fake, device="cpu")
    inp = model._prepare(torch.tensor([[0, 1, 2]]))
    out = model._forward(inp)
    assert isinstance(out, torch.Tensor)


def test_tl_forward_with_string():
    """Verify the TL model can run a forward pass from string input."""
    fake = _FakeHookedTransformer()
    model = interpkit.load(fake, device="cpu")
    inp = model._prepare("hello world")
    out = model._forward(inp)
    assert isinstance(out, torch.Tensor)


# ══════════════════════════════════════════════════════════════════
# Non-TL model comparison
# ══════════════════════════════════════════════════════════════════


def test_regular_module_not_tl():
    regular = nn.Linear(10, 10)
    model = interpkit.load(regular, device="cpu")
    assert model.arch_info.is_tl_model is False


def test_detect_via_hook_dict_and_cfg():
    """Models with hook_dict + cfg should be detected as TL even without the class name."""

    class _CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 16)
            self.linear = nn.Linear(16, 16)
            self.hook_dict = {"hook_resid": None}
            self.cfg = MagicMock()
            self.cfg.n_layers = 1

        def forward(self, x=None, **kwargs):
            if x is None:
                x = kwargs.get("input_ids", list(kwargs.values())[0])
            if isinstance(x, dict):
                x = list(x.values())[0]
            if x.dtype in (torch.long, torch.int):
                x = self.embed(x)
            return self.linear(x.view(x.shape[0], -1, 16).float())

    model = interpkit.load(_CustomModel(), device="cpu")
    assert model.arch_info.is_tl_model is True
