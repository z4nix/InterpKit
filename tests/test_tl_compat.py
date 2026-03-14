"""Tests for TransformerLens interop — name translation and model loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import torch
import torch.nn as nn

import mechkit
from mechkit.core.tl_compat import list_tl_hooks, to_native_name, to_tl_name


# ── Name translation tests ──────────────────────────────────────


def test_to_tl_name_gpt2_mlp():
    assert to_tl_name("transformer.h.8.mlp") == "blocks.8.mlp"


def test_to_tl_name_gpt2_attn():
    assert to_tl_name("transformer.h.8.attn") == "blocks.8.attn"


def test_to_tl_name_llama_self_attn():
    assert to_tl_name("model.layers.3.self_attn") == "blocks.3.attn"


def test_to_tl_name_llama_q_proj():
    assert to_tl_name("model.layers.3.self_attn.q_proj") == "blocks.3.attn.hook_q"


def test_to_tl_name_llama_k_proj():
    assert to_tl_name("model.layers.3.self_attn.k_proj") == "blocks.3.attn.hook_k"


def test_to_tl_name_bare_layer():
    assert to_tl_name("transformer.h.5") == "blocks.5"


def test_to_tl_name_unknown_passthrough():
    assert to_tl_name("some.random.module") == "some.random.module"


def test_to_native_name_with_arch_info(gpt2_model):
    native = to_native_name("blocks.8.mlp", gpt2_model.arch_info)
    assert native == "transformer.h.8.mlp"


def test_to_native_name_attn(gpt2_model):
    native = to_native_name("blocks.8.attn", gpt2_model.arch_info)
    assert native == "transformer.h.8.attn"


def test_to_native_name_bare_block(gpt2_model):
    native = to_native_name("blocks.8", gpt2_model.arch_info)
    assert native == "transformer.h.8"


def test_to_native_name_without_arch_info():
    native = to_native_name("blocks.5.mlp", None)
    # Without arch info, falls back to "blocks" prefix
    assert "5" in native
    assert "mlp" in native


def test_roundtrip_gpt2(gpt2_model):
    original = "transformer.h.8.mlp"
    tl = to_tl_name(original)
    back = to_native_name(tl, gpt2_model.arch_info)
    assert back == original


# ── list_tl_hooks tests ─────────────────────────────────────────


def test_list_tl_hooks_with_hook_dict():
    model = MagicMock()
    model.hook_dict = {"blocks.0.hook_resid_pre": None, "blocks.0.attn.hook_q": None}
    hooks = list_tl_hooks(model)
    assert "blocks.0.attn.hook_q" in hooks
    assert "blocks.0.hook_resid_pre" in hooks


def test_list_tl_hooks_no_hooks():
    model = nn.Linear(10, 10)
    hooks = list_tl_hooks(model)
    assert hooks == []


# ── HookedTransformer loading tests ─────────────────────────────


class _FakeTokenizer:
    """Minimal tokenizer mock for testing TL model loading."""

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


class _FakeHookedTransformer(nn.Module):
    """Mimics a HookedTransformer with hook_dict, cfg, and tokenizer."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.linear = nn.Linear(32, 32)
        self.hook_dict = {"blocks.0.hook_resid_pre": None}
        self.cfg = MagicMock()
        self.cfg.n_layers = 1
        self.tokenizer = _FakeTokenizer()

    def forward(self, x):
        if isinstance(x, dict):
            x = list(x.values())[0]
        if x.dtype in (torch.long, torch.int):
            x = self.embed(x)
        return self.linear(x.view(x.shape[0], -1, 32).float())


# Override class name to trigger detection
_FakeHookedTransformer.__name__ = "HookedTransformer"
_FakeHookedTransformer.__qualname__ = "HookedTransformer"


def test_load_hooked_transformer_detects_tl():
    fake_model = _FakeHookedTransformer()
    model = mechkit.load(fake_model, device="cpu")
    assert model.arch_info.is_tl_model is True


def test_load_hooked_transformer_extracts_tokenizer():
    fake_model = _FakeHookedTransformer()
    model = mechkit.load(fake_model, device="cpu")
    assert model._tokenizer is not None
    assert model._tokenizer.eos_token == "<eos>"


def test_load_hooked_transformer_sets_pad_token():
    fake_model = _FakeHookedTransformer()
    model = mechkit.load(fake_model, device="cpu")
    assert model._tokenizer.pad_token == "<eos>"


def test_load_regular_model_not_tl():
    regular = nn.Linear(10, 10)
    model = mechkit.load(regular, device="cpu")
    assert model.arch_info.is_tl_model is False


def test_load_hooked_transformer_with_explicit_tokenizer():
    fake_model = _FakeHookedTransformer()
    custom_tok = _FakeTokenizer()
    custom_tok.eos_token = "<custom>"
    model = mechkit.load(fake_model, tokenizer=custom_tok, device="cpu")
    assert model._tokenizer.eos_token == "<custom>"
