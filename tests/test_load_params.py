"""Tests for load() parameter variants — device_map, tokenizer, dtype."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from interpkit.core.loader import load

# ── Helpers ──────────────────────────────────────────────────────


class _SimpleModel(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(100, hidden)
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs.get("input_ids", list(kwargs.values())[0])
        if isinstance(x, dict):
            x = list(x.values())[0]
        if x.dtype in (torch.long, torch.int):
            x = self.embed(x)
        return self.linear(x.view(x.shape[0], -1, 32).float())


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, text, return_tensors=None, **kwargs):
        tokens = text.split()
        ids = list(range(len(tokens)))
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{i}" for i in ids]

    def decode(self, ids):
        return " ".join(f"tok_{i}" for i in ids)

    def batch_decode(self, batch, **kwargs):
        return [self.decode(ids) for ids in batch]


# ══════════════════════════════════════════════════════════════════
# dtype mapping — exhaustive
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "dtype_str, expected",
    [
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("auto", "auto"),
    ],
)
def test_dtype_map_string_to_torch(dtype_str, expected):
    """Every _dtype_map entry resolves correctly before being passed to HF."""
    captured = {}

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        captured["torch_dtype"] = torch_dtype
        model = _SimpleModel()
        return model, _FakeTokenizer(), None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        load("gpt2", dtype=dtype_str, device="cpu")

    assert captured["torch_dtype"] == expected


def test_dtype_torch_object_passed_directly():
    captured = {}

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        captured["torch_dtype"] = torch_dtype
        return _SimpleModel(), _FakeTokenizer(), None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        load("gpt2", dtype=torch.bfloat16, device="cpu")

    assert captured["torch_dtype"] is torch.bfloat16


def test_dtype_unknown_string_raises():
    with pytest.raises(ValueError, match="Unknown dtype"):
        load("gpt2", dtype="int8", device="cpu")


def test_dtype_none_not_forwarded():
    captured = {}

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        captured["torch_dtype"] = torch_dtype
        return _SimpleModel(), _FakeTokenizer(), None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        load("gpt2", device="cpu")

    assert captured["torch_dtype"] is None


# ══════════════════════════════════════════════════════════════════
# device_map
# ══════════════════════════════════════════════════════════════════


def test_device_map_auto_forwarded_to_hf():
    captured = {}

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        captured["device_map"] = device_map
        captured["device"] = device
        return _SimpleModel(), _FakeTokenizer(), None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        load("gpt2", device_map="auto")

    assert captured["device_map"] == "auto"


def test_device_map_skips_device_default():
    """When device_map is set, device should remain None (not defaulted to cuda/cpu)."""
    captured = {}

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        captured["device"] = device
        return _SimpleModel(), _FakeTokenizer(), None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        load("gpt2", device_map="auto")

    assert captured["device"] is None


def test_device_map_infers_device_from_params():
    """With device_map, model._device should be inferred from model parameters."""

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        model = _SimpleModel()
        return model, _FakeTokenizer(), None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        model = load("gpt2", device_map="auto")

    assert model._device == torch.device("cpu")


# ══════════════════════════════════════════════════════════════════
# Custom tokenizer passthrough
# ══════════════════════════════════════════════════════════════════


def test_custom_tokenizer_used_for_hf_model():
    custom_tok = _FakeTokenizer()
    custom_tok.eos_token = "<custom_eos>"
    custom_tok.pad_token = "<custom_pad>"

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        return _SimpleModel(), tokenizer, None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        model = load("gpt2", tokenizer=custom_tok, device="cpu")

    assert model._tokenizer is custom_tok
    assert model._tokenizer.eos_token == "<custom_eos>"
    assert model._tokenizer.pad_token == "<custom_pad>"


def test_custom_tokenizer_pad_token_preserved():
    """If custom tokenizer already has pad_token, it should NOT be overwritten."""
    custom_tok = _FakeTokenizer()
    custom_tok.pad_token = "<my_pad>"
    custom_tok.eos_token = "<eos>"

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        return _SimpleModel(), tokenizer, None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        model = load("gpt2", tokenizer=custom_tok, device="cpu")

    assert model._tokenizer.pad_token == "<my_pad>"


def test_custom_tokenizer_pad_token_set_from_eos():
    """If custom tokenizer has pad_token=None, HF loader sets it to eos_token."""
    custom_tok = _FakeTokenizer()
    custom_tok.pad_token = None
    custom_tok.eos_token = "<eos>"

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return _SimpleModel(), tokenizer, None

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        model = load("gpt2", tokenizer=custom_tok, device="cpu")

    assert model._tokenizer.pad_token == "<eos>"


# ══════════════════════════════════════════════════════════════════
# nn.Module + tokenizer
# ══════════════════════════════════════════════════════════════════


def test_load_nn_module_with_tokenizer():
    """Pass raw nn.Module + tokenizer, verify tokenizer is stored."""
    tok = _FakeTokenizer()
    tok.pad_token = "<pad>"
    model = load(_SimpleModel(), tokenizer=tok, device="cpu")
    assert model._tokenizer is tok


def test_load_nn_module_no_tokenizer():
    """Load raw nn.Module without tokenizer — _tokenizer should be None."""
    model = load(_SimpleModel(), device="cpu")
    assert model._tokenizer is None


def test_load_nn_module_string_input_with_tokenizer():
    """When a tokenizer is provided for a raw nn.Module, string inputs should work."""
    tok = _FakeTokenizer()
    tok.pad_token = "<pad>"
    model = load(_SimpleModel(), tokenizer=tok, device="cpu")
    result = model.activations("hello world", at="embed")
    assert isinstance(result, torch.Tensor)


# ══════════════════════════════════════════════════════════════════
# Device handling
# ══════════════════════════════════════════════════════════════════


def test_load_nn_module_moves_to_device():
    """When device is specified for nn.Module, model should be moved there."""
    model = load(_SimpleModel(), device="cpu")
    assert model._device == torch.device("cpu") or str(model._device) == "cpu"


def test_load_default_device_when_no_device_or_map():
    """When neither device nor device_map is set, device is auto-detected."""
    from interpkit.core.loader import _resolve_device

    device = _resolve_device()
    assert device in ("cuda", "mps", "cpu")


# ══════════════════════════════════════════════════════════════════
# Combined parameters
# ══════════════════════════════════════════════════════════════════


def test_all_params_forwarded_together():
    captured = {}

    def _fake_load_from_hf(name, *, tokenizer, image_processor, device, torch_dtype, device_map):
        captured["name"] = name
        captured["device"] = device
        captured["torch_dtype"] = torch_dtype
        captured["device_map"] = device_map
        captured["tokenizer"] = tokenizer
        return _SimpleModel(), tokenizer, None

    custom_tok = _FakeTokenizer()
    custom_tok.pad_token = "<p>"

    with patch("interpkit.core.loader._load_from_hf", side_effect=_fake_load_from_hf):
        load(
            "my-model",
            tokenizer=custom_tok,
            dtype="bfloat16",
            device_map="auto",
        )

    assert captured["name"] == "my-model"
    assert captured["torch_dtype"] is torch.bfloat16
    assert captured["device_map"] == "auto"
    assert captured["device"] is None
    assert captured["tokenizer"] is custom_tok
