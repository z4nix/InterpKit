"""Tests for Model.chat() and chat-template input dispatch on a real model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# ── Stub-model unit tests ────────────────────────────────────────────────


class _StubChatTokenizer:
    """Tokenizer with a deterministic chat template, suitable for unit tests."""

    pad_token_id = 0
    eos_token_id = 0
    chat_template = "fake"

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        tokenize=True,
    ):
        text = ""
        for m in messages:
            text += f"<|{m['role']}|>{m['content']}"
        if add_generation_prompt:
            text += "<|assistant|>"
        if not tokenize:
            return text
        ids = torch.tensor([[ord(c) % 50 + 1 for c in text]])
        if return_dict:
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return ids

    def decode(self, ids, *, skip_special_tokens=True):
        chars = [chr((int(i) - 1) % 50 + ord("a")) for i in ids.tolist()]
        return "".join(chars)


class _StubModel(nn.Module):
    """Generates *new_token_count* deterministic tokens after the prompt."""

    def __init__(self, new_token_count: int = 4) -> None:
        super().__init__()
        self.new_token_count = new_token_count
        self.last_kwargs: dict | None = None
        self.config = SimpleNamespace(is_encoder_decoder=False)

    def generate(self, *, input_ids, **kwargs):
        self.last_kwargs = kwargs
        new_tokens = torch.arange(
            10, 10 + self.new_token_count, dtype=input_ids.dtype, device=input_ids.device,
        ).unsqueeze(0)
        return torch.cat([input_ids, new_tokens], dim=-1)


def _make_stub_chat_model() -> object:
    from interpkit.core.model import Model

    arch_info = SimpleNamespace(
        modules=[],
        layer_names=[],
        is_tl_model=False,
        is_language_model=True,
        unembedding_name=None,
        has_lm_head=False,
        arch_family="stub",
        output_head_name=None,
    )
    return Model(
        _StubModel(new_token_count=4),
        tokenizer=_StubChatTokenizer(),
        arch_info=arch_info,
        device="cpu",
    )


def test_chat_string_input_round_trip():
    m = _make_stub_chat_model()
    out = m.chat("hello there", max_new_tokens=4)

    assert set(out.keys()) == {"prompt", "response", "messages", "input_ids", "output_ids"}
    assert out["prompt"].startswith("<|user|>hello there")
    assert out["prompt"].endswith("<|assistant|>")
    assert out["messages"] == [{"role": "user", "content": "hello there"}]
    assert isinstance(out["input_ids"], torch.Tensor)
    assert out["output_ids"].shape[-1] == out["input_ids"].shape[-1] + 4
    assert isinstance(out["response"], str) and out["response"]


def test_chat_with_system_prompt():
    m = _make_stub_chat_model()
    out = m.chat("hi", system="be terse")
    assert out["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    assert "<|system|>be terse" in out["prompt"]


def test_chat_messages_list_input():
    m = _make_stub_chat_model()
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]
    out = m.chat(msgs)
    assert out["messages"] == msgs
    assert "<|system|>be helpful" in out["prompt"]
    assert "<|user|>hello" in out["prompt"]


def test_chat_messages_list_with_system_kwarg_raises():
    m = _make_stub_chat_model()
    with pytest.raises(ValueError, match="system"):
        m.chat(
            [{"role": "user", "content": "hi"}],
            system="redundant",
        )


def test_chat_invalid_message_raises():
    m = _make_stub_chat_model()
    with pytest.raises(ValueError, match="must be a string"):
        m.chat(12345)  # type: ignore[arg-type]


def test_chat_no_chat_template_raises():
    from interpkit.core.model import Model

    class NoTemplate(_StubChatTokenizer):
        chat_template = None
        default_chat_template = None

    arch_info = SimpleNamespace(
        modules=[], layer_names=[], is_tl_model=False, is_language_model=True,
        unembedding_name=None, has_lm_head=False, arch_family="stub",
        output_head_name=None,
    )
    m = Model(_StubModel(), tokenizer=NoTemplate(), arch_info=arch_info, device="cpu")

    with pytest.raises(RuntimeError, match="chat template"):
        m.chat("hi")


def test_chat_no_tokenizer_raises():
    from interpkit.core.model import Model

    arch_info = SimpleNamespace(
        modules=[], layer_names=[], is_tl_model=False, is_language_model=True,
        unembedding_name=None, has_lm_head=False, arch_family="stub",
        output_head_name=None,
    )
    m = Model(_StubModel(), tokenizer=None, arch_info=arch_info, device="cpu")
    with pytest.raises(RuntimeError, match="tokenizer"):
        m.chat("hi")


def test_chat_no_generate_raises():
    from interpkit.core.model import Model

    class NoGenerate(nn.Module):
        config = SimpleNamespace(is_encoder_decoder=False)

    arch_info = SimpleNamespace(
        modules=[], layer_names=[], is_tl_model=False, is_language_model=True,
        unembedding_name=None, has_lm_head=False, arch_family="stub",
        output_head_name=None,
    )
    m = Model(NoGenerate(), tokenizer=_StubChatTokenizer(), arch_info=arch_info, device="cpu")
    with pytest.raises(RuntimeError, match="generate"):
        m.chat("hi")


def test_chat_sampling_kwargs_propagate():
    m = _make_stub_chat_model()
    m.chat("hi", do_sample=True, temperature=0.7, top_p=0.95)
    inner = m._model
    assert inner.last_kwargs is not None
    assert inner.last_kwargs["do_sample"] is True
    assert inner.last_kwargs["temperature"] == pytest.approx(0.7)
    assert inner.last_kwargs["top_p"] == pytest.approx(0.95)


def test_chat_greedy_omits_sampling_kwargs():
    m = _make_stub_chat_model()
    m.chat("hi", do_sample=False)
    inner = m._model
    assert inner.last_kwargs is not None
    assert inner.last_kwargs["do_sample"] is False
    # Not strictly required, but the helper should not bother passing them
    assert "temperature" not in inner.last_kwargs
    assert "top_p" not in inner.last_kwargs


# ── Smoke test against a real small chat model (network-gated) ───────────


@pytest.fixture(scope="module")
def smol_chat_model():
    from .conftest import load_or_skip

    return load_or_skip("HuggingFaceTB/SmolLM2-135M-Instruct")


def test_chat_smollm_generates_response(smol_chat_model):
    out = smol_chat_model.chat("Say hi.", max_new_tokens=8, do_sample=False)
    assert isinstance(out["prompt"], str) and out["prompt"]
    assert isinstance(out["response"], str)
    assert out["output_ids"].shape[-1] >= out["input_ids"].shape[-1]
