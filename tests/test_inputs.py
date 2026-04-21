"""Tests for interpkit.core.inputs (input dispatch + image loader fallbacks)."""

from __future__ import annotations

import os
import sys
import tempfile

import pytest


def _make_test_png() -> str:
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(10, 20, 30))
    path = os.path.join(tempfile.gettempdir(), "interpkit_inputs_test.png")
    img.save(path)
    return path


def test_load_image_uses_image_processor_when_provided():
    """When an image_processor is passed, torchvision is never touched."""
    import torch

    from interpkit.core.inputs import _load_image

    class StubProcessor:
        def __call__(self, *, images, return_tensors):
            return {"pixel_values": torch.zeros(1, 3, 8, 8)}

    out = _load_image(_make_test_png(), image_processor=StubProcessor())
    assert isinstance(out, dict)
    assert "pixel_values" in out


def test_load_image_raises_runtime_error_without_torchvision(monkeypatch):
    """Without an image_processor and with torchvision unavailable, raise a clear RuntimeError."""
    from interpkit.core import inputs

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torchvision" or name.startswith("torchvision."):
            raise ImportError("No module named 'torchvision'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(sys.modules, "torchvision", None)
    if isinstance(__builtins__, dict):
        monkeypatch.setitem(__builtins__, "__import__", blocked_import)
    else:
        monkeypatch.setattr(__builtins__, "__import__", blocked_import)

    with pytest.raises(RuntimeError) as exc_info:
        inputs._load_image(_make_test_png())

    msg = str(exc_info.value)
    assert "torchvision" in msg
    assert "interpkit[vision]" in msg
    assert "image_processor" in msg
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_torchvision_required_msg_constant_is_used():
    """The exported constant should match the runtime message."""
    from interpkit.core.inputs import TORCHVISION_REQUIRED_MSG

    assert "torchvision" in TORCHVISION_REQUIRED_MSG
    assert "interpkit[vision]" in TORCHVISION_REQUIRED_MSG
    assert "image_processor" in TORCHVISION_REQUIRED_MSG


# ── chat message dispatch ─────────────────────────────────────────────────


def test_is_message_list_positive():
    from interpkit.core.inputs import _is_message_list

    assert _is_message_list([{"role": "user", "content": "hi"}])
    assert _is_message_list([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ])
    # extra keys are allowed
    assert _is_message_list([{"role": "user", "content": "hi", "name": "bob"}])


def test_is_message_list_negative():
    from interpkit.core.inputs import _is_message_list

    assert not _is_message_list([])
    assert not _is_message_list("hi")
    assert not _is_message_list([{"role": "user"}])
    assert not _is_message_list([{"content": "hi"}])
    assert not _is_message_list([{"role": 1, "content": "hi"}])
    assert not _is_message_list(["just a string"])


class _ChatTokenizer:
    """Minimal tokenizer stub with a hand-rolled chat template."""

    pad_token_id = 0
    eos_token_id = 0
    chat_template = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt=True,
        return_tensors=None,
        return_dict=False,
        tokenize=True,
    ):
        import torch

        self.calls.append({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": return_tensors,
            "return_dict": return_dict,
            "tokenize": tokenize,
        })
        if not tokenize:
            return "<|user|>" + messages[-1]["content"] + "<|assistant|>"
        ids = torch.arange(1, len(messages) * 3 + 1).unsqueeze(0)
        if return_dict:
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return ids


def test_prepare_input_dispatches_messages():
    import torch

    from interpkit.core.inputs import prepare_input

    tok = _ChatTokenizer()
    out = prepare_input(
        [{"role": "user", "content": "hi"}],
        tokenizer=tok,
        device="cpu",
    )
    assert isinstance(out, dict)
    assert "input_ids" in out
    assert "attention_mask" in out
    assert isinstance(out["input_ids"], torch.Tensor)
    assert tok.calls, "apply_chat_template should have been called"
    assert tok.calls[0]["add_generation_prompt"] is True


def test_prepare_input_messages_without_tokenizer_raises():
    from interpkit.core.inputs import prepare_input

    with pytest.raises(ValueError, match="no tokenizer"):
        prepare_input([{"role": "user", "content": "hi"}], tokenizer=None)


def test_prepare_input_messages_no_chat_template_raises():
    from interpkit.core.inputs import NO_CHAT_TEMPLATE_MSG, prepare_input

    class NoTemplate(_ChatTokenizer):
        chat_template = None
        default_chat_template = None

    with pytest.raises(ValueError) as exc_info:
        prepare_input(
            [{"role": "user", "content": "hi"}],
            tokenizer=NoTemplate(),
        )
    assert str(exc_info.value) == NO_CHAT_TEMPLATE_MSG


def test_prepare_input_messages_legacy_tokenizer_no_return_dict():
    """Tokenizers without return_dict support should still work via fallback."""
    import torch

    from interpkit.core.inputs import prepare_input

    class LegacyChatTok:
        chat_template = "fake"
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(
            self, messages, *, add_generation_prompt=True, return_tensors=None,
        ):
            return torch.tensor([[1, 2, 3, 4]])

    out = prepare_input(
        [{"role": "user", "content": "hi"}],
        tokenizer=LegacyChatTok(),
    )
    assert isinstance(out, dict)
    assert out["input_ids"].shape == (1, 4)
    assert out["attention_mask"].shape == (1, 4)
    assert (out["attention_mask"] == 1).all()


def test_prepare_pair_messages_pads_to_equal_length():
    import torch

    from interpkit.core.inputs import prepare_pair

    class LengthAwareTok(_ChatTokenizer):
        def apply_chat_template(
            self,
            messages,
            *,
            add_generation_prompt=True,
            return_tensors=None,
            return_dict=False,
            tokenize=True,
        ):
            length = len(messages[-1]["content"])
            ids = torch.arange(1, length + 1).unsqueeze(0)
            if return_dict:
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            return ids

    a, b = prepare_pair(
        [{"role": "user", "content": "abc"}],
        [{"role": "user", "content": "abcdef"}],
        tokenizer=LengthAwareTok(),
        device="cpu",
    )
    assert a["input_ids"].shape == b["input_ids"].shape
    assert a["input_ids"].shape[-1] == 6
    assert (a["attention_mask"][0, :3] == 1).all()
    assert (a["attention_mask"][0, 3:] == 0).all()


def test_prepare_pair_string_and_messages_routes_independently():
    """Mixed (string, messages) pair must not crash and must route each branch."""
    from interpkit.core.inputs import prepare_pair

    class TextTok(_ChatTokenizer):
        def __call__(self, raw, *, return_tensors="pt"):
            import torch

            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    a, b = prepare_pair(
        "hello",
        [{"role": "user", "content": "hi"}],
        tokenizer=TextTok(),
        device="cpu",
    )
    assert "input_ids" in a
    assert "input_ids" in b
