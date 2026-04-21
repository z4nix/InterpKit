"""Regression tests for robustness audit fixes.

This file covers the second-pass robustness audit that followed the
initial chat-model / SAE / steering / vision feedback work.  Each test
maps to a specific robustness gap that was fixed:

* Loader's ``AutoTokenizer`` no longer swallows arbitrary exceptions.
* ``run_attribute`` accepts chat-message lists without crashing.
* ``scan`` and ``report`` route message lists through their text paths.
* SAE device-mismatch is auto-corrected in ``run_dla`` (third callsite).
* The leading-space tokenization warning fires in
  ``run_contrastive_features`` too, with a configurable op label.
* ``dla(token=...)`` warning suggests the leading-space variant when it
  is a single token.
* CLI surfaces ``--sae-subfolder`` and a new ``chat`` command.

Most tests use stubs / monkeypatching to stay fast and offline; a few
opt into the shared ``gpt2_model`` fixture for end-to-end smoke checks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────
# 1. Loader — AutoTokenizer broad-except tightened
# ─────────────────────────────────────────────────────────────────────────


def test_loader_warns_on_unexpected_tokenizer_exception(monkeypatch, capsys):
    """Real (non-OSError/KeyError/ValueError) errors should now surface a warning."""
    import transformers

    from interpkit.core import loader

    captured: list[str] = []

    class _BadTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(transformers, "AutoTokenizer", _BadTokenizer)
    monkeypatch.setattr(transformers, "AutoConfig", _StubAutoConfig)
    monkeypatch.setattr(transformers, "AutoModel", _StubAutoModel)
    for name in (
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelForMaskedLM",
        "AutoModelForImageClassification",
    ):
        monkeypatch.setattr(transformers, name, _StubAutoModel)

    from rich.console import Console as _RichConsole

    real_print = _RichConsole.print

    def _capturing_print(self, *args, **kwargs):
        captured.append(" ".join(str(a) for a in args))
        return real_print(self, *args, **kwargs)

    monkeypatch.setattr(_RichConsole, "print", _capturing_print)

    model, tok, _ = loader._load_from_hf(
        "dummy/model",
        tokenizer=None,
        image_processor=None,
        device="cpu",
    )
    capsys.readouterr()

    assert tok is None
    assert any("RuntimeError" in m and "simulated unexpected failure" in m for m in captured), (
        captured
    )


def test_loader_silently_skips_value_error(monkeypatch):
    """Vision-only models legitimately raise (ValueError|OSError|KeyError) — silent skip is OK."""
    import transformers

    from interpkit.core import loader

    class _NoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ValueError("model does not have a tokenizer")

    monkeypatch.setattr(transformers, "AutoTokenizer", _NoTokenizer)
    monkeypatch.setattr(transformers, "AutoConfig", _StubAutoConfig)
    monkeypatch.setattr(transformers, "AutoModel", _StubAutoModel)
    for name in (
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelForMaskedLM",
        "AutoModelForImageClassification",
    ):
        monkeypatch.setattr(transformers, name, _StubAutoModel)

    _, tok, _ = loader._load_from_hf(
        "dummy/vision",
        tokenizer=None,
        image_processor=None,
        device="cpu",
    )
    assert tok is None


class _StubAutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return SimpleNamespace(architectures=[], is_encoder_decoder=False)


class _StubAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        m = nn.Linear(8, 8)
        m.config = SimpleNamespace(is_encoder_decoder=False)
        return m


# ─────────────────────────────────────────────────────────────────────────
# 2. Attribute — chat message list dispatch
# ─────────────────────────────────────────────────────────────────────────


def test_attribute_routes_messages_to_attribute_messages(monkeypatch):
    """The dispatch should pick ``_attribute_messages`` for chat-style lists."""
    from interpkit.ops import attribute

    captured: dict[str, Any] = {}

    def _fake_attribute_messages(model, messages, **kwargs):
        captured["messages"] = messages
        return {"tokens": ["a"], "scores": [1.0], "target": 0, "method": "fake"}

    monkeypatch.setattr(attribute, "_attribute_messages", _fake_attribute_messages)
    monkeypatch.setattr(
        attribute,
        "_attribute_text",
        lambda *a, **kw: pytest.fail("messages must not route to text"),
    )

    msgs = [{"role": "user", "content": "hi"}]
    attribute.run_attribute(model=object(), input_data=msgs)
    assert captured["messages"] == msgs


def test_attribute_text_still_routes_to_text(monkeypatch):
    from interpkit.ops import attribute

    seen: list[str] = []

    def _fake_attribute_text(model, text, **kwargs):
        seen.append(text)
        return {"tokens": ["a"], "scores": [1.0], "target": 0, "method": "fake"}

    monkeypatch.setattr(attribute, "_attribute_text", _fake_attribute_text)
    attribute.run_attribute(model=object(), input_data="hello world")
    assert seen == ["hello world"]


def test_attribute_messages_requires_tokenizer():
    from interpkit.ops.attribute import _attribute_messages

    fake_model = SimpleNamespace(_tokenizer=None)
    with pytest.raises(ValueError, match="No tokenizer"):
        _attribute_messages(fake_model, [{"role": "user", "content": "hi"}], target=None)


def test_attribute_chat_messages_end_to_end(gpt2_model):
    """Smoke test: run gradient attribution over a templated chat exchange.

    GPT-2 has no chat template by default — assigning a minimal template
    here keeps the test offline and verifies the new dispatch path runs
    a full backward pass without crashing.
    """
    tok = gpt2_model._tokenizer
    original_template = getattr(tok, "chat_template", None)
    tok.chat_template = (
        "{% for m in messages %}<{{m['role']}}>{{m['content']}}</{{m['role']}}>"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    try:
        result = gpt2_model.attribute(
            [{"role": "user", "content": "Hello, world"}],
            method="gradient",
        )
    finally:
        tok.chat_template = original_template

    assert "tokens" in result and "scores" in result
    assert len(result["tokens"]) == len(result["scores"])
    assert len(result["tokens"]) > 0


# ─────────────────────────────────────────────────────────────────────────
# 3. scan / report — chat message awareness
# ─────────────────────────────────────────────────────────────────────────


def test_scan_marks_message_input_type(gpt2_model):
    tok = gpt2_model._tokenizer
    original_template = getattr(tok, "chat_template", None)
    tok.chat_template = (
        "{% for m in messages %}<{{m['role']}}>{{m['content']}}</{{m['role']}}>"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    try:
        result = gpt2_model.scan([{"role": "user", "content": "hi"}])
    finally:
        tok.chat_template = original_template

    assert result["input_type"] == "messages"
    assert "prediction" in result


def test_report_chat_message_does_not_crash(gpt2_model, tmp_path):
    """report() with chat messages should populate sections + render HTML."""
    tok = gpt2_model._tokenizer
    original_template = getattr(tok, "chat_template", None)
    tok.chat_template = (
        "{% for m in messages %}<{{m['role']}}>{{m['content']}}</{{m['role']}}>"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    out = tmp_path / "report.html"
    try:
        result = gpt2_model.report(
            [{"role": "user", "content": "hi"}],
            save=str(out),
        )
    finally:
        tok.chat_template = original_template

    assert out.exists() and out.stat().st_size > 0
    assert "errors" in result
    assert isinstance(result["errors"], dict)


# ─────────────────────────────────────────────────────────────────────────
# 4. SAE device guard reused in run_dla
# ─────────────────────────────────────────────────────────────────────────


def test_ensure_sae_on_device_returns_self_when_already_on_device():
    from interpkit.ops.sae import SAE, _ensure_sae_on_device

    sae = SAE(
        W_enc=torch.randn(4, 3),
        W_dec=torch.randn(3, 4),
        b_enc=torch.zeros(3),
        b_dec=torch.zeros(4),
        d_in=4,
        d_sae=3,
    )
    same = _ensure_sae_on_device(sae, sae.W_enc.device)
    assert same is sae


def test_ensure_sae_on_device_moves_to_meta():
    from interpkit.ops.sae import SAE, _ensure_sae_on_device

    sae = SAE(
        W_enc=torch.randn(4, 3),
        W_dec=torch.randn(3, 4),
        b_enc=torch.zeros(3),
        b_dec=torch.zeros(4),
        d_in=4,
        d_sae=3,
        metadata={"hint": "preserved"},
    )
    moved = _ensure_sae_on_device(sae, torch.device("meta"))
    assert moved is not sae
    assert moved.W_enc.device.type == "meta"
    assert moved.W_dec.device.type == "meta"
    assert moved.b_enc.device.type == "meta"
    assert moved.b_dec.device.type == "meta"
    assert moved.metadata == {"hint": "preserved"}


def test_run_dla_uses_ensure_sae_on_device(monkeypatch):
    """Verify the helper is called from run_dla too (third callsite)."""
    import interpkit.ops.dla as dla_module
    from interpkit.ops import sae as sae_module

    calls: list[Any] = []

    real_helper = sae_module._ensure_sae_on_device

    def _spy(sae, target):
        calls.append((sae, target))
        return real_helper(sae, target)

    monkeypatch.setattr(sae_module, "_ensure_sae_on_device", _spy)

    src = dla_module.__loader__.get_source(dla_module.__name__)
    assert "_ensure_sae_on_device" in src, (
        "run_dla should reference the shared SAE-device helper"
    )


# ─────────────────────────────────────────────────────────────────────────
# 5. Leading-space tokenization warning shared between steer + features
# ─────────────────────────────────────────────────────────────────────────


def test_warn_if_leading_space_better_no_op_for_tensor():
    from interpkit.core.inputs import warn_if_leading_space_better

    counter = [0]
    warn_if_leading_space_better(
        tokenizer=None,
        text=torch.tensor([1, 2, 3]),
        op_label="features",
        role="positive",
        warned_count=counter,
    )
    assert counter == [0]


def test_warn_if_leading_space_better_no_op_for_leading_space():
    from interpkit.core.inputs import warn_if_leading_space_better

    counter = [0]
    warn_if_leading_space_better(
        tokenizer=_StubBpeTokenizer(),
        text=" love",
        op_label="features",
        role="positive",
        warned_count=counter,
    )
    assert counter == [0]


def test_warn_if_leading_space_better_uses_provided_console():
    from interpkit.core.inputs import warn_if_leading_space_better

    captured: list[str] = []

    class _Console:
        def print(self, msg):
            captured.append(str(msg))

    counter = [0]
    warn_if_leading_space_better(
        tokenizer=_StubBpeTokenizer(),
        text="love",
        op_label="features",
        role="positive",
        warned_count=counter,
        console=_Console(),
    )
    assert counter == [1]
    assert any("features:" in c and "'love'" in c and "' love'" in c for c in captured), (
        captured
    )


def test_warn_if_leading_space_better_caps_at_max_warnings():
    from interpkit.core.inputs import (
        MAX_LEADING_SPACE_WARNINGS,
        warn_if_leading_space_better,
    )

    captured: list[str] = []

    class _Console:
        def print(self, msg):
            captured.append(str(msg))

    counter = [0]
    for word in ["love", "hate", "joy", "fear", "anger", "peace", "calm", "pain"]:
        warn_if_leading_space_better(
            tokenizer=_StubBpeTokenizer(),
            text=word,
            op_label="features",
            role="positive",
            warned_count=counter,
            console=_Console(),
        )

    assert len(captured) <= MAX_LEADING_SPACE_WARNINGS


def test_contrastive_features_emits_features_label_warning(gpt2_model, monkeypatch):
    from interpkit.ops import sae as sae_module

    captured: list[str] = []
    real_print = sae_module.console.print

    def _capturing_print(*args, **kwargs):
        if args:
            captured.append(str(args[0]))
        return real_print(*args, **kwargs)

    monkeypatch.setattr(sae_module.console, "print", _capturing_print)

    sae_obj = sae_module.SAE(
        W_enc=torch.randn(768, 64),
        W_dec=torch.randn(64, 768),
        b_enc=torch.zeros(64),
        b_dec=torch.zeros(768),
        d_in=768,
        d_sae=64,
    )

    sae_module.run_contrastive_features(
        gpt2_model,
        ["Hate"],
        ["Volkswagen"],
        at="transformer.h.8",
        sae=sae_obj,
        top_k=4,
        print_results=False,
    )

    assert any("features:" in m and "'Hate'" in m for m in captured), captured
    assert any("features:" in m and "'Volkswagen'" in m for m in captured), captured


class _StubBpeTokenizer:
    """Mimics a BPE tokenizer where leading-space ⇒ single token."""

    def encode(self, text: str, *, add_special_tokens: bool = True):
        if text.startswith(" "):
            return [99]
        if not text:
            return []
        return [ord(c) for c in text]


# ─────────────────────────────────────────────────────────────────────────
# 6. dla(token=str) leading-space tip
# ─────────────────────────────────────────────────────────────────────────


def test_dla_target_token_warning_suggests_leading_space(gpt2_model):
    """``Volkswagen`` tokenizes to multiple subwords but ``" Volkswagen"`` is one token."""
    with pytest.warns(UserWarning, match=r"Tip: pass token=' Volkswagen'"):
        gpt2_model.dla("I drive a", token="Volkswagen", top_k=3)


def test_dla_target_token_warning_no_tip_when_already_spaced(gpt2_model):
    """Already-spaced targets shouldn't get the tip suffix."""
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        gpt2_model.dla("I drive a", token=" Volkswagen", top_k=3)

    tip_warnings = [str(w.message) for w in caught if "Tip: pass token=" in str(w.message)]
    assert not tip_warnings, tip_warnings


# ─────────────────────────────────────────────────────────────────────────
# 7. CLI surface area — chat command + --sae-subfolder
# ─────────────────────────────────────────────────────────────────────────


def test_cli_features_has_sae_subfolder_option():
    from typer.testing import CliRunner

    from interpkit.cli.main import app

    runner = CliRunner()
    res = runner.invoke(app, ["features", "--help"])
    assert res.exit_code == 0
    assert "--sae-subfolder" in res.output


def test_cli_dla_has_sae_subfolder_option():
    from typer.testing import CliRunner

    from interpkit.cli.main import app

    runner = CliRunner()
    res = runner.invoke(app, ["dla", "--help"])
    assert res.exit_code == 0
    assert "--sae-subfolder" in res.output


def test_cli_chat_command_registered():
    from typer.testing import CliRunner

    from interpkit.cli.main import app

    runner = CliRunner()
    res = runner.invoke(app, ["chat", "--help"])
    assert res.exit_code == 0
    assert "--system" in res.output
    assert "--max-new-tokens" in res.output


# ─────────────────────────────────────────────────────────────────────────
# 8. SAE config download narrowed except
# ─────────────────────────────────────────────────────────────────────────


def test_download_config_returns_empty_on_entry_not_found(monkeypatch):
    from huggingface_hub.utils import EntryNotFoundError

    from interpkit.ops import sae as sae_module

    def _missing(*args, **kwargs):
        raise EntryNotFoundError("no cfg")

    monkeypatch.setattr(sae_module, "hf_hub_download" if False else "_download_config", sae_module._download_config)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _missing)

    cfg = sae_module._download_config("nonexistent/repo")
    assert cfg == {}


def test_download_config_warns_on_corrupt_json(monkeypatch, tmp_path):
    """JSON parse errors now print a warning instead of being silently swallowed."""
    from interpkit.ops import sae as sae_module

    bad = tmp_path / "cfg.json"
    bad.write_text("{this is not json")

    def _fake_download(*args, **kwargs):
        return str(bad)

    captured: list[str] = []

    class _Console:
        def print(self, msg, *args, **kwargs):
            captured.append(str(msg))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_download)
    monkeypatch.setattr(sae_module, "console", _Console())

    cfg = sae_module._download_config("dummy/repo")
    assert cfg == {}
    assert any("sae:" in m and "cfg.json" in m for m in captured), captured


# ─────────────────────────────────────────────────────────────────────────
# 9. Chat-message lists must not be misread as batches by ops that take
#    "single example or list of examples" (steer_vector, contrastive_features,
#    find_circuit).  Regression for the cryptic
#    "ValueError: You must specify exactly one of input_ids or inputs_embeds"
#    that surfaced across every chat-tuned model.
# ─────────────────────────────────────────────────────────────────────────


def test_normalize_input_group_string_is_wrapped():
    from interpkit.core.inputs import normalize_input_group

    assert normalize_input_group("hello") == ["hello"]


def test_normalize_input_group_chat_messages_stay_one_example():
    from interpkit.core.inputs import normalize_input_group

    msgs = [{"role": "user", "content": "hi"}]
    out = normalize_input_group(msgs)
    assert out == [msgs]
    assert len(out) == 1 and out[0] is msgs


def test_normalize_input_group_chat_messages_with_extra_keys():
    """Forward-compat extra keys (e.g. tool calls) must still be one example."""
    from interpkit.core.inputs import normalize_input_group

    msgs = [
        {"role": "system", "content": "be brief", "name": "sys"},
        {"role": "user", "content": "hi"},
    ]
    out = normalize_input_group(msgs)
    assert out == [msgs]


def test_normalize_input_group_string_batch_passes_through():
    from interpkit.core.inputs import normalize_input_group

    batch = ["one", "two", "three"]
    assert normalize_input_group(batch) is batch


def test_normalize_input_group_list_of_chats_passes_through():
    from interpkit.core.inputs import normalize_input_group

    chat_a = [{"role": "user", "content": "hi"}]
    chat_b = [{"role": "user", "content": "bye"}]
    out = normalize_input_group([chat_a, chat_b])
    assert out == [chat_a, chat_b]


def test_normalize_input_group_tensor_is_wrapped():
    from interpkit.core.inputs import normalize_input_group

    t = torch.zeros(1, 4)
    out = normalize_input_group(t)
    assert len(out) == 1 and out[0] is t


def test_steer_vector_accepts_chat_messages(gpt2_model):
    """Calling steer_vector with chat-message lists must not iterate per dict.

    GPT-2 has no chat template, so we monkeypatch a minimal one to keep the
    test fast and deterministic.  The point of the regression is purely
    that ``normalize_input_group`` keeps ``[{"role":..., "content":...}]``
    as a single example rather than treating each dict as its own input.
    """
    tok = gpt2_model._tokenizer
    saved = getattr(tok, "chat_template", None)
    try:
        tok.chat_template = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
        pos = [{"role": "user", "content": "I love this."}]
        neg = [{"role": "user", "content": "I hate this."}]
        v = gpt2_model.steer_vector(pos, neg, at="transformer.h.0")
        assert isinstance(v, torch.Tensor) and v.dim() == 1
        # Hidden size for gpt2 is 768
        assert v.shape[0] == 768
    finally:
        if saved is None:
            tok.chat_template = None
        else:
            tok.chat_template = saved


def test_contrastive_features_treats_bare_chat_as_single_example(monkeypatch):
    """A bare chat-message list must be normalised to one example.

    Stub ``run_activations`` so we can assert exactly which inputs the
    contrastive op iterated over without paying for a real SAE.
    """
    from interpkit.ops import sae as sae_module

    seen: list[Any] = []

    def _stub_activations(_model, inp, **_kw):
        seen.append(inp)
        return torch.zeros(1, 4, 8)  # (batch, seq, hidden)

    class _FakeSAE:
        d_in = 8
        d_sae = 16
        W_enc = torch.zeros(8, 16)

        def forward(self, x):
            return torch.zeros(x.shape[0], 16), x

    monkeypatch.setattr("interpkit.ops.activations.run_activations", _stub_activations)
    monkeypatch.setattr(sae_module, "_ensure_sae_on_device", lambda s, _d: s)

    pos_chat = [{"role": "user", "content": "I love cats."}]
    neg_chat = [{"role": "user", "content": "I hate cats."}]

    out = sae_module.run_contrastive_features(
        model=SimpleNamespace(_tokenizer=None, _device="cpu"),
        positive_inputs=pos_chat,
        negative_inputs=neg_chat,
        at="layer.0",
        sae=_FakeSAE(),
        top_k=2,
        print_results=False,
    )
    assert "top_differential_features" in out
    assert out["num_positive"] == 1 and out["num_negative"] == 1
    # Exactly two run_activations calls — one per side, each receiving the
    # whole chat list (NOT one call per message dict).
    assert len(seen) == 2
    assert seen[0] is pos_chat
    assert seen[1] is neg_chat


def test_find_circuit_normalizes_chat_messages(monkeypatch):
    """run_find_circuit must normalise a chat list to a single (clean,corrupt) pair.

    Pre-fix it iterated each message dict as its own pair, which both
    crashed and was semantically meaningless.  Stub the heavy bits so the
    test stays fast.
    """
    from interpkit.ops import find_circuit as fc_module

    captured_pairs: list[tuple[Any, Any]] = []

    class _StubArch:
        layer_names = ["layer.0"]
        attention_paths = []
        mlp_paths = []
        num_attention_heads = 0
        unembedding_name = None

        def layers(self):
            return []

    class _StubModel:
        arch_info = _StubArch()
        _tokenizer = None
        _device = "cpu"
        _model = SimpleNamespace()

        def _prepare_pair(self, c, r):
            captured_pairs.append((c, r))
            # Return tensors so _forward can produce logits
            return (
                {"input_ids": torch.zeros(1, 1, dtype=torch.long)},
                {"input_ids": torch.zeros(1, 1, dtype=torch.long)},
            )

        def _forward(self, _x):
            return torch.zeros(1, 1, 4)

    # Bypass the heavy circuit search — we only care about input normalisation.
    def _early_return(*_a, **_kw):
        return {
            "circuit": [],
            "excluded": [],
            "verification": {"circuit_effect": 0.0, "faithfulness": 0.0},
            "threshold": 0.0,
        }

    monkeypatch.setattr(fc_module, "_search_circuit", _early_return, raising=False)

    chat_a = [{"role": "user", "content": "Paris"}]
    chat_b = [{"role": "user", "content": "Berlin"}]

    # Direct call so we can intercept before render
    from interpkit.core.inputs import normalize_input_group

    cleans = normalize_input_group(chat_a)
    corrupteds = normalize_input_group(chat_b)
    assert cleans == [chat_a]
    assert corrupteds == [chat_b]
    assert len(cleans) == len(corrupteds) == 1


# ─────────────────────────────────────────────────────────────────────────
# 10. ``python -m interpkit`` entry point
# ─────────────────────────────────────────────────────────────────────────


def test_python_dash_m_interpkit_module_imports():
    """``python -m interpkit`` must dispatch to the same Typer app as the
    console script.  Importing the module exposes ``main`` and ``app``."""
    import importlib

    mod = importlib.import_module("interpkit.__main__")
    assert hasattr(mod, "main") and callable(mod.main)
    from interpkit.cli.main import app as expected_app
    assert mod.app is expected_app


def test_python_dash_m_interpkit_help_runs():
    """End-to-end smoke: ``python -m interpkit --help`` exits 0."""
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "-m", "interpkit", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "interpkit" in (res.stdout + res.stderr).lower()
