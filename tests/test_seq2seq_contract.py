"""Workstream F — seq2seq op contract test (two tiers).

Routing tier (every PR, tiny model): every (model, op) cell either returns a
non-None result or raises ``OperationNotSupportedForArchitecture`` with a
sensible message. It must NOT raise ``LensPipelineMismatch``,
``AttentionBackendUnavailable``, a generic ``RuntimeError``, or return a silent
``None``. This catches seq2seq routing regressions cheaply.

Numerics tier (``-m slow``, real models): the actual invariants
(Σ decompose ≈ residual, lens validation passes).
"""

from __future__ import annotations

import contextlib
import io

import pytest
import torch

import interpkit
from interpkit.core.exceptions import (
    AttentionBackendUnavailable,
    LensPipelineMismatch,
    OperationNotSupportedForArchitecture,
)

TINY_T5 = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
REAL_SEQ2SEQ = ["t5-small", "facebook/bart-base", "google/flan-t5-small"]

OPS = [
    "lens", "encoder_lens", "dla", "decompose",
    "attention_self", "attention_cross", "attention_encoder",
    "head_activations", "attribute", "ablate", "patch", "trace",
]

FORBIDDEN = (LensPipelineMismatch, AttentionBackendUnavailable)


def _first_layer_and_attn(arch):
    layer = next((li.name for li in arch.layer_infos if li.name), None)
    attn = next((li.attn_path for li in arch.layer_infos if li.attn_path), None)
    return layer, attn


def _call_op(m, op, text, text2):
    arch = m.arch_info
    layer, attn = _first_layer_and_attn(arch)
    if op == "lens":
        return m.lens(text)
    if op == "encoder_lens":
        return m.encoder_lens(text)
    if op == "dla":
        return m.dla(text)
    if op == "decompose":
        return m.decompose(text)
    if op == "attention_self":
        return m.attention(text, kind="self")
    if op == "attention_cross":
        return m.attention(text, kind="cross")
    if op == "attention_encoder":
        return m.attention(text, kind="encoder")
    if op == "head_activations":
        return m.head_activations(text, at=attn or layer)
    if op == "attribute":
        return m.attribute(text)
    if op == "ablate":
        return m.ablate(text, at=layer, method="zero")
    if op == "patch":
        return m.patch(text, text2, at=layer)
    if op == "trace":
        return m.trace(text, text2)
    raise AssertionError(f"unknown op {op}")


@pytest.mark.parametrize("op", OPS)
def test_seq2seq_routing_tiny_t5(op):
    """Routing tier on tiny-random T5 — fast, every PR."""
    try:
        m = interpkit.load(TINY_T5, device="cpu", validate_pipeline=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{TINY_T5} unavailable offline: {type(exc).__name__}")

    text, text2 = "Hello world", "Goodbye world"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = _call_op(m, op, text, text2)
    except OperationNotSupportedForArchitecture:
        return  # intentional loud gate — acceptable
    except FORBIDDEN as exc:
        pytest.fail(f"{op} raised forbidden {type(exc).__name__}: {exc}")
    except RuntimeError as exc:
        pytest.fail(f"{op} raised a generic RuntimeError (routing bug): {exc}")
    # Some ops legitimately return None as a soft 'not available' — the contract
    # forbids that for seq2seq routing, so flag it.
    assert result is not None, f"{op} returned a silent None (should PASS or raise OperationNotSupported)"


@pytest.mark.slow
@pytest.mark.parametrize("model_id", REAL_SEQ2SEQ)
def test_seq2seq_numerics_real(model_id):
    """Numerics tier on real seq2seq models — nightly / -m slow."""
    try:
        m = interpkit.load(model_id, device="cpu")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{model_id} unavailable offline: {type(exc).__name__}")

    text = "translate English to German: Hello world." if "t5" in model_id else "The capital of France is."
    with contextlib.redirect_stdout(io.StringIO()):
        d = m.decompose(text)
    resid = d["residual"].float()
    summed = torch.zeros_like(resid)
    for c in d["components"]:
        summed = summed + c["vector"].float()
    rel = (summed - resid).norm().item() / max(resid.norm().item(), 1e-9)
    assert rel < 1e-3, f"{model_id}: decompose Σ-invariant rel={rel:.3g}"

    with contextlib.redirect_stdout(io.StringIO()):
        assert m.lens(text) is not None
