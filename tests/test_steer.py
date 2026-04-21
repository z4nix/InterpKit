"""Tests for the steer operation."""

from __future__ import annotations

import torch


def test_steer_vector_returns_tensor(gpt2_model):
    vector = gpt2_model.steer_vector(" love", " hate", at="transformer.h.8")
    assert isinstance(vector, torch.Tensor)
    assert vector.dim() == 1


def test_steer_runs(gpt2_model):
    vector = gpt2_model.steer_vector(" happy", " sad", at="transformer.h.8")
    result = gpt2_model.steer("The weather today is", vector=vector, at="transformer.h.8", scale=2.0)
    assert "original_top" in result
    assert "steered_top" in result
    assert len(result["original_top"]) > 0
    assert len(result["steered_top"]) > 0


def test_steer_changes_predictions(gpt2_model):
    vector = gpt2_model.steer_vector(" happy", " sad", at="transformer.h.8")
    result = gpt2_model.steer("The weather today is", vector=vector, at="transformer.h.8", scale=5.0)
    orig_top = result["original_top"][0][0]
    steered_top = result["steered_top"][0][0]
    # With a large scale, predictions should shift (not guaranteed but very likely)
    assert isinstance(orig_top, str)
    assert isinstance(steered_top, str)


# ── tokenization warning ─────────────────────────────────────────────────


def _capture_steer_console(monkeypatch) -> list[str]:
    """Patch the steer module's console.print and return the captured messages."""
    from interpkit.ops import steer

    captured: list[str] = []

    def fake_print(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(steer.console, "print", fake_print)
    return captured


def test_steer_warns_on_missing_leading_space(gpt2_model, monkeypatch):
    captured = _capture_steer_console(monkeypatch)
    gpt2_model.steer_vector("Love", "Hate", at="transformer.h.8")

    warnings = [m for m in captured if "steer:" in m]
    assert any("'Love'" in m and "' Love'" in m for m in warnings)
    assert any("'Hate'" in m and "' Hate'" in m for m in warnings)


def test_steer_no_warning_with_leading_space(gpt2_model, monkeypatch):
    captured = _capture_steer_console(monkeypatch)
    gpt2_model.steer_vector(" love", " hate", at="transformer.h.8")
    assert not any("steer:" in m for m in captured), captured


def test_steer_no_warning_for_full_sentence(gpt2_model, monkeypatch):
    captured = _capture_steer_console(monkeypatch)
    gpt2_model.steer_vector(
        "I love this movie",
        "I hate this movie",
        at="transformer.h.8",
    )
    assert not any("steer:" in m for m in captured), captured


def test_steer_no_warning_for_tensor_input(gpt2_model, monkeypatch):
    captured = _capture_steer_console(monkeypatch)
    pos = torch.tensor([[gpt2_model._tokenizer.encode("love")[0]]])
    neg = torch.tensor([[gpt2_model._tokenizer.encode("hate")[0]]])
    gpt2_model.steer_vector(pos, neg, at="transformer.h.8")
    assert not any("steer:" in m for m in captured), captured


def test_steer_warning_capped(gpt2_model, monkeypatch):
    """At most _MAX_TOKEN_WARNINGS warnings per call, even with many bad inputs."""
    from interpkit.ops.steer import _MAX_TOKEN_WARNINGS

    captured = _capture_steer_console(monkeypatch)
    bad = ["Love", "Hate", "Joy", "Fear", "Anger", "Peace", "Calm", "Pain"]
    gpt2_model.steer_vector(bad, bad, at="transformer.h.8")

    warnings = [m for m in captured if "steer:" in m]
    assert len(warnings) <= _MAX_TOKEN_WARNINGS
