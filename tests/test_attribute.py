"""Tests for the attribute operation."""

from __future__ import annotations


def test_attribute_gpt2_runs(gpt2_model, capsys):
    gpt2_model.attribute("The capital of France is")
    captured = capsys.readouterr()
    assert "Attribution" in captured.out or "attribution" in captured.out.lower()


def test_attribute_gpt2_with_target(gpt2_model, capsys):
    gpt2_model.attribute("The capital of France is", target=0)
    captured = capsys.readouterr()
    assert len(captured.out) > 0
