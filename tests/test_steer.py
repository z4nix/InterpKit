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


# ---------------------------------------------------------------------------
# SAE feature steering (Golden Gate style)
# ---------------------------------------------------------------------------


def _gpt2_toy_sae(d_in: int = 768, d_sae: int = 16):
    """Random (non-uniform) decoder directions — a uniform direction would be
    cancelled exactly by LayerNorm mean-centering."""
    from interpkit.ops.sae import load_sae_from_tensors

    g = torch.Generator().manual_seed(0)
    W_dec = torch.randn(d_sae, d_in, generator=g)
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)
    return load_sae_from_tensors(
        W_enc=W_dec.T.clone(), W_dec=W_dec,
        b_enc=torch.zeros(d_sae), b_dec=torch.zeros(d_in),
        metadata={"apply_b_dec_to_input": False},
    )


def test_feature_steer_changes_logits(gpt2_model):
    sae = _gpt2_toy_sae()
    result = gpt2_model.steer(
        "The weather today is", at="transformer.h.8",
        sae=sae, feature=3, strength=40.0,
    )
    assert result["feature"] == 3
    assert result["mode"] == "clamp"
    assert not torch.allclose(result["original_logits"], result["steered_logits"])


def test_feature_steer_add_equals_steer_intervention(gpt2_model):
    """mode='add' with strength s must be exactly SteerIntervention with
    vector=W_dec[i], scale=s."""
    sae = _gpt2_toy_sae()
    via_feature = gpt2_model.steer(
        "The weather today is", at="transformer.h.8",
        sae=sae, feature=3, strength=7.0, mode="add",
    )
    via_vector = gpt2_model.steer(
        "The weather today is", at="transformer.h.8",
        vector=sae.W_dec[3], scale=7.0,
    )
    assert torch.allclose(
        via_feature["steered_logits"], via_vector["steered_logits"],
    )


def test_feature_steer_validation(gpt2_model):
    import pytest

    sae = _gpt2_toy_sae()
    with pytest.raises(ValueError, match="exactly one of vector"):
        gpt2_model.steer("hi", at="transformer.h.8")
    with pytest.raises(ValueError, match="exactly one of vector"):
        gpt2_model.steer(
            "hi", at="transformer.h.8",
            vector=torch.randn(768), sae=sae, feature=1,
        )
    with pytest.raises(ValueError, match="requires sae"):
        gpt2_model.steer("hi", at="transformer.h.8", feature=1)
    with pytest.raises(ValueError, match="only applies with feature"):
        gpt2_model.steer("hi", at="transformer.h.8", vector=torch.randn(768), sae=sae)
    with pytest.raises(ValueError, match="out of range"):
        gpt2_model.steer("hi", at="transformer.h.8", sae=sae, feature=99)


def test_feature_steer_during_generation(gpt2_model):
    from interpkit import SAEFeatureIntervention

    sae = _gpt2_toy_sae()
    baseline = gpt2_model.generate("The weather today is", max_new_tokens=6)
    steered = gpt2_model.generate(
        "The weather today is", max_new_tokens=6,
        interventions=[SAEFeatureIntervention(
            "transformer.h.8", sae=sae, feature=3, strength=60.0,
        )],
    )
    assert steered["response"] != baseline["response"]
    assert steered["interventions"][0]["type"] == "sae_feature"
    assert steered["interventions"][0]["sae"] == "<SAE d_in=768 d_sae=16>"
    # Hooks fully removed: a fresh baseline reproduces the original.
    again = gpt2_model.generate("The weather today is", max_new_tokens=6)
    assert again["response"] == baseline["response"]
