"""Tests for max-activating-examples search (model.max_activating)."""

from __future__ import annotations

import pytest
import torch

TEXTS = [
    "The cat sat on the mat.",
    "Paris is the capital of France.",
    "I love programming in Python.",
    "The weather today is sunny and warm.",
    "Quantum physics is fascinating to study.",
    "She bought three apples at the market.",
    "The stock market crashed yesterday morning.",
    "Mountains are covered in deep snow.",
    "He plays the guitar every single evening.",
    "Coffee tastes better in the morning.",
]
AT = "transformer.h.6.mlp"
NEURON = 42


def _make_sae(d_in: int = 768, d_sae: int = 32):
    from interpkit.ops.sae import load_sae_from_tensors

    g = torch.Generator().manual_seed(0)
    return load_sae_from_tensors(
        W_enc=torch.randn(d_in, d_sae, generator=g) * 0.05,
        W_dec=torch.randn(d_sae, d_in, generator=g) * 0.05,
        b_enc=torch.zeros(d_sae),
        b_dec=torch.zeros(d_in),
    )


def test_neuron_schema_and_ordering(gpt2_model):
    result = gpt2_model.max_activating(TEXTS, at=AT, neuron=NEURON, top_k=8)
    assert result["unit"] == {"kind": "neuron", "at": AT, "index": NEURON}
    assert result["n_examples_scanned"] == len(TEXTS)
    assert result["n_positions_scanned"] > 0
    examples = result["examples"]
    assert len(examples) == 8
    scores = [e["score"] for e in examples]
    assert scores == sorted(scores, reverse=True)
    for i, e in enumerate(examples):
        assert e["rank"] == i
        assert e["text"] in TEXTS
        assert isinstance(e["token"], str)
        assert 0 <= e["context_offset"] < len(e["context_tokens"])
        assert len(e["context_scores"]) == len(e["context_tokens"])
        # The peak token's window score equals the example score.
        assert e["context_scores"][e["context_offset"]] == pytest.approx(
            e["score"], abs=1e-5,
        )


def test_neuron_agrees_with_bruteforce(gpt2_model):
    """The streaming scan must find the same global maximum as a direct
    per-text activation sweep."""
    best_score = float("-inf")
    best_text = None
    for text in TEXTS:
        acts = gpt2_model.activations(text, at=AT)
        row_max = float(acts[0, :, NEURON].max().item())
        if row_max > best_score:
            best_score = row_max
            best_text = text
    result = gpt2_model.max_activating(TEXTS, at=AT, neuron=NEURON, top_k=3)
    top = result["examples"][0]
    assert top["text"] == best_text
    assert top["score"] == pytest.approx(best_score, abs=1e-4)


def test_padding_does_not_contaminate(gpt2_model):
    """A short text batched with long ones (heavy padding) must score the
    same as when scanned alone — pad positions are masked out."""
    short = "Cats purr."
    alone = gpt2_model.max_activating([short], at=AT, neuron=NEURON, top_k=1)
    mixed = gpt2_model.max_activating(
        [short] + TEXTS, at=AT, neuron=NEURON, top_k=len(TEXTS) * 30,
        batch_size=11,
    )
    mixed_short = [e for e in mixed["examples"] if e["text"] == short]
    assert mixed_short
    assert mixed_short[0]["score"] == pytest.approx(
        alone["examples"][0]["score"], abs=1e-4,
    )


def test_deterministic(gpt2_model):
    a = gpt2_model.max_activating(TEXTS, at=AT, neuron=NEURON, top_k=5)
    b = gpt2_model.max_activating(TEXTS, at=AT, neuron=NEURON, top_k=5)
    assert [(e["text_idx"], e["position"], e["score"]) for e in a["examples"]] == [
        (e["text_idx"], e["position"], e["score"]) for e in b["examples"]
    ]


def test_sae_feature_mode(gpt2_model):
    sae = _make_sae()
    result = gpt2_model.max_activating(
        TEXTS, at="transformer.h.6", feature=3, sae=sae, top_k=5,
    )
    assert result["unit"]["kind"] == "sae_feature"
    assert len(result["examples"]) == 5
    assert all(e["score"] >= 0 for e in result["examples"])  # post-ReLU


def test_head_mode(gpt2_model):
    result = gpt2_model.max_activating(
        TEXTS, at="transformer.h.6.attn", head=2, top_k=5,
    )
    assert result["unit"]["kind"] == "head"
    assert all(e["score"] >= 0 for e in result["examples"])  # L2 norm


def test_unit_validation(gpt2_model):
    with pytest.raises(ValueError, match="exactly one"):
        gpt2_model.max_activating(TEXTS, at=AT)
    with pytest.raises(ValueError, match="exactly one"):
        gpt2_model.max_activating(TEXTS, at=AT, neuron=1, head=2)
    with pytest.raises(ValueError, match="requires sae"):
        gpt2_model.max_activating(TEXTS, at=AT, feature=1)
    with pytest.raises(ValueError, match="only applies with feature"):
        gpt2_model.max_activating(TEXTS, at=AT, neuron=1, sae=_make_sae())
    with pytest.raises(ValueError, match="out of range"):
        gpt2_model.max_activating(TEXTS, at=AT, neuron=10_000_000)
    with pytest.raises(ValueError, match="out of range"):
        gpt2_model.max_activating(
            TEXTS, at="transformer.h.6", feature=99, sae=_make_sae(d_sae=32),
        )
    with pytest.raises(KeyError, match="not found"):
        gpt2_model.max_activating(TEXTS, at="transformr.h.6", neuron=1)


def test_dataset_validation(gpt2_model):
    with pytest.raises(ValueError, match="non-empty list"):
        gpt2_model.max_activating([], at=AT, neuron=1)
    with pytest.raises(ValueError, match="max_examples is required"):
        gpt2_model.max_activating("hf:imdb", at=AT, neuron=1)
    with pytest.raises(ValueError, match="list of strings"):
        gpt2_model.max_activating("not-a-spec", at=AT, neuron=1)


def test_max_examples_caps_scan(gpt2_model):
    result = gpt2_model.max_activating(
        TEXTS, at=AT, neuron=NEURON, top_k=3, max_examples=4,
    )
    assert result["n_examples_scanned"] == 4
    assert all(e["text"] in TEXTS[:4] for e in result["examples"])
