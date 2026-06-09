"""Tests for the attention operation."""

from __future__ import annotations


def test_attention_gpt2_returns_results(gpt2_model):
    results = gpt2_model.attention("The capital of France is")
    assert isinstance(results, list)
    assert len(results) > 0


def test_attention_gpt2_has_fields(gpt2_model):
    results = gpt2_model.attention("hello world")
    for r in results:
        assert "layer" in r
        assert "head" in r
        assert "top_pairs" in r
        assert "entropy" in r


def test_attention_gpt2_specific_layer(gpt2_model):
    results = gpt2_model.attention("hello world", layer=0)
    assert all(r["layer"] == 0 for r in results)


def test_attention_gpt2_specific_head(gpt2_model):
    results = gpt2_model.attention("hello world", layer=0, head=0)
    assert len(results) == 1
    assert results[0]["layer"] == 0
    assert results[0]["head"] == 0


def test_attention_resnet_raises_unsupported(resnet_model):
    """attention() on a CNN raises OperationNotSupportedForArchitecture.

    Pre-1.0 silently returned None which masked the architectural mismatch
    (F-001/F-002 family). Now we fail loud with a helpful suggestion.
    """
    import os
    import tempfile

    import pytest
    from PIL import Image

    from interpkit.core.exceptions import OperationNotSupportedForArchitecture

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = os.path.join(tempfile.gettempdir(), "test_attn.jpg")
    img.save(path)

    with pytest.raises(OperationNotSupportedForArchitecture, match="attention"):
        resnet_model.attention(path)


# ---------------------------------------------------------------------------
# N-003 — encoder-decoder attention extraction
# ---------------------------------------------------------------------------


def test_n003_gpt2_attention_kind_is_self(gpt2_model):
    """Causal-LM attention always reports attention_kind='self', regardless
    of what the user passes for ``kind`` (no encoder/cross to choose from)."""
    results = gpt2_model.attention("hello world")
    assert all(r["attention_kind"] == "self" for r in results)


def _load_tiny_t5_or_skip():
    """Load a tiny T5 random model for fast seq2seq attention tests."""

    from tests.conftest import load_or_skip
    return load_or_skip("hf-internal-testing/tiny-random-T5ForConditionalGeneration")


def test_n003_t5_attention_self_kind_returns_decoder_self():
    """T5 self-attention reads ``decoder_attentions`` (the decoder's own
    self-attention stack); pre-N-003 this raised AttentionBackendUnavailable
    because the code only looked for ``out.attentions``."""
    m = _load_tiny_t5_or_skip()
    results = m.attention("translate: hello world", kind="self")
    assert results is not None and len(results) > 0
    for r in results:
        assert r["attention_kind"] == "self"
        # Decoder self-attention on a 1-token decoder_input_ids stack →
        # weights shape ends in (..., 1, 1) but is at least square.
        assert r["weights"].dim() == 2


def test_n003_t5_attention_cross_returns_decoder_to_encoder():
    """``kind="cross"`` returns the decoder→encoder cross-attention tensor.
    Each row carries ``attention_kind="cross"`` so callers can distinguish
    it from decoder self-attention."""
    m = _load_tiny_t5_or_skip()
    results = m.attention("translate: hello world", kind="cross")
    assert results is not None and len(results) > 0
    assert all(r["attention_kind"] == "cross" for r in results)


def test_n003_t5_attention_encoder_returns_encoder_self():
    """``kind="encoder"`` returns encoder self-attention."""
    m = _load_tiny_t5_or_skip()
    results = m.attention("translate: hello world", kind="encoder")
    assert results is not None and len(results) > 0
    assert all(r["attention_kind"] == "encoder" for r in results)


def test_n003_invalid_kind_raises():
    """Unknown ``kind`` values raise ValueError with the valid options
    listed — never a silent fallthrough."""
    import pytest
    m = _load_tiny_t5_or_skip()
    with pytest.raises(ValueError, match="invalid"):
        m.attention("hello", kind="bogus")
