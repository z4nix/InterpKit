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


def test_attention_resnet_returns_none(resnet_model):
    from PIL import Image
    import tempfile, os

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = os.path.join(tempfile.gettempdir(), "test_attn.jpg")
    img.save(path)

    result = resnet_model.attention(path)
    assert result is None
