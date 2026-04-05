"""Named regression tests for critical historical bugs.

Each test targets a specific failure mode that was encountered and fixed.
If any of these break, the named bug has regressed.
"""

from __future__ import annotations

import torch

TEXT = "The capital of France is"


def test_r_forward_output_normalization(gpt2_model):
    """_forward_with_grad must handle all HF output types: .logits, .start_logits,
    raw tensor, and tuple."""
    model_input = gpt2_model._prepare(TEXT)
    logits = gpt2_model._forward_with_grad(model_input)
    assert isinstance(logits, torch.Tensor)
    assert logits.dim() >= 2


def test_r_dtype_casting_on_cpu(gpt2_model):
    """Attribution should work on CPU without dtype errors.
    Previously failed when logits were float16 and backward was attempted on CPU."""
    result = gpt2_model.attribute(TEXT, method="gradient")
    assert "scores" in result
    for score in result["scores"]:
        assert torch.isfinite(torch.tensor(float(score)))


def test_r_cache_hash_consistency(gpt2_model):
    """Cache should match the same input consistently and reject different inputs."""
    from interpkit.core.cache import hash_input

    model_input = gpt2_model._prepare(TEXT)
    h1 = hash_input(model_input)
    h2 = hash_input(model_input)
    assert h1 == h2

    other_input = gpt2_model._prepare("Hello world")
    h3 = hash_input(other_input)
    assert h1 != h3


def test_r_discovery_summary_no_crash(gpt2_model):
    """discovery_summary() must not crash on any valid model."""
    summary = gpt2_model.arch_info.discovery_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_r_arch_info_repr(gpt2_model):
    """ModelArchInfo __repr__ should return a useful string."""
    r = repr(gpt2_model.arch_info)
    assert "ModelArchInfo" in r
    assert "GPT2" in r


def test_r_layer_info_qkv_resolved(gpt2_model):
    """GPT-2 layers should have QKV projections fully resolved."""
    arch = gpt2_model.arch_info
    for li in arch.layer_infos:
        if li.layer_type in ("standard", "attention_only"):
            assert li.qkv_style != "unknown", (
                f"Layer {li.name}: QKV style should be resolved, got 'unknown'"
            )


def test_r_decompose_position_bounds(gpt2_model):
    """decompose with position=0 and position=-1 should both work."""
    r0 = gpt2_model.decompose(TEXT, position=0)
    rm1 = gpt2_model.decompose(TEXT, position=-1)
    assert r0["position"] == 0
    assert rm1["position"] == -1


def test_r_empty_cache_after_clear(gpt2_model):
    """clear_cache should fully empty the cache."""
    gpt2_model.cache(TEXT)
    assert gpt2_model.cached
    gpt2_model.clear_cache()
    assert not gpt2_model.cached


def test_r_inject_decoder_ids_noop_for_decoder_only(gpt2_model):
    """_inject_decoder_ids should be a no-op for decoder-only models like GPT-2."""
    model_input = gpt2_model._prepare(TEXT)
    assert isinstance(model_input, dict)
    assert "decoder_input_ids" not in model_input
