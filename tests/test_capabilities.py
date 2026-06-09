"""Capability-based op gating + structural mechanism taxonomy.

These pin the architecture-compatibility rework: op support is decided by
structurally-detected *capabilities* (``has_unembedding`` /
``has_residual_stream`` / ``has_attention`` / ``is_generative``), not by a
fixed family enum. This is what lets a novel HF architecture with the right
shape work without per-model code, and lets pure-SSM / hybrid models be
described accurately and fail loud where attention is absent.
"""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import interpkit
from interpkit.core.arch import ArchFamily
from interpkit.core.arch.probe import _classify_block_mechanism
from interpkit.core.arch.types import ArchInfo, BlockSpec, LayerInfo
from interpkit.core.exceptions import OperationNotSupportedForArchitecture
from interpkit.core.support_matrix import Requires, check_op_supported

# ---------------------------------------------------------------------------
# Capability properties on ArchInfo
# ---------------------------------------------------------------------------


def test_has_unembedding_requires_head_and_not_encoder_only():
    assert ArchInfo(family=ArchFamily.CAUSAL_LM, head_path="lm_head").has_unembedding
    # No head → no unembedding.
    assert not ArchInfo(family=ArchFamily.CAUSAL_LM).has_unembedding
    # ENCODER_ONLY is the resolver's verdict that there is no valid
    # unembedding, even if a stray head_path was resolved.
    assert not ArchInfo(family=ArchFamily.ENCODER_ONLY, head_path="x").has_unembedding


def test_has_residual_stream_distinguishes_plain_vs_residual_blocks():
    res = ArchInfo(blocks=[BlockSpec(path="b.0", has_residual=True)])
    plain = ArchInfo(blocks=[BlockSpec(path="b.0", has_residual=False)])
    assert res.has_residual_stream
    assert not plain.has_residual_stream
    assert not ArchInfo().has_residual_stream  # no blocks


def test_has_attention_from_layer_infos():
    with_attn = ArchInfo(layer_infos=[LayerInfo(name="l.0", index=0, attn_path="l.0.attn")])
    without = ArchInfo(layer_infos=[LayerInfo(name="l.0", index=0, attn_path=None)])
    assert with_attn.has_attention
    assert not without.has_attention


def test_is_generative_only_for_causal_and_seq2seq():
    assert ArchInfo(family=ArchFamily.CAUSAL_LM).is_generative
    assert ArchInfo(family=ArchFamily.SEQ2SEQ_LM).is_generative
    assert not ArchInfo(family=ArchFamily.MLM).is_generative
    assert not ArchInfo(family=ArchFamily.VISION_TRANSFORMER).is_generative


# ---------------------------------------------------------------------------
# Capability gating is decoupled from the family enum
# ---------------------------------------------------------------------------


def test_unknown_family_but_capable_is_supported():
    """A model the classifier couldn't name (UNKNOWN) but which structurally
    has a head + attention + a residual stream is still supported for
    lens/dla/attention — the whole point of capability gating."""
    arch = ArchInfo(
        family=ArchFamily.UNKNOWN,
        head_path="lm_head",
        blocks=[BlockSpec(path="b.0", has_residual=True, has_attention=True)],
        layer_infos=[LayerInfo(name="b.0", index=0, attn_path="b.0.attn")],
    )
    # None of these should raise despite family == UNKNOWN.
    for op in ("lens", "dla", "decompose", "attention", "qk_scores", "head_activations"):
        check_op_supported(op, arch)


def test_no_attention_model_fails_loud_with_capability_message():
    """A pure-recurrent / SSM model (no attention layers) is refused for
    attention ops with a message naming the missing capability."""
    arch = ArchInfo(
        family=ArchFamily.CAUSAL_LM,  # e.g. MambaForCausalLM
        head_path="lm_head",
        blocks=[BlockSpec(path="b.0", has_residual=True, mechanism="ssm")],
        layer_infos=[LayerInfo(name="b.0", index=0, attn_path=None)],
    )
    # Mechanism-agnostic ops still run.
    for op in ("lens", "dla", "decompose"):
        check_op_supported(op, arch)
    # Attention ops fail loud, referencing the missing capability.
    for op in ("attention", "qk_scores", "ov_scores", "head_activations"):
        with pytest.raises(OperationNotSupportedForArchitecture, match="attention layer"):
            check_op_supported(op, arch)


def test_requires_missing_reports_each_unmet_capability():
    arch = ArchInfo(family=ArchFamily.UNKNOWN)  # nothing detected
    assert Requires(unembedding=True).missing(arch) == ["a usable output head (unembedding)"]
    assert Requires(attention=True).missing(arch) == ["at least one attention layer"]
    assert Requires().missing(arch) == []


# ---------------------------------------------------------------------------
# Structural mechanism classification (no model-class lists)
# ---------------------------------------------------------------------------


def _attn_block(d=8):
    class _Attn(nn.Module):
        def __init__(s):
            super().__init__()
            s.q_proj = nn.Linear(d, d)
            s.k_proj = nn.Linear(d, d)
            s.v_proj = nn.Linear(d, d)

    class _Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.norm = nn.LayerNorm(d)
            s.attn = _Attn()

    return _Block()


def _ssm_block(d=8):
    class _Mixer(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv1d = nn.Conv1d(d, d, 3, groups=d)
            s.in_proj = nn.Linear(d, 2 * d)
            s.out_proj = nn.Linear(d, d)
            s.A_log = nn.Parameter(torch.randn(d, d))
            s.D = nn.Parameter(torch.randn(d))

    class _Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.norm = nn.LayerNorm(d)
            s.mixer = _Mixer()

    return _Block()


def _recurrent_block(d=8):
    class _RGLRU(nn.Module):
        def __init__(s):
            super().__init__()
            s.gate = nn.Linear(d, d)

    class _Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.norm = nn.LayerNorm(d)
            s.rg_lru = _RGLRU()

    return _Block()


def _conv_block(d=8):
    class _Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.c = nn.Conv2d(d, d, 3, padding=1)
            s.r = nn.ReLU()

    return _Block()


def _mlp_block(d=8):
    class _Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc1 = nn.Linear(d, 4 * d)
            s.fc2 = nn.Linear(4 * d, d)

    return _Block()


@pytest.mark.parametrize(
    "factory,expected",
    [
        (_attn_block, "attention"),
        (_ssm_block, "ssm"),
        (_recurrent_block, "recurrent"),
        (_conv_block, "conv"),
        (_mlp_block, "mlp"),
    ],
)
def test_classify_block_mechanism(factory, expected):
    assert _classify_block_mechanism(factory()) == expected


# ---------------------------------------------------------------------------
# Real pure-SSM model end-to-end (skip if unavailable offline)
# ---------------------------------------------------------------------------

TINY_MAMBA = "hf-internal-testing/tiny-random-MambaForCausalLM"


@pytest.fixture(scope="module")
def tiny_mamba():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return interpkit.load(TINY_MAMBA, device="cpu")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"{TINY_MAMBA} unavailable: {type(exc).__name__}")


def test_mamba_blocks_classified_ssm(tiny_mamba):
    assert tiny_mamba.arch_info.block_mechanisms
    assert all(m == "ssm" for m in tiny_mamba.arch_info.block_mechanisms)
    assert tiny_mamba.arch_info.has_attention is False


def test_mamba_mechanism_agnostic_ops_run(tiny_mamba):
    """lens projects the residual stream through the head — works on an SSM."""
    result = tiny_mamba.lens("hello world", position=-1)
    assert result is not None and len(result) > 0


def test_mamba_attention_ops_fail_loud(tiny_mamba):
    with pytest.raises(OperationNotSupportedForArchitecture, match="attention layer"):
        check_op_supported("attention", tiny_mamba.arch_info)
