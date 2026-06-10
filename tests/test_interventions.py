"""Model-free unit tests for core.interventions (hook compile + writeback math)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from interpkit.core.interventions import (
    AblateIntervention,
    CaptureProbe,
    FnIntervention,
    GenerationContext,
    Intervention,
    PatchIntervention,
    SteerIntervention,
    cast_like,
    replace_in_output,
)

# ---------------------------------------------------------------------------
# replace_in_output / cast_like
# ---------------------------------------------------------------------------


def test_replace_in_output_tensor():
    new = torch.ones(2)
    assert replace_in_output(torch.zeros(2), new) is new


def test_replace_in_output_tuple_preserves_tail():
    t = torch.zeros(2)
    new = torch.ones(2)
    out = replace_in_output((t, "kv_a", "kv_b"), new)
    assert out == (new, "kv_a", "kv_b")


def test_replace_in_output_list():
    t = torch.zeros(2)
    new = torch.ones(2)
    out = replace_in_output([t, 3], new)
    assert isinstance(out, tuple)
    assert out[0] is new and out[1] == 3


def test_replace_in_output_non_tensor_passthrough():
    new = torch.ones(2)
    assert replace_in_output((None, "x"), new) == (None, "x")
    assert replace_in_output("not a tensor", new) == "not a tensor"
    assert replace_in_output((), new) == ()


def test_cast_like_dtype_and_device():
    src = torch.ones(4, dtype=torch.float32)
    ref = torch.zeros(4, dtype=torch.float16)
    out = cast_like(src, ref)
    assert out.dtype == torch.float16
    assert out.device == ref.device


# ---------------------------------------------------------------------------
# GenerationContext
# ---------------------------------------------------------------------------


def test_generation_context_prefill_then_decode():
    ctx = GenerationContext(prompt_len=5)
    ctx.advance(5)  # prefill
    assert ctx.offset == 0 and ctx.step == -1
    ctx.advance(1)  # decode step 0
    assert ctx.offset == 5 and ctx.step == 0
    ctx.advance(1)  # decode step 1
    assert ctx.offset == 6 and ctx.step == 1


def test_generation_context_monotonicity_guard():
    ctx = GenerationContext()
    ctx.advance(4)
    ctx.advance(1)  # offset now 4
    ctx._total = 0  # simulate re-fed tokens (beam search)
    with pytest.raises(RuntimeError, match="num_beams=1"):
        ctx.advance(4)


# ---------------------------------------------------------------------------
# Hook behaviour on toy modules
# ---------------------------------------------------------------------------


class TupleOut(nn.Module):
    """Returns (tensor, 'kv') like an HF block with present_key_value."""

    def forward(self, x):
        return x, "kv"


def _run_with_hook(module: nn.Module, iv: Intervention, x: torch.Tensor, ctx=None):
    handle = module.register_forward_hook(iv.build_hook(ctx))
    try:
        return module(x)
    finally:
        handle.remove()


def test_steer_hook_adds_scaled_vector():
    mod = nn.Identity()
    vec = torch.ones(8)
    x = torch.zeros(1, 4, 8)
    out = _run_with_hook(mod, SteerIntervention("m", vector=vec, scale=3.0), x)
    assert torch.allclose(out, torch.full((1, 4, 8), 3.0))


def test_steer_hook_dim_mismatch_message():
    mod = nn.Identity()
    iv = SteerIntervention("blk", vector=torch.ones(16))
    with pytest.raises(ValueError, match=r"Steering vector dimension \(16\) does not match"):
        _run_with_hook(mod, iv, torch.zeros(1, 4, 8))


def test_steer_hook_positions_only():
    mod = nn.Identity()
    iv = SteerIntervention("m", vector=torch.ones(8), scale=2.0, positions=(1,))
    out = _run_with_hook(mod, iv, torch.zeros(1, 3, 8))
    assert torch.allclose(out[:, 1, :], torch.full((1, 8), 2.0))
    assert torch.allclose(out[:, 0, :], torch.zeros(1, 8))
    assert torch.allclose(out[:, 2, :], torch.zeros(1, 8))


def test_steer_hook_tuple_output_preserved():
    mod = TupleOut()
    out = _run_with_hook(mod, SteerIntervention("m", vector=torch.ones(8)), torch.zeros(2, 8))
    assert isinstance(out, tuple) and out[1] == "kv"
    assert torch.allclose(out[0], torch.full((2, 8), 2.0))  # default scale 2.0


def test_ablate_zero_and_mean():
    mod = nn.Identity()
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    out = _run_with_hook(mod, AblateIntervention("m", method="zero"), x)
    assert torch.allclose(out, torch.zeros_like(x))
    out = _run_with_hook(mod, AblateIntervention("m", method="mean"), x)
    expected = x.mean(dim=-2, keepdim=True).expand_as(x)
    assert torch.allclose(out, expected)


def test_ablate_resample_and_fallback():
    mod = nn.Identity()
    x = torch.zeros(1, 2, 4)
    repl = torch.ones(1, 2, 4)
    out = _run_with_hook(mod, AblateIntervention("m", method="resample", replacement=repl), x)
    assert torch.allclose(out, repl)
    # No replacement → zeros (legacy find_circuit fallback)
    out = _run_with_hook(mod, AblateIntervention("m", method="resample"), torch.ones(1, 2, 4))
    assert torch.allclose(out, torch.zeros(1, 2, 4))


def test_ablate_invalid_method_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        AblateIntervention("m", method="nuke")


def test_patch_full_replace_casts_dtype():
    mod = nn.Identity()
    src = torch.ones(1, 2, 4, dtype=torch.float32)
    x = torch.zeros(1, 2, 4, dtype=torch.float16)
    out = _run_with_hook(mod, PatchIntervention("m", source=src), x)
    assert out.dtype == torch.float16
    assert torch.allclose(out.float(), src)


def test_patch_positions_bounds_checked():
    mod = nn.Identity()
    src = torch.ones(1, 3, 4)
    x = torch.zeros(1, 3, 4)
    # position 99 silently skipped (legacy patch.py semantics), 1 applied
    out = _run_with_hook(mod, PatchIntervention("m", source=src, positions=(1, 99)), x)
    assert torch.allclose(out[:, 1, :], torch.ones(1, 4))
    assert torch.allclose(out[:, 0, :], torch.zeros(1, 4))


def test_positions_translated_by_generation_context():
    mod = nn.Identity()
    ctx = GenerationContext(prompt_len=5)
    ctx.advance(5)  # prefill: window [0, 5)
    ctx.advance(1)  # decode step 0: window [5, 6)
    iv = SteerIntervention("m", vector=torch.ones(4), scale=1.0, positions=(5,))
    out = _run_with_hook(mod, iv, torch.zeros(1, 1, 4), ctx)
    assert torch.allclose(out, torch.ones(1, 1, 4))
    # Out-of-window absolute position → untouched
    iv2 = SteerIntervention("m", vector=torch.ones(4), scale=1.0, positions=(3,))
    out2 = _run_with_hook(mod, iv2, torch.zeros(1, 1, 4), ctx)
    assert torch.allclose(out2, torch.zeros(1, 1, 4))


def test_fn_intervention_applies_and_casts():
    mod = nn.Identity()
    iv = FnIntervention("m", fn=lambda t, _ctx: t + 7.0)
    out = _run_with_hook(mod, iv, torch.zeros(2, 4))
    assert torch.allclose(out, torch.full((2, 4), 7.0))


def test_fn_intervention_rejects_positions():
    with pytest.raises(ValueError, match="does not take `positions`"):
        FnIntervention("m", fn=lambda t, _ctx: t, positions=(0,))


def test_capture_probe_stores_detached_copy():
    mod = nn.Linear(4, 4)
    store: dict[str, torch.Tensor] = {}
    iv = CaptureProbe("m", store=store, key="x")
    x = torch.randn(2, 4, requires_grad=True)
    out = _run_with_hook(mod, iv, x)
    assert "x" in store
    assert not store["x"].requires_grad
    assert torch.allclose(store["x"], out)


def test_capture_probe_rejects_positions():
    with pytest.raises(ValueError, match="positions"):
        CaptureProbe("m", store={}, key="x", positions=(0,))


def test_required_field_validation():
    with pytest.raises(ValueError, match="vector"):
        SteerIntervention("m")
    with pytest.raises(ValueError, match="source"):
        PatchIntervention("m")
    with pytest.raises(ValueError, match="fn"):
        FnIntervention("m")
    with pytest.raises(ValueError, match="store"):
        CaptureProbe("m")


def test_describe_elides_tensors():
    d = SteerIntervention("blk.6", vector=torch.ones(8), scale=4.0).describe()
    assert d["type"] == "steer"
    assert d["at"] == "blk.6"
    assert d["scale"] == 4.0
    assert d["vector"].startswith("<tensor")
    d = AblateIntervention("blk.2", method="mean", positions=(1, 2)).describe()
    assert d["type"] == "ablate" and d["method"] == "mean" and d["positions"] == (1, 2)
