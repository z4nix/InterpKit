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
    SAEFeatureIntervention,
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


def _toy_sae(hidden: int = 16, n_features: int = 8):
    """Orthonormal-decoder SAE with W_enc = W_dec.T, so encode∘(x + c·W_dec[i])
    shifts feature i's pre-activation by exactly c. Rows are random (a uniform
    direction would be cancelled by LayerNorm in real-model tests)."""
    from interpkit.ops.sae import load_sae_from_tensors

    g = torch.Generator().manual_seed(0)
    rows = torch.linalg.qr(torch.randn(hidden, hidden, generator=g))[0][:n_features]
    return load_sae_from_tensors(
        W_enc=rows.T.clone(), W_dec=rows.clone(),
        b_enc=torch.zeros(n_features), b_dec=torch.zeros(hidden),
        metadata={"apply_b_dec_to_input": False},
    )


def test_sae_feature_add_matches_manual_delta():
    sae = _toy_sae()
    mod = nn.Identity()
    x = torch.randn(1, 5, 16, generator=torch.Generator().manual_seed(1))
    iv = SAEFeatureIntervention("m", sae=sae, feature=3, strength=4.0, mode="add")
    out = _run_with_hook(mod, iv, x)
    assert torch.allclose(out, x + 4.0 * sae.W_dec[3])


def test_sae_feature_clamp_pins_reencoded_activation():
    sae = _toy_sae()
    mod = nn.Identity()
    g = torch.Generator().manual_seed(1)
    # Force positive pre-activations so the post-ReLU clamp is exact.
    x = torch.randn(1, 5, 16, generator=g) + 5.0 * sae.W_dec[3]
    iv = SAEFeatureIntervention("m", sae=sae, feature=3, strength=4.0, mode="clamp")
    out = _run_with_hook(mod, iv, x)
    feats = sae.encode(out)[..., 3]
    assert torch.allclose(feats, torch.full_like(feats, 4.0), atol=1e-5)


def test_sae_feature_clamp_negative_preact_invariant():
    """Where the feature is under the ReLU threshold (pre-activation < 0) the
    residual-space clamp lands at relu(strength + pre_act) — the standard
    behaviour: the current activation reads 0, so strength·d is added on top
    of the (negative) existing projection."""
    sae = _toy_sae()
    mod = nn.Identity()
    x = torch.randn(1, 5, 16, generator=torch.Generator().manual_seed(1))
    iv = SAEFeatureIntervention("m", sae=sae, feature=3, strength=4.0, mode="clamp")
    out = _run_with_hook(mod, iv, x)
    pre = x @ sae.W_dec[3]
    expected = torch.relu(4.0 + torch.clamp(pre, max=0.0))
    assert torch.allclose(sae.encode(out)[..., 3], expected, atol=1e-5)


def test_sae_feature_positions_only():
    sae = _toy_sae()
    mod = nn.Identity()
    x = torch.randn(1, 5, 16, generator=torch.Generator().manual_seed(2))
    iv = SAEFeatureIntervention(
        "m", sae=sae, feature=3, strength=4.0, mode="clamp", positions=(2,),
    )
    out = _run_with_hook(mod, iv, x)
    untouched = [0, 1, 3, 4]
    assert torch.allclose(out[:, untouched], x[:, untouched])
    assert not torch.allclose(out[:, 2], x[:, 2])


def test_sae_feature_tuple_output_preserved():
    sae = _toy_sae()
    mod = TupleOut()
    iv = SAEFeatureIntervention("m", sae=sae, feature=0, strength=1.0, mode="add")
    out = _run_with_hook(mod, iv, torch.zeros(2, 16))
    assert isinstance(out, tuple) and out[1] == "kv"
    assert torch.allclose(out[0], sae.W_dec[0].expand(2, 16))


def test_sae_feature_validation():
    sae = _toy_sae()
    with pytest.raises(ValueError, match="requires an `sae`"):
        SAEFeatureIntervention("m", feature=1)
    with pytest.raises(ValueError, match="must be >= 0"):
        SAEFeatureIntervention("m", sae=sae, feature=-1)
    with pytest.raises(ValueError, match="out of range"):
        SAEFeatureIntervention("m", sae=sae, feature=99)
    with pytest.raises(ValueError, match="Unknown mode"):
        SAEFeatureIntervention("m", sae=sae, feature=1, mode="boost")
    iv = SAEFeatureIntervention("m", sae=sae, feature=1)
    with pytest.raises(ValueError, match=r"SAE decoder dimension \(16\) does not match"):
        _run_with_hook(nn.Identity(), iv, torch.zeros(1, 4, 8))


def test_sae_feature_describe_is_json_safe():
    import json

    sae = _toy_sae()
    iv = SAEFeatureIntervention("m", sae=sae, feature=3, strength=4.0)
    desc = iv.describe()
    json.dumps(desc)
    assert desc["type"] == "sae_feature"
    assert desc["sae"] == "<SAE d_in=16 d_sae=8>"
    assert desc["feature"] == 3 and desc["mode"] == "clamp"


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
