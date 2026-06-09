"""External-validity / causal-faithfulness validation tests.

The rest of the suite is *verification* — invariants (decompose Σ, lens
top-1 == model top-1, head-activation sums), fail-loud contracts, golden
resolution snapshots, and plumbing on tiny-random models. Those prove the
implementation is self-consistent and regression-safe, but a tool can be
perfectly self-consistent and still attribute to the wrong component.

This module adds *validation*: does interpkit recover structure that is
independently known to be true, on a real trained model?

1. ``test_ioi_name_movers_recovered`` — ground-truth recovery. On the IOI
   task (Wang et al. 2022, "Interpretability in the Wild"), the GPT-2-small
   "name mover" heads (9.9, 9.6, 10.0) carry the largest *direct* logit
   attribution to the indirect-object token. We assert interpkit's DLA
   surfaces exactly those heads at the top.

2. ``test_dla_last_layer_causal_faithfulness`` — causal faithfulness of DLA's
   *direct* effect. DLA measures the direct path to the logit (bypassing
   downstream layers), so it must be validated against the *direct* effect,
   NOT a full zero-ablation (which also captures indirect/mediated effects).
   At the final layer there are no downstream layers, so direct ≈ total: a
   component's signed DLA contribution must match the sign of its causal
   ablation effect, and the larger |direct| component must have the larger
   |causal effect|.

Why no zero-ablation *necessity* test for the IOI heads: zero-ablation is
off-distribution and conflates a head's direct write with its downstream and
suppression effects — empirically, zero-ablating the name-mover layer
*increases* the IO−S logit diff on GPT-2. Faithful necessity for IOI needs
path/mean patching, not zero-ablation; asserting otherwise would falsely
indict a correct tool. See the direct-vs-total distinction above.

These require a real trained model (semantics are meaningless on random
weights), so they are marked ``slow`` and skip if the model is unavailable.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import pytest
import torch

import interpkit


def _quiet(fn, *args, **kwargs):
    """Call an op, suppressing its rich console rendering (keeps -s clean)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def gpt2():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return interpkit.load("gpt2", device="cpu")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"gpt2 unavailable: {type(exc).__name__}")


@pytest.mark.slow
def test_ioi_name_movers_recovered(gpt2):
    """DLA recovers the documented GPT-2-small IOI name-mover heads.

    Ground truth (Wang et al. 2022): heads 9.9, 9.6, 10.0 have the largest
    direct logit attribution to the indirect-object name. We assert all
    three appear in the top-5 heads by interpkit's DLA toward the IO token.
    """
    prompt = "When John and Mary went to the store, John gave a drink to"
    d = _quiet(gpt2.dla, prompt, token=" Mary", top_k=200)
    assert d["target_token"] == " Mary"

    top5 = {(h["layer"], h["head"]) for h in d["head_contributions"][:5]}
    name_movers = {(9, 9), (9, 6), (10, 0)}
    missing = name_movers - top5
    assert not missing, (
        f"DLA failed to recover IOI name-mover heads {sorted(missing)} in the "
        f"top-5; got {sorted(top5)}"
    )


@pytest.mark.slow
def test_dla_last_layer_causal_faithfulness(gpt2):
    """At the final layer (direct ≈ total), DLA's direct attribution is
    causally faithful: signs match the ablation effect, and the larger
    |direct contribution| component has the larger |causal effect|."""
    arch = gpt2.arch_info
    last = arch.num_layers - 1
    prompt = "When John and Mary went to the store, John gave a drink to"

    with torch.no_grad():
        clean_logits = gpt2._forward(gpt2._prepare(prompt))
    target = int(clean_logits[0, -1].argmax())

    d = _quiet(gpt2.dla, prompt, token=target, top_k=1000)
    last_comps = [c for c in d["contributions"] if c["layer"] == last]
    assert len(last_comps) >= 2, "expected attn + mlp components at the last layer"

    def path_of(c):
        li = arch.layer_infos[c["layer"]]
        return li.attn_path if c["type"] == "attn" else li.mlp_path

    measured = []
    for c in last_comps:
        r = _quiet(gpt2.ablate, prompt, at=path_of(c), method="zero")
        drop = float(r["clean_logits"][0, -1, target] - r["ablated_logits"][0, -1, target])
        measured.append((c, c["logit_contribution"], drop))

    # Sign faithfulness: a component whose direct contribution is +ve, when
    # removed, lowers the target logit (drop > 0); a -ve one raises it.
    # Guard on a non-trivial |direct| to avoid noise near zero.
    for c, direct, drop in measured:
        if abs(direct) > 1.0:
            assert (direct > 0) == (drop > 0), (
                f"{c['component']}: direct={direct:+.3f} but ablation drop="
                f"{drop:+.3f} (sign mismatch — direct effect not faithful)"
            )

    # Magnitude faithfulness: the largest |direct| component is also the
    # largest |causal effect| component.
    dom_by_direct = max(measured, key=lambda t: abs(t[1]))[0]["component"]
    dom_by_effect = max(measured, key=lambda t: abs(t[2]))[0]["component"]
    assert dom_by_direct == dom_by_effect, (
        f"DLA's dominant last-layer component ({dom_by_direct}) is not the "
        f"strongest causal one ({dom_by_effect})"
    )
