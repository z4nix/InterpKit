"""patch — activation patching at a named module between clean and corrupted inputs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from interpkit.core.enums import VALID_METRICS, _validate_enum
from interpkit.core.interventions import PatchIntervention, apply_interventions

# Backwards-compat shim: the canonical helper moved to core.paths.get_module.
# steer.py / ablate.py / trace.py / find_circuit.py / _atp.py import it from
# here (same pattern as steer._warn_if_token_mismatch).
from interpkit.core.paths import get_module as _get_module  # noqa: F401
from interpkit.core.paths import validate_module_path

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_patch(
    model: Model,
    clean: Any,
    corrupted: Any,
    *,
    at: str,
    head: int | None = None,
    positions: list[int] | None = None,
    metric: str = "logit_diff",
) -> dict[str, Any]:
    """Patch the output of module *at* from the clean run into the corrupted run.

    Parameters
    ----------
    head:
        If specified, patch only this attention head's contribution.
        Requires ``at`` to point to an attention module with a detectable
        output projection.
    positions:
        If specified, patch only these token positions.  Can be combined
        with *head* for fine-grained patching.
    metric:
        Effect metric: ``"logit_diff"`` (default), ``"kl_div"``,
        ``"target_prob"``, or ``"l2_prob"`` (legacy).

    Returns a dict with ``effect`` measuring how much the patched corrupted
    run's output shifted toward the clean output.

    The result also includes a ``warnings`` list. For ``logit_diff`` and
    ``target_prob_effect``, when the clean-vs-corrupted gap is below the
    numeric guard the metric is undefined; the effect returns ``NaN`` and
    ``warnings`` includes ``"degenerate_gap"`` (F-010). Pre-1.0 silently
    returned 0 for this case, which looked like "the patch did nothing"
    when in fact the metric was simply undefined.
    """
    from interpkit.core.render import render_patch

    # F-018: validate metric at the entry, not deep in dispatch. Pre-1.0
    # silently fell back to defaults on typos.
    _validate_enum(metric, VALID_METRICS, "metric")

    # F-022: reject typo'd module paths up-front with a friendly KeyError
    # rather than letting `_get_module` emit a raw HF `AttributeError`.
    validate_module_path(at, model.arch_info)

    from interpkit.ops._hooks import register_capture_hook

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    clean_store: dict[str, torch.Tensor] = {}
    target_mod = _get_module(model._model, at)

    handle = register_capture_hook(target_mod, clean_store, "clean")
    try:
        clean_logits = model._forward(clean_input)
    finally:
        handle.remove()

    if "clean" not in clean_store:
        raise RuntimeError(f"Module '{at}' produced no tensor output during clean forward pass.")
    clean_activation = clean_store["clean"]

    corrupted_logits = model._forward(corrupted_input)

    # Build the patching hook based on head / positions.
    # The head-level path below intentionally stays inline: it performs
    # *input* surgery via pre-hooks on the attention output projection,
    # a different contract from the output-replacement Interventions in
    # core.interventions (see that module's deferral ledger). Revisit
    # when per-head nodes land with EAP (roadmap phase 2).
    if head is not None:
        num_heads = model.arch_info.num_attention_heads
        if num_heads is None:
            raise ValueError("Head-level patching requires num_attention_heads in model config.")

        from interpkit.ops.heads import _find_output_proj

        _, _, proj_mod = _find_output_proj(model._model, at)
        if proj_mod is None or not hasattr(proj_mod, "weight"):
            raise RuntimeError(
                f"Head-level patching requires an output projection in '{at}'."
            )

        clean_pre: list[torch.Tensor] = []
        corrupted_pre: list[torch.Tensor] = []

        def _cap_pre_hook(store: list):
            def hook_fn(_m, inp, _out):
                t = inp[0] if isinstance(inp, tuple) else inp
                if isinstance(t, torch.Tensor):
                    store.append(t.detach().clone())
            return hook_fn

        h = proj_mod.register_forward_hook(_cap_pre_hook(clean_pre))
        model._forward(clean_input)
        h.remove()

        h = proj_mod.register_forward_hook(_cap_pre_hook(corrupted_pre))
        model._forward(corrupted_input)
        h.remove()

        if not clean_pre or not corrupted_pre:
            raise RuntimeError(
                f"Head-level patching failed: could not capture pre-projection "
                f"activations for module '{at}'. The output projection may not "
                f"match the expected structure."
            )
        if clean_pre and corrupted_pre:
            cp = clean_pre[0].float()
            crp = corrupted_pre[0].float()
            if cp.dim() == 2:
                cp = cp.unsqueeze(0)
                crp = crp.unsqueeze(0)
            head_dim = cp.shape[-1] // num_heads

            mixed = crp.clone()
            start = head * head_dim
            end = start + head_dim
            if positions is not None:
                for p in positions:
                    mixed[:, p, start:end] = cp[:, p, start:end]
            else:
                mixed[:, :, start:end] = cp[:, :, start:end]

            def _pre_hook(_mod, inp):
                t = inp[0] if isinstance(inp, tuple) else inp
                if isinstance(t, torch.Tensor):
                    # F-008: cast back to the module's input dtype/device,
                    # not just device. Surgery happened in fp32 (.float() above)
                    # so non-fp32 models would otherwise see Float vs Half
                    # mismatch in out_proj/o_proj.
                    cast = mixed.to(device=t.device, dtype=t.dtype)
                    if isinstance(inp, tuple) and len(inp) > 1:
                        return (cast,) + inp[1:]
                    return (cast,)
                return inp

            handle = proj_mod.register_forward_pre_hook(_pre_hook)
            patched_logits = model._forward(corrupted_input)
            handle.remove()

    else:
        # F-008 dtype/device casting and the position-bounds semantics live
        # in PatchIntervention (core.interventions) — the single canonical
        # implementation of activation writeback.
        patch_iv = PatchIntervention(
            at,
            source=clean_activation,
            positions=tuple(positions) if positions is not None else None,
        )
        with apply_interventions(model, [patch_iv]):
            patched_logits = model._forward(corrupted_input)

    effect, warnings = _compute_effect(
        clean_logits, corrupted_logits, patched_logits, metric=metric,
    )

    result = {
        "module": at,
        "effect": effect,
        "warnings": warnings,
        "metric": metric,
        "clean_logits": clean_logits,
        "corrupted_logits": corrupted_logits,
        "patched_logits": patched_logits,
    }
    if head is not None:
        result["head"] = head
    if positions is not None:
        result["positions"] = positions

    render_patch(result)
    return result


def _compute_effect(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    patched: torch.Tensor,
    *,
    metric: str = "logit_diff",
) -> tuple[float, list[str]]:
    """Normalised patching effect: 0 = patched == corrupted, 1 = patched == clean.

    Returns ``(effect_value, warnings_list)``. When the metric is undefined
    (degenerate gap below numeric guard for ratio-style metrics), returns
    ``(NaN, ["degenerate_gap"])`` rather than silently masking with 0.

    Parameters
    ----------
    metric:
        ``"logit_diff"`` — Logit difference of the top clean token,
            normalised by the clean-vs-corrupted gap. Standard in circuit
            analysis (Wang et al. 2022). Returns NaN when the gap is
            below 1e-8 (the metric is undefined; F-010).
        ``"kl_div"`` — KL(clean || patched) normalised by
            KL(clean || corrupted). Captures full distributional shift.
        ``"target_prob"`` — Raw probability of the top clean token in the
            patched run. NOT normalised — value range is [0, 1] and an
            identity patch returns ``p_corrupted``, not 0 (F-009).
        ``"target_prob_effect"`` — Normalised effect:
            ``p_patched - p_corrupted``. Returns 0 for an identity patch,
            consistent with the other ratio metrics (F-009, new in 1.0).
        ``"l2_prob"`` — Legacy metric: L2 distance between probability
            vectors, normalised.
    """
    warnings: list[str] = []

    clean_flat = clean.view(-1, clean.shape[-1]).float()
    corrupted_flat = corrupted.view(-1, corrupted.shape[-1]).float()
    patched_flat = patched.view(-1, patched.shape[-1]).float()

    if clean_flat.shape[0] > 1:
        clean_flat = clean_flat[-1:]
        corrupted_flat = corrupted_flat[-1:]
        patched_flat = patched_flat[-1:]

    if metric == "logit_diff":
        target_idx = int(clean_flat[0].argmax().item())
        clean_logit = float(clean_flat[0, target_idx].item())
        corrupted_logit = float(corrupted_flat[0, target_idx].item())
        patched_logit = float(patched_flat[0, target_idx].item())
        denom = clean_logit - corrupted_logit
        if abs(denom) < 1e-8:
            # F-010: degenerate gap. Pre-1.0 returned 0 silently, which
            # looked like "the patch did nothing" when actually the
            # metric is mathematically undefined. Return NaN + warning so
            # downstream visualisations make the issue visible.
            warnings.append("degenerate_gap")
            return float("nan"), warnings
        return (patched_logit - corrupted_logit) / denom, warnings

    elif metric == "kl_div":
        clean_lp = F.log_softmax(clean_flat, dim=-1)
        corrupted_lp = F.log_softmax(corrupted_flat, dim=-1)
        patched_lp = F.log_softmax(patched_flat, dim=-1)
        clean_probs = clean_lp.exp()
        kl_corrupted = float(F.kl_div(corrupted_lp, clean_probs, reduction="batchmean").item())
        kl_patched = float(F.kl_div(patched_lp, clean_probs, reduction="batchmean").item())
        if kl_corrupted < 1e-10:
            warnings.append("degenerate_gap")
            return float("nan"), warnings
        return 1.0 - (kl_patched / kl_corrupted), warnings

    elif metric == "target_prob":
        # F-009: target_prob is the raw probability, not a normalised effect.
        # Documented clearly so users don't expect 0 for identity-patch.
        target_idx = int(clean_flat[0].argmax().item())
        patched_probs = torch.softmax(patched_flat, dim=-1)
        return float(patched_probs[0, target_idx].item()), warnings

    elif metric == "target_prob_effect":
        # F-009: normalised effect — difference between patched and corrupted
        # probabilities. Identity patch returns 0; full reversion returns
        # ``p_clean - p_corrupted``. Symmetric with logit_diff conventions.
        target_idx = int(clean_flat[0].argmax().item())
        corrupted_probs = torch.softmax(corrupted_flat, dim=-1)
        patched_probs = torch.softmax(patched_flat, dim=-1)
        clean_probs = torch.softmax(clean_flat, dim=-1)
        p_clean = float(clean_probs[0, target_idx].item())
        p_corrupted = float(corrupted_probs[0, target_idx].item())
        p_patched = float(patched_probs[0, target_idx].item())
        denom = p_clean - p_corrupted
        if abs(denom) < 1e-8:
            warnings.append("degenerate_gap")
            return float("nan"), warnings
        return (p_patched - p_corrupted) / denom, warnings

    elif metric == "l2_prob":
        clean_probs = torch.softmax(clean_flat, dim=-1)
        corrupted_probs = torch.softmax(corrupted_flat, dim=-1)
        patched_probs = torch.softmax(patched_flat, dim=-1)
        dist_corrupted_clean = float(torch.norm(corrupted_probs - clean_probs).item())
        dist_patched_clean = float(torch.norm(patched_probs - clean_probs).item())
        if dist_corrupted_clean < 1e-8:
            warnings.append("degenerate_gap")
            return float("nan"), warnings
        return 1.0 - (dist_patched_clean / dist_corrupted_clean), warnings

    # Should be unreachable due to _validate_enum in run_patch.
    raise ValueError(f"Unknown metric {metric!r}.")


# Backwards-compat helper for callers (trace, etc.) that expect a scalar.
def _compute_effect_value(
    clean: torch.Tensor, corrupted: torch.Tensor, patched: torch.Tensor,
    *, metric: str = "logit_diff",
) -> float:
    """Like :func:`_compute_effect` but returns just the effect value.

    Used internally by ``trace`` and ``find_circuit`` which iterate over
    many components and don't currently surface per-call warnings. Future
    refactors can switch them to the tuple-returning form.
    """
    value, warnings = _compute_effect(clean, corrupted, patched, metric=metric)
    if warnings and not math.isnan(value):
        # If we got a value and warnings, the effect is well-defined.
        pass
    return value






