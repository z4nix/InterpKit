"""ablate — zero or mean ablate a module and measure the effect on output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from interpkit.core.enums import VALID_ABLATE_METHODS, _validate_enum
from interpkit.core.interventions import AblateIntervention, apply_interventions
from interpkit.core.paths import validate_module_path
from interpkit.ops._hooks import register_capture_hook
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_ablate(
    model: Model,
    input_data: Any,
    *,
    at: str,
    method: str = "zero",
    reference: Any | None = None,
) -> dict[str, Any]:
    """Ablate module *at* and measure the effect on output logits.

    Parameters
    ----------
    method:
        ``"zero"`` replaces the module output with zeros.
        ``"mean"`` replaces it with the mean activation across the sequence dimension.
        ``"resample"`` replaces it with activations from a *reference* input.
    reference:
        A different input whose activations replace the target module's
        output.  Required when ``method="resample"``.

    Raises
    ------
    ValueError
        If *method* is not one of ``"zero"`` / ``"mean"`` / ``"resample"``.
        Pre-1.0 silently fell back to a default on typos (F-018 family).
    """
    # F-018: validate method at the entry. No silent fallback.
    _validate_enum(method, VALID_ABLATE_METHODS, "method")

    # F-022: reject typo'd module paths up-front with a friendly KeyError.
    validate_module_path(at, model.arch_info)

    from interpkit.core.render import render_ablate

    model_input = model._prepare(input_data)
    target_mod = _get_module(model._model, at)

    # 1. Clean forward — get baseline logits
    with torch.no_grad():
        clean_logits = model._forward(model_input)

    # 2. For resample, cache the reference activation
    resample_act: torch.Tensor | None = None
    if method == "resample":
        if reference is None:
            raise ValueError("method='resample' requires a 'reference' input.")
        ref_input = model._prepare(reference)

        ref_store: dict[str, torch.Tensor] = {}
        h = register_capture_hook(target_mod, ref_store, "ref")
        try:
            with torch.no_grad():
                model._forward(ref_input)
        finally:
            h.remove()
        resample_act = ref_store.get("ref")

    # 3. Ablated forward — replacement math lives in AblateIntervention.
    ablate_iv = AblateIntervention(at, method=method, replacement=resample_act)
    with apply_interventions(model, [ablate_iv]), torch.no_grad():
        ablated_logits = model._forward(model_input)

    effect = _compute_ablation_effect(clean_logits, ablated_logits)

    result = {
        "module": at,
        "method": method,
        "effect": effect,
        "clean_logits": clean_logits,
        "ablated_logits": ablated_logits,
    }
    render_ablate(result)
    return result


def _compute_ablation_effect(clean: torch.Tensor, ablated: torch.Tensor) -> float:
    """Measure how much ablation changed the output (0 = no change, 1 = max change)."""
    clean_flat = clean.view(-1, clean.shape[-1]).float()
    ablated_flat = ablated.view(-1, ablated.shape[-1]).float()

    if clean_flat.shape[0] > 1:
        clean_flat = clean_flat[-1:]
        ablated_flat = ablated_flat[-1:]

    clean_probs = torch.softmax(clean_flat, dim=-1)
    ablated_probs = torch.softmax(ablated_flat, dim=-1)

    cosine_sim = torch.nn.functional.cosine_similarity(clean_probs, ablated_probs, dim=-1)
    return (1.0 - cosine_sim.item())
