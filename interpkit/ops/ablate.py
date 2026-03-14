"""ablate — zero or mean ablate a module and measure the effect on output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_ablate(
    model: "Model",
    input_data: Any,
    *,
    at: str,
    method: str = "zero",
) -> dict[str, Any]:
    """Ablate module *at* and measure the effect on output logits.

    Parameters
    ----------
    method:
        ``"zero"`` replaces the module output with zeros.
        ``"mean"`` replaces it with the mean activation across the sequence dimension.
    """
    from interpkit.core.render import render_ablate

    model_input = model._prepare(input_data)
    target_mod = _get_module(model._model, at)

    # 1. Clean forward — get baseline logits
    clean_logits = model._forward(model_input)

    # 2. Ablated forward
    def _ablate_hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> Any:
        t = output if isinstance(output, torch.Tensor) else (
            output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
        )
        if t is None:
            return output

        if method == "zero":
            replacement = torch.zeros_like(t)
        elif method == "mean":
            if t.dim() >= 3:
                replacement = t.mean(dim=-2, keepdim=True).expand_as(t)
            else:
                replacement = t.mean(dim=-1, keepdim=True).expand_as(t)
        else:
            raise ValueError(f"Unknown ablation method: {method!r}. Use 'zero' or 'mean'.")

        if isinstance(output, torch.Tensor):
            return replacement
        return (replacement,) + tuple(output[1:])

    handle = target_mod.register_forward_hook(_ablate_hook)
    ablated_logits = model._forward(model_input)
    handle.remove()

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
