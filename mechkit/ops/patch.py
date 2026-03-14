"""patch — activation patching at a named module between clean and corrupted inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mechkit.core.model import Model


def _get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    parts = name.split(".")
    mod = model
    for part in parts:
        mod = getattr(mod, part)
    return mod


def run_patch(
    model: "Model",
    clean: Any,
    corrupted: Any,
    *,
    at: str,
) -> dict[str, Any]:
    """Patch the output of module *at* from the clean run into the corrupted run.

    Returns a dict with ``effect`` — a normalised scalar in [0, 1] measuring how
    much the patched corrupted run's output shifted toward the clean output.
    """
    from mechkit.core.render import render_patch

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    # 1. Clean forward — cache the target module's output
    cached_activation: list[torch.Tensor] = []

    target_mod = _get_module(model._model, at)

    def _cache_hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor):
            cached_activation.append(output.detach().clone())
        elif isinstance(output, (tuple, list)):
            cached_activation.append(output[0].detach().clone())

    handle = target_mod.register_forward_hook(_cache_hook)
    clean_logits = model._forward(clean_input)
    handle.remove()

    if not cached_activation:
        raise RuntimeError(f"Module '{at}' produced no tensor output during clean forward pass.")

    # 2. Corrupted forward (baseline)
    corrupted_logits = model._forward(corrupted_input)

    # 3. Patched forward — replace target module's output with cached clean activation
    def _patch_hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return cached_activation[0]
        elif isinstance(output, (tuple, list)):
            return (cached_activation[0],) + tuple(output[1:])
        return output

    handle = target_mod.register_forward_hook(_patch_hook)
    patched_logits = model._forward(corrupted_input)
    handle.remove()

    # 4. Compute normalised effect
    effect = _compute_effect(clean_logits, corrupted_logits, patched_logits)

    result = {
        "module": at,
        "effect": effect,
        "clean_logits": clean_logits,
        "corrupted_logits": corrupted_logits,
        "patched_logits": patched_logits,
    }
    render_patch(result)
    return result


def _compute_effect(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    patched: torch.Tensor,
) -> float:
    """Normalised patching effect: 0 = patched == corrupted, 1 = patched == clean."""
    # Use KL divergence on the last-token logits as the distance metric
    clean_flat = clean.view(-1, clean.shape[-1]).float()
    corrupted_flat = corrupted.view(-1, corrupted.shape[-1]).float()
    patched_flat = patched.view(-1, patched.shape[-1]).float()

    # Take last position for sequence models
    if clean_flat.shape[0] > 1:
        clean_flat = clean_flat[-1:]
        corrupted_flat = corrupted_flat[-1:]
        patched_flat = patched_flat[-1:]

    clean_probs = torch.softmax(clean_flat, dim=-1)
    corrupted_probs = torch.softmax(corrupted_flat, dim=-1)
    patched_probs = torch.softmax(patched_flat, dim=-1)

    dist_corrupted_clean = torch.norm(corrupted_probs - clean_probs).item()
    dist_patched_clean = torch.norm(patched_probs - clean_probs).item()

    if dist_corrupted_clean < 1e-8:
        return 0.0

    effect = 1.0 - (dist_patched_clean / dist_corrupted_clean)
    return max(0.0, min(1.0, effect))
