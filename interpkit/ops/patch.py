"""patch — activation patching at a named module between clean and corrupted inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from interpkit.core.model import Model


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
    head: int | None = None,
    positions: list[int] | None = None,
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

    Returns a dict with ``effect`` — a normalised scalar in [0, 1] measuring how
    much the patched corrupted run's output shifted toward the clean output.
    """
    from interpkit.core.render import render_patch

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

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

    corrupted_logits = model._forward(corrupted_input)

    # Build the patching hook based on head / positions
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
                    return (mixed.to(t.device),) + inp[1:] if isinstance(inp, tuple) and len(inp) > 1 else (mixed.to(t.device),)
                return inp

            handle = proj_mod.register_forward_pre_hook(_pre_hook)
            patched_logits = model._forward(corrupted_input)
            handle.remove()
        else:
            patched_logits = corrupted_logits

    elif positions is not None:
        clean_cached = cached_activation[0]

        def _pos_patch_hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> Any:
            t = output if isinstance(output, torch.Tensor) else (
                output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
            )
            if t is None:
                return output
            patched = t.clone()
            for p in positions:
                if patched.dim() == 3 and p < patched.shape[1]:
                    patched[:, p, :] = clean_cached[:, p, :]
                elif patched.dim() == 2 and p < patched.shape[0]:
                    patched[p, :] = clean_cached[p, :]
            if isinstance(output, torch.Tensor):
                return patched
            return (patched,) + tuple(output[1:])

        handle = target_mod.register_forward_hook(_pos_patch_hook)
        patched_logits = model._forward(corrupted_input)
        handle.remove()

    else:
        def _patch_hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> Any:
            if isinstance(output, torch.Tensor):
                return cached_activation[0]
            elif isinstance(output, (tuple, list)):
                return (cached_activation[0],) + tuple(output[1:])
            return output

        handle = target_mod.register_forward_hook(_patch_hook)
        patched_logits = model._forward(corrupted_input)
        handle.remove()

    effect = _compute_effect(clean_logits, corrupted_logits, patched_logits)

    result = {
        "module": at,
        "effect": effect,
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
) -> float:
    """Normalised patching effect: 0 = patched == corrupted, 1 = patched == clean.

    Uses L2 distance between probability distributions as the distance metric.
    """
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
