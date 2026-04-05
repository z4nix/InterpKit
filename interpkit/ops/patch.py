"""patch — activation patching at a named module between clean and corrupted inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from interpkit.core.model import Model


def _get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    parts = name.split(".")
    mod = model
    for part in parts:
        mod = getattr(mod, part)
    return mod


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
                    return (mixed.to(t.device),) + inp[1:] if isinstance(inp, tuple) and len(inp) > 1 else (mixed.to(t.device),)
                return inp

            handle = proj_mod.register_forward_pre_hook(_pre_hook)
            patched_logits = model._forward(corrupted_input)
            handle.remove()

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

    effect = _compute_effect(clean_logits, corrupted_logits, patched_logits, metric=metric)

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
    *,
    metric: str = "logit_diff",
) -> float:
    """Normalised patching effect: 0 = patched == corrupted, 1 = patched == clean.

    Parameters
    ----------
    metric:
        ``"logit_diff"`` — Logit difference of the top clean token,
            normalised by the clean-vs-corrupted gap.  Standard in
            circuit analysis (Wang et al. 2022).
        ``"kl_div"`` — KL(clean || patched) normalised by
            KL(clean || corrupted).  Captures full distributional shift.
        ``"target_prob"`` — Probability of the top clean token in the
            patched run (not normalised, raw probability).
        ``"l2_prob"`` — Legacy metric: L2 distance between probability
            vectors, normalised.
    """
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
            return 0.0
        return (patched_logit - corrupted_logit) / denom

    elif metric == "kl_div":
        clean_lp = F.log_softmax(clean_flat, dim=-1)
        corrupted_lp = F.log_softmax(corrupted_flat, dim=-1)
        patched_lp = F.log_softmax(patched_flat, dim=-1)
        clean_probs = clean_lp.exp()
        kl_corrupted = float(F.kl_div(corrupted_lp, clean_probs, reduction="batchmean").item())
        kl_patched = float(F.kl_div(patched_lp, clean_probs, reduction="batchmean").item())
        if kl_corrupted < 1e-10:
            return 0.0
        return 1.0 - (kl_patched / kl_corrupted)

    elif metric == "target_prob":
        target_idx = int(clean_flat[0].argmax().item())
        patched_probs = torch.softmax(patched_flat, dim=-1)
        return float(patched_probs[0, target_idx].item())

    elif metric == "l2_prob":
        clean_probs = torch.softmax(clean_flat, dim=-1)
        corrupted_probs = torch.softmax(corrupted_flat, dim=-1)
        patched_probs = torch.softmax(patched_flat, dim=-1)
        dist_corrupted_clean = float(torch.norm(corrupted_probs - clean_probs).item())
        dist_patched_clean = float(torch.norm(patched_probs - clean_probs).item())
        if dist_corrupted_clean < 1e-8:
            return 0.0
        return 1.0 - (dist_patched_clean / dist_corrupted_clean)

    else:
        raise ValueError(
            f"Unknown metric {metric!r}. "
            f"Use 'logit_diff', 'kl_div', 'target_prob', or 'l2_prob'."
        )
