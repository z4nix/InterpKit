"""steer — extract and apply steering vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_steer_vector(
    model: "Model",
    positive: Any,
    negative: Any,
    *,
    at: str,
) -> torch.Tensor:
    """Extract a steering vector: activation(positive) - activation(negative) at module *at*.

    Both inputs are padded to the same length. The vector is the mean
    difference across the sequence dimension.
    """
    from interpkit.ops.activations import run_activations

    pos_act = run_activations(model, positive, at=at, print_stats=False)
    neg_act = run_activations(model, negative, at=at, print_stats=False)

    # Mean across sequence dim if present
    if pos_act.dim() >= 3:
        pos_mean = pos_act[0].mean(dim=0)  # (hidden,)
        neg_mean = neg_act[0].mean(dim=0)
    elif pos_act.dim() == 2:
        pos_mean = pos_act.mean(dim=0)
        neg_mean = neg_act.mean(dim=0)
    else:
        pos_mean = pos_act
        neg_mean = neg_act

    return pos_mean - neg_mean


def run_steer(
    model: "Model",
    input_data: Any,
    *,
    vector: torch.Tensor,
    at: str,
    scale: float = 2.0,
    save: str | None = None,
) -> dict[str, Any]:
    """Run inference with and without a steering vector, compare top predictions."""
    from interpkit.core.render import render_steer

    model_input = model._prepare(input_data)

    # 1. Original forward
    original_logits = model._forward(model_input)

    # 2. Steered forward
    target_mod = _get_module(model._model, at)

    def _steer_hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output + scale * vector.to(output.device)
        elif isinstance(output, (tuple, list)):
            steered = output[0] + scale * vector.to(output[0].device)
            return (steered,) + tuple(output[1:])
        return output

    handle = target_mod.register_forward_hook(_steer_hook)
    steered_logits = model._forward(model_input)
    handle.remove()

    # Extract top tokens
    original_tokens = _top_tokens(model, original_logits)
    steered_tokens = _top_tokens(model, steered_logits)

    render_steer(original_tokens, steered_tokens, at, scale)

    if save is not None:
        from interpkit.core.plot import plot_steer

        plot_steer(original_tokens, steered_tokens, module_name=at, scale=scale, save_path=save)

    return {
        "original_logits": original_logits,
        "steered_logits": steered_logits,
        "original_top": original_tokens,
        "steered_top": steered_tokens,
    }


def _top_tokens(
    model: "Model",
    logits: torch.Tensor,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Extract top-k predicted tokens from logits."""
    if logits.dim() == 3:
        last_logits = logits[0, -1, :]
    elif logits.dim() == 2:
        last_logits = logits[-1, :]
    else:
        last_logits = logits.view(-1)

    probs = torch.softmax(last_logits.float(), dim=-1)
    top_probs, top_ids = probs.topk(k)

    if model._tokenizer is not None:
        tokens = [model._tokenizer.decode([tid]) for tid in top_ids.tolist()]
    else:
        tokens = [str(tid) for tid in top_ids.tolist()]

    return list(zip(tokens, top_probs.tolist()))
