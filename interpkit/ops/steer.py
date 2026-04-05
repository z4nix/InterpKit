"""steer — extract and apply steering vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def _activation_mean(model: Model, text: Any, *, at: str) -> torch.Tensor:
    """Return the mean activation vector for a single input at *at*."""
    from interpkit.ops.activations import run_activations

    act = run_activations(model, text, at=at, print_stats=False)
    if act.dim() == 3:
        return act[0].mean(dim=0)
    elif act.dim() == 2:
        return act.mean(dim=0)
    elif act.dim() == 1:
        return act
    raise ValueError(
        f"Steering requires activations with 1–3 dimensions (got {act.dim()}D "
        f"with shape {tuple(act.shape)}). Use a module that outputs (batch, seq, hidden) "
        f"shaped activations."
    )


def run_steer_vector(
    model: Model,
    positive: Any | list[Any],
    negative: Any | list[Any],
    *,
    at: str,
) -> torch.Tensor:
    """Extract a steering vector: mean(act(positives)) - mean(act(negatives)) at module *at*.

    *positive* and *negative* may each be a single input or a list of
    inputs.  When lists are provided the activations are averaged across
    all examples before computing the difference, producing a more robust
    direction (Contrastive Activation Addition).
    """
    positives = positive if isinstance(positive, list) else [positive]
    negatives = negative if isinstance(negative, list) else [negative]

    if not positives:
        raise ValueError("At least one positive example is required.")
    if not negatives:
        raise ValueError("At least one negative example is required.")

    total = len(positives) + len(negatives)
    use_progress = total > 2

    pos_sum: torch.Tensor | None = None
    neg_sum: torch.Tensor | None = None

    if use_progress:
        with Progress(console=console, transient=True) as progress:
            task = progress.add_task("Computing steering vector", total=total)
            for p in positives:
                mv = _activation_mean(model, p, at=at)
                pos_sum = mv if pos_sum is None else pos_sum + mv
                progress.advance(task)
            for n in negatives:
                mv = _activation_mean(model, n, at=at)
                neg_sum = mv if neg_sum is None else neg_sum + mv
                progress.advance(task)
    else:
        for p in positives:
            mv = _activation_mean(model, p, at=at)
            pos_sum = mv if pos_sum is None else pos_sum + mv
        for n in negatives:
            mv = _activation_mean(model, n, at=at)
            neg_sum = mv if neg_sum is None else neg_sum + mv

    pos_mean = pos_sum / len(positives)  # type: ignore[operator]
    neg_mean = neg_sum / len(negatives)  # type: ignore[operator]

    return pos_mean - neg_mean


def run_steer(
    model: Model,
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
        t = output if isinstance(output, torch.Tensor) else (
            output[0] if isinstance(output, (tuple, list)) and len(output) > 0 else None
        )
        if t is not None and t.shape[-1] != vector.shape[-1]:
            raise ValueError(
                f"Steering vector dimension ({vector.shape[-1]}) does not match "
                f"module output dimension ({t.shape[-1]}) at '{at}'."
            )
        if isinstance(output, torch.Tensor):
            return output + scale * vector.to(output.device)
        elif isinstance(output, (tuple, list)):
            steered = output[0] + scale * vector.to(output[0].device)
            return (steered,) + tuple(output[1:])
        return output

    handle = target_mod.register_forward_hook(_steer_hook)
    try:
        steered_logits = model._forward(model_input)
    finally:
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
    model: Model,
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
