"""steer — extract and apply steering vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.core.inputs import (
    MAX_LEADING_SPACE_WARNINGS as _MAX_TOKEN_WARNINGS,
)
from interpkit.core.inputs import (
    warn_if_leading_space_better,
)
from interpkit.core.paths import validate_module_path

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


__all__ = [
    "_MAX_TOKEN_WARNINGS",
    "_warn_if_token_mismatch",
    "run_steer",
    "run_steer_vector",
]


def _warn_if_token_mismatch(
    model: Model,
    text: Any,
    *,
    role: str,
    warned_count: list[int],
) -> None:
    """Backwards-compatible wrapper around :func:`warn_if_leading_space_better`.

    Kept so existing tests and external callers that imported the
    private helper from :mod:`interpkit.ops.steer` continue to work.
    Routes through this module's :data:`console` so test fixtures that
    monkeypatch ``steer.console.print`` keep observing the warning.
    """
    warn_if_leading_space_better(
        getattr(model, "_tokenizer", None),
        text,
        op_label="steer",
        role=role,
        warned_count=warned_count,
        console=console,
    )


def _activation_mean(model: Model, text: Any, *, at: str) -> torch.Tensor:
    """Return the mean activation vector for a single input at *at*."""
    from interpkit.ops.activations import run_activations

    act_result = run_activations(model, text, at=at, print_stats=False)
    assert isinstance(act_result, torch.Tensor)
    act = act_result
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
    from interpkit.core.inputs import normalize_input_group

    # F-022: reject typo'd module paths up-front with a friendly KeyError.
    validate_module_path(at, model.arch_info)

    positives = normalize_input_group(positive)
    negatives = normalize_input_group(negative)

    if not positives:
        raise ValueError("At least one positive example is required.")
    if not negatives:
        raise ValueError("At least one negative example is required.")

    warned: list[int] = [0]
    for p in positives:
        _warn_if_token_mismatch(model, p, role="positive", warned_count=warned)
    for n in negatives:
        _warn_if_token_mismatch(model, n, role="negative", warned_count=warned)

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

    assert pos_sum is not None, "No positive examples processed"
    assert neg_sum is not None, "No negative examples processed"
    pos_mean = pos_sum / len(positives)
    neg_mean = neg_sum / len(negatives)

    return pos_mean - neg_mean


def run_steer(
    model: Model,
    input_data: Any,
    *,
    vector: torch.Tensor | None = None,
    at: str,
    scale: float = 2.0,
    sae: Any = None,
    feature: int | None = None,
    mode: str = "clamp",
    strength: float = 10.0,
    save: str | None = None,
) -> dict[str, Any]:
    """Run inference with and without steering, compare top predictions.

    The steering unit is either a raw *vector* (contrastive activation
    steering; scaled by *scale*) or an SAE *feature* (requires *sae*):
    the feature's decoder direction is added (``mode="add"``) or its
    activation clamped to *strength* (``mode="clamp"``, Golden Gate
    style). Pass exactly one of ``vector=`` / ``feature=``.
    """
    from interpkit.core.render import render_steer
    from interpkit.core.support_matrix import check_op_supported

    # N-004: gate DeBERTa-v3 — steering hooks fire the broken
    # relative-position-bias broadcast path.
    check_op_supported("steer", model.arch_info)
    # F-022: reject typo'd module paths up-front with a friendly KeyError.
    validate_module_path(at, model.arch_info)

    if (vector is None) == (feature is None):
        raise ValueError(
            "Pass exactly one of vector= (contrastive steering) or "
            "feature= (SAE feature steering)."
        )
    if feature is not None and sae is None:
        raise ValueError(
            "feature= requires sae= (an SAE object, HF repo ID, or local path)."
        )
    if feature is None and sae is not None:
        raise ValueError("sae= only applies with feature= (SAE feature steering).")

    from interpkit.core.interventions import (
        SAEFeatureIntervention,
        SteerIntervention,
        apply_interventions,
    )

    steer_iv: SteerIntervention | SAEFeatureIntervention
    if feature is not None:
        from interpkit.ops.sae import _ensure_sae_on_device

        sae = _ensure_sae_on_device(sae, model._device)
        steer_iv = SAEFeatureIntervention(
            at, sae=sae, feature=feature, strength=strength, mode=mode,
        )
        label = f"feature {feature} {mode}@{strength:g}"
        plot_scale = strength
    else:
        steer_iv = SteerIntervention(at, vector=vector, scale=scale)
        label = None
        plot_scale = scale

    model_input = model._prepare(input_data)

    # 1. Original forward
    original_logits = model._forward(model_input)

    # 2. Steered forward — hook plumbing lives in core.interventions.
    with apply_interventions(model, [steer_iv]):
        steered_logits = model._forward(model_input)

    # Extract top tokens
    original_tokens = _top_tokens(model, original_logits)
    steered_tokens = _top_tokens(model, steered_logits)

    render_steer(original_tokens, steered_tokens, at, scale, label=label)

    if save is not None:
        from interpkit.core.plot import plot_steer

        plot_steer(
            original_tokens, steered_tokens,
            module_name=at, scale=plot_scale, save_path=save,
        )

    result: dict[str, Any] = {
        "original_logits": original_logits,
        "steered_logits": steered_logits,
        "original_top": original_tokens,
        "steered_top": steered_tokens,
    }
    if feature is not None:
        result["feature"] = feature
        result["mode"] = mode
        result["strength"] = strength
    return result


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
