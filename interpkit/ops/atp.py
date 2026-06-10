"""atp — Attribution Patching: first-order approximation of patch effects.

Public home of the AtP machinery that shipped in 1.0 as the private
shortlister for :mod:`interpkit.ops.trace`'s approximate mode
(``ops/_atp.py``, now a re-export shim). Reference:

- Nanda's mechanistic-interpretability notes
- Syed et al. 2023, "Attribution Patching Outperforms Automated Circuit
  Discovery"

Core formula
------------
For each module ``m`` whose output we'd otherwise patch:

    effect_approx[m] = grad(metric, m_output_corrupted) · (m_output_clean - m_output_corrupted)

This is the first-order Taylor expansion of the true patch effect around
the corrupted activation. Computing it requires one *clean* forward pass
(record each module's output), one *corrupted* forward pass that retains
the graph, and one backward pass on the corrupted metric. Total: ~3×
forward-pass cost, gives effect estimates for **every** module
simultaneously — versus one full forward per module for exhaustive
patching. Correlation with true patching effect is typically 0.85–0.95.

AtP is a *first-order* approximation: modules whose effect is locally
non-linear (sign cancellation through non-monotone paths) may rank-order
incorrectly. ``trace(method="approximate")`` therefore re-confirms the
top AtP candidates with full patching; ``run_atp`` reports raw scores
with that caveat in ``meta``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from interpkit.core.enums import VALID_METRICS, _validate_enum
from interpkit.ops._hooks import register_capture_hook, register_grad_capture_hook
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

__all__ = ["compute_atp_scores", "run_atp"]


def compute_atp_scores(
    model: Model,
    clean_input: Any,
    corrupted_input: Any,
    module_paths: list[str],
    *,
    metric: str = "logit_diff",
) -> dict[str, float]:
    """Compute Attribution Patching scores for every module in *module_paths*.

    Returns ``{module_path: float}`` mapping each module to its first-order
    approximation of the patch effect ``patch(corrupted, clean@m) → metric``.

    Parameters
    ----------
    metric:
        Currently only ``"logit_diff"`` is implemented in AtP form.
        Other metrics fall through with NaN scores (caller should detect
        and run full patching for those modules).

    Returns
    -------
    dict[str, float]
        Module path → AtP score. Larger absolute values indicate larger
        predicted effect under full patching.
    """
    if metric != "logit_diff":
        # Other metrics need a different formulation; return NaN to signal
        # the caller should fall back to full patching.
        return {p: float("nan") for p in module_paths}

    # Pass 1: clean forward — capture each module's output.
    clean_acts: dict[str, torch.Tensor] = {}

    handles = []
    for path in module_paths:
        try:
            mod = _get_module(model._model, path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        handles.append(register_capture_hook(mod, clean_acts, path))
    try:
        with torch.no_grad():
            clean_logits = model._forward(clean_input)
    finally:
        for h in handles:
            h.remove()

    # Pass 2: corrupted forward + backward — capture corrupted activations
    # WITH the graph attached (retain_grad) so backward fills .grad.
    captured_corrupted: dict[str, torch.Tensor] = {}
    grad_handles: list[torch.utils.hooks.RemovableHandle] = []

    for path in module_paths:
        try:
            mod = _get_module(model._model, path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        grad_handles.append(
            register_grad_capture_hook(mod, captured_corrupted, path)
        )

    try:
        corrupted_logits = model._forward_with_grad(corrupted_input).float()
    finally:
        for h in grad_handles:
            h.remove()

    # Pick the same scalar metric as patch.py's logit_diff for consistency
    # with the full-patch path. Use the top clean token at the last position.
    clean_flat = clean_logits.view(-1, clean_logits.shape[-1]).float()
    if clean_flat.shape[0] > 1:
        clean_flat = clean_flat[-1:]
    target_idx = int(clean_flat[0].argmax().item())

    if corrupted_logits.dim() == 3:
        score = corrupted_logits[0, -1, target_idx]
    elif corrupted_logits.dim() == 2:
        score = corrupted_logits[0, target_idx]
    else:
        score = corrupted_logits[target_idx]

    score.backward(retain_graph=False)

    # Compute AtP score per module: grad · (clean - corrupted)
    scores: dict[str, float] = {}
    for path in module_paths:
        if path not in captured_corrupted or path not in clean_acts:
            scores[path] = float("nan")
            continue
        corrupted_t = captured_corrupted[path]
        if corrupted_t.grad is None:
            scores[path] = float("nan")
            continue
        delta = (clean_acts[path] - corrupted_t.detach()).float()
        grad = corrupted_t.grad.float()
        if grad.shape != delta.shape:
            scores[path] = float("nan")
            continue
        # Per-module effect = (grad * (clean - corrupted)).sum()
        effect_approx = (grad * delta).sum().item()
        scores[path] = float(effect_approx)

    return scores


def run_atp(
    model: Model,
    clean: Any,
    corrupted: Any,
    *,
    top_k: int | None = 20,
    metric: str = "logit_diff",
) -> dict[str, Any]:
    """Rank every module by Attribution Patching score for a clean/corrupted pair.

    Three model passes total (clean forward, corrupted forward, one
    backward) score *all* modules at once — the fast first look before
    committing to :meth:`Model.trace`'s per-module full patching.

    Parameters
    ----------
    top_k:
        Number of top modules (by absolute score) to return. ``None`` or
        0 returns all.
    metric:
        Only ``"logit_diff"`` has an AtP formulation; other metrics
        return NaN scores (use ``trace`` for those).

    Returns
    -------
    dict with ``results`` (``{"module", "role", "score", "rank"}`` sorted
    by absolute score) and ``meta`` (method provenance + the first-order
    caveat).
    """
    from interpkit.core.render import render_atp
    from interpkit.core.support_matrix import check_op_supported

    _validate_enum(metric, VALID_METRICS, "metric")
    check_op_supported("atp", model.arch_info)

    arch = model.arch_info
    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    candidates = [m for m in arch.modules if m.param_count > 0]
    names = [m.name for m in candidates]
    role_map = {m.name: m.role for m in candidates}

    scores = compute_atp_scores(
        model, clean_input, corrupted_input, names, metric=metric,
    )
    ranked = sorted(
        scores.items(),
        key=lambda kv: abs(kv[1]) if kv[1] == kv[1] else -1.0,  # NaN sorts last
        reverse=True,
    )
    results = [
        {"module": name, "role": role_map.get(name), "score": s, "rank": i}
        for i, (name, s) in enumerate(ranked)
    ]
    if top_k:
        results = results[:top_k]

    meta: dict[str, Any] = {
        "method": "atp",
        "metric": metric,
        "n_modules": len(names),
        "n_forward_passes": 2,
        "n_backward_passes": 1,
        "caveat": (
            "first-order approximation — ranks can be wrong for locally "
            "non-linear modules; confirm top candidates with trace() or patch()"
        ),
    }
    if metric != "logit_diff":
        meta["warning"] = (
            f"metric {metric!r} has no AtP formulation; all scores are NaN. "
            f"Use trace() for this metric."
        )

    output = {"results": results, "meta": meta}
    render_atp(results, meta)
    return output
