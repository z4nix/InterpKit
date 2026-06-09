"""Attribution Patching (AtP) — first-order Taylor approximation of patch effects.

Pre-1.0 ``trace`` used a cheap proxy (activation-norm delta) to shortlist
modules before running the expensive full-patch on the top-K. The audit
showed this proxy missed important modules (e.g. ``transformer.wte`` ranks
low by activation-norm delta but ties for top-1 by true causal effect),
producing silently incorrect "top-K" results (F-015).

The 1.0 fix replaces the proxy with **Attribution Patching** — a much
stronger first-order approximation. Reference:
- Nanda's mechanistic-interpretability notes
- Syed et al. 2023, "Attribution Patching Outperforms Automated Circuit
  Discovery"

Core formula
------------
For each module ``m`` whose output we'd otherwise patch:

    effect_approx[m] = grad(metric, m_output_corrupted) · (m_output_clean - m_output_corrupted)

This is the first-order Taylor expansion of the true patch effect around
the corrupted activation. Computing it requires:

- One *clean* forward pass that records each module's output.
- One *corrupted* forward pass that records each module's output AND
  retains the computation graph for backprop.
- One backward pass on the corrupted metric.

Total: ~3× forward-pass cost, gives effect estimates for **every** module
simultaneously. Correlation with true patching effect is typically
0.85–0.95 — vastly better than the activation-norm proxy's 0.3–0.6.

Used by :mod:`interpkit.ops.trace` as the fast shortlist / approximate
mode in the three-tier dispatcher (auto / exhaustive / approximate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from interpkit.ops._hooks import first_tensor
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model


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

    Notes
    -----
    AtP is a *first-order* approximation. For modules whose effect is
    locally non-linear (e.g. through a non-monotone path with sign
    cancellation), the AtP score may rank-order incorrectly. The trace
    dispatcher always *re-confirms* the top AtP-ranked candidates with
    full patching before reporting; AtP is only used to shortlist.
    """
    if metric != "logit_diff":
        # Other metrics need a different formulation; return NaN to signal
        # the caller should fall back to full patching.
        return {p: float("nan") for p in module_paths}

    # Pass 1: clean forward — capture each module's output.
    clean_acts: dict[str, torch.Tensor] = {}

    def make_clean_hook(name: str):
        def fn(_m: nn.Module, _inp: Any, out: Any) -> None:
            t = first_tensor(out)
            if t is not None:
                clean_acts[name] = t.detach().clone()

        return fn

    handles = []
    for path in module_paths:
        try:
            mod = _get_module(model._model, path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        handles.append(mod.register_forward_hook(make_clean_hook(path)))
    with torch.no_grad():
        clean_logits = model._forward(clean_input)
    for h in handles:
        h.remove()

    # Pass 2: corrupted forward + backward — capture corrupted activations
    # AND retain the graph so we can compute gradient(metric, m_output).
    corrupted_acts: dict[str, torch.Tensor] = {}
    grad_handles: list[torch.utils.hooks.RemovableHandle] = []

    # Use forward hooks to capture corrupted module outputs WITH the graph
    # attached (no .detach()), then attach a tensor.retain_grad() so the
    # backward fills the .grad attribute.
    captured_corrupted: dict[str, torch.Tensor] = {}

    def make_capture_hook(name: str):
        def fn(_m: nn.Module, _inp: Any, out: Any) -> None:
            t = first_tensor(out)
            if t is not None:
                t.retain_grad()
                captured_corrupted[name] = t
                corrupted_acts[name] = t.detach().clone()

        return fn

    for path in module_paths:
        try:
            mod = _get_module(model._model, path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        grad_handles.append(mod.register_forward_hook(make_capture_hook(path)))

    corrupted_logits = model._forward_with_grad(corrupted_input).float()
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
        delta = (clean_acts[path] - corrupted_acts[path]).float()
        grad = corrupted_t.grad.float()
        if grad.shape != delta.shape:
            scores[path] = float("nan")
            continue
        # Per-module effect = (grad * (clean - corrupted)).sum()
        effect_approx = (grad * delta).sum().item()
        scores[path] = float(effect_approx)

    return scores


__all__ = ["compute_atp_scores"]
