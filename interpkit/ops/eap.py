"""eap — Edge Attribution Patching: gradient-based edge scores for circuit discovery.

Where :mod:`interpkit.ops.atp` scores *nodes* (modules), EAP scores
*edges* between an upstream component's output delta and the residual
stream it feeds. References:

- Syed et al. 2023, "Attribution Patching Outperforms Automated Circuit
  Discovery"
- Hanna et al. 2024, "Have Faith in Faithfulness: Going Beyond Circuit
  Overlap When Finding Model Mechanisms" (EAP-IG)

v1 edge semantics (deliberately LN-free)
----------------------------------------
Upstream nodes ``u`` are the embedding and each layer's attention / MLP
output; downstream nodes are the residual-stream outputs ``resid_l`` of
each block. For ``l >= layer(u)``:

    score(u → resid_l) = Σ ∂metric/∂resid_l · (u_clean − u_corrupted)

computed from **one clean forward + one corrupted forward + one
backward** (all edges at once). Because a pre-LN residual stream is
additive, ``Δu`` is exactly the perturbation ``u`` would inject at every
block at or after its own layer; the gradient at ``resid_l`` measures
that perturbation's first-order effect through everything downstream of
block ``l``. The edge at ``l = layer(u)`` therefore equals ``u``'s total
node effect (≈ its AtP score), and deeper edges show how much of the
effect is mediated below layer ``l``. Module-level downstream nodes
(through-LN pullbacks) and per-head granularity are documented deferrals.

EAP-IG (``ig_steps > 0``) replaces the single corrupted backward with
``ig_steps`` backwards at embeddings interpolated from corrupted toward
clean (midpoint rule), averaging the gradients — more faithful scores
when the corrupted point sits in a saturated region.

Scope: causal LMs with an additive residual stream. Encoder-decoder and
shared-weight (ALBERT-style) models raise
``OperationNotSupportedForArchitecture`` (documented deferrals, same
ledger pattern as ``core/interventions.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from interpkit.core.enums import VALID_EAP_METRICS, _validate_enum
from interpkit.core.exceptions import OperationNotSupportedForArchitecture
from interpkit.core.interventions import FnIntervention
from interpkit.ops._hooks import register_capture_hook, register_grad_capture_hook
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

__all__ = ["run_eap"]


def _upstream_nodes(arch: Any) -> list[dict[str, Any]]:
    """Enumerate upstream nodes: embedding + each layer's attn / mlp output."""
    nodes: list[dict[str, Any]] = []
    if arch.embed_path:
        nodes.append({
            "node": "embed", "path": arch.embed_path, "layer": -1, "type": "embed",
        })
    for li in arch.layer_infos:
        if li.attn_path:
            nodes.append({
                "node": f"L{li.index}.attn", "path": li.attn_path,
                "layer": li.index, "type": "attn",
            })
        if li.mlp_path:
            nodes.append({
                "node": f"L{li.index}.mlp", "path": li.mlp_path,
                "layer": li.index, "type": "mlp",
            })
    return nodes


def _metric_scalar(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    """Logit of *target_idx* at the last position (the AtP/logit_diff scalar)."""
    if logits.dim() == 3:
        return logits[0, -1, target_idx]
    if logits.dim() == 2:
        return logits[0, target_idx]
    return logits[target_idx]


def run_eap(
    model: Model,
    clean: Any,
    corrupted: Any,
    *,
    ig_steps: int = 0,
    top_k_edges: int | None = 30,
    metric: str = "logit_diff",
    render: bool = True,
) -> dict[str, Any]:
    """Score every (component → residual-stream) edge for a clean/corrupted pair.

    Parameters
    ----------
    ig_steps:
        0 (default) = plain EAP: gradients from one corrupted backward.
        m > 0 = EAP-IG: gradients averaged over *m* backwards at
        embeddings interpolated from corrupted toward clean (midpoint
        rule). 5 is a reasonable starting point.
    top_k_edges:
        Number of edges (by absolute score) returned in ``edges``.
        ``None`` or 0 returns all. ``nodes`` always lists every node.
    metric:
        Only ``"logit_diff"`` has an EAP formulation.

    Returns
    -------
    dict with:
        ``edges`` — ``{"src", "dst", "src_layer", "dst_layer", "score",
            "rank"}`` sorted by absolute score.
        ``nodes`` — per-node totals (the edge at each node's own
            injection layer ≈ its AtP score), every node, sorted by
            absolute score.
        ``meta`` — method / ig_steps / pass counts / caveat.
    """
    from interpkit.core.render import render_eap
    from interpkit.core.support_matrix import check_op_supported

    _validate_enum(metric, VALID_EAP_METRICS, "metric")
    if ig_steps < 0:
        raise ValueError(f"ig_steps must be >= 0, got {ig_steps}.")
    check_op_supported("eap", model.arch_info)

    arch = model.arch_info
    if arch.is_encoder_decoder:
        raise OperationNotSupportedForArchitecture(
            "eap is not yet supported on encoder-decoder models: encoder and "
            "decoder maintain separate residual streams with different "
            "gradient semantics. Documented deferral (see ops/eap.py)."
        )
    blocks = arch.blocks
    block_paths = [b.path for b in blocks]
    if len(set(block_paths)) < len(block_paths):
        raise OperationNotSupportedForArchitecture(
            "eap is not yet supported on shared-weight models (ALBERT-style): "
            "per-block gradient captures would alias the same physical module. "
            "Documented deferral (see ops/eap.py)."
        )
    nodes = _upstream_nodes(arch)
    if not nodes or not block_paths:
        raise OperationNotSupportedForArchitecture(
            "eap requires detected per-layer attention/MLP structure and "
            "residual blocks; this model resolved neither."
        )

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)
    if isinstance(clean_input, dict) and isinstance(corrupted_input, dict):
        ids_c = clean_input.get("input_ids")
        ids_r = corrupted_input.get("input_ids")
        if ids_c is not None and ids_r is not None and ids_c.shape != ids_r.shape:
            raise ValueError(
                f"eap requires token-aligned clean/corrupted pairs (same "
                f"length); got {tuple(ids_c.shape)} vs {tuple(ids_r.shape)}. "
                f"Rephrase the pair so both sides tokenize to the same number "
                f"of tokens."
            )
        # prepare_pair pads text pairs to a common shape, so a real length
        # mismatch surfaces as differing attention-mask sums. Padding is
        # poison for EAP specifically: edge scores sum Δ·grad over *all*
        # positions, so pad-position deltas would contaminate every edge.
        mask_c = clean_input.get("attention_mask")
        mask_r = corrupted_input.get("attention_mask")
        if mask_c is not None and mask_r is not None:
            len_c, len_r = int(mask_c.sum().item()), int(mask_r.sum().item())
            if len_c != len_r:
                raise ValueError(
                    f"eap requires token-aligned clean/corrupted pairs (same "
                    f"length); clean tokenizes to {len_c} tokens but corrupted "
                    f"to {len_r} (the pair was padded to match). Rephrase the "
                    f"pair so both sides tokenize to the same number of tokens."
                )

    node_paths = [n["path"] for n in nodes]

    # ── Pass 1: clean forward — upstream node outputs + target token ──
    clean_acts: dict[str, torch.Tensor] = {}
    handles = []
    for path in node_paths:
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

    clean_flat = clean_logits.view(-1, clean_logits.shape[-1]).float()
    if clean_flat.shape[0] > 1:
        clean_flat = clean_flat[-1:]
    target_idx = int(clean_flat[0].argmax().item())

    # ── Pass 2: corrupted — upstream outputs + residual-stream grads ──
    corrupted_acts: dict[str, torch.Tensor] = {}
    grad_avgs: dict[str, torch.Tensor] = {}
    n_backwards = max(1, ig_steps)

    try:
        if ig_steps == 0:
            grad_store: dict[str, torch.Tensor] = {}
            handles = []
            for path in node_paths:
                try:
                    mod = _get_module(model._model, path)
                except (AttributeError, IndexError, KeyError, TypeError):
                    continue
                handles.append(register_capture_hook(mod, corrupted_acts, path))
            for path in block_paths:
                handles.append(
                    register_grad_capture_hook(
                        _get_module(model._model, path), grad_store, path,
                    )
                )
            try:
                logits = model._forward_with_grad(corrupted_input).float()
            finally:
                for h in handles:
                    h.remove()
            _metric_scalar(logits, target_idx).backward()
            for path, t in grad_store.items():
                if t.grad is not None:
                    grad_avgs[path] = t.grad.float()
        else:
            # EAP-IG: corrupted activations from a grad-free pass, then
            # ig_steps interpolated backwards for the gradients.
            handles = [
                register_capture_hook(_get_module(model._model, p), corrupted_acts, p)
                for p in node_paths
            ]
            try:
                with torch.no_grad():
                    model._forward(corrupted_input)
            finally:
                for h in handles:
                    h.remove()

            embed_path = arch.embed_path
            if embed_path is None or embed_path not in clean_acts or embed_path not in corrupted_acts:
                raise OperationNotSupportedForArchitecture(
                    "eap with ig_steps requires a resolved embedding module "
                    "(interpolation anchor); this model has none."
                )
            clean_embed = clean_acts[embed_path]

            grad_sums: dict[str, torch.Tensor] = {}
            embed_mod = _get_module(model._model, embed_path)
            for k in range(ig_steps):
                alpha = (k + 0.5) / ig_steps  # midpoint rule

                def _interp(t: torch.Tensor, _ctx: Any, _a: float = alpha) -> torch.Tensor:
                    return t + _a * (clean_embed.to(t.device, t.dtype) - t)

                iv = FnIntervention(embed_path, fn=_interp)
                grad_store = {}
                handles = [embed_mod.register_forward_hook(iv.build_hook(None))]
                handles += [
                    register_grad_capture_hook(
                        _get_module(model._model, p), grad_store, p,
                    )
                    for p in block_paths
                ]
                try:
                    logits = model._forward_with_grad(corrupted_input).float()
                finally:
                    for h in handles:
                        h.remove()
                _metric_scalar(logits, target_idx).backward()
                for path, t in grad_store.items():
                    if t.grad is None:
                        continue
                    g = t.grad.float()
                    grad_sums[path] = grad_sums.get(path, torch.zeros_like(g)) + g
            grad_avgs = {p: g / ig_steps for p, g in grad_sums.items()}
    finally:
        # Multiple backwards leave parameter .grad buffers behind.
        model._model.zero_grad(set_to_none=True)

    # ── Edge scores: grad_resid_l · Δu for l >= layer(u), fp32 ──
    edges: list[dict[str, Any]] = []
    node_totals: list[dict[str, Any]] = []
    for n in nodes:
        path = n["path"]
        if path not in clean_acts or path not in corrupted_acts:
            continue
        delta = (clean_acts[path].float() - corrupted_acts[path].float())
        injection_score: float | None = None
        for l_idx, block_path in enumerate(block_paths):
            if l_idx < n["layer"]:
                continue
            grad = grad_avgs.get(block_path)
            if grad is None or grad.shape != delta.shape:
                continue
            score = float((grad * delta.to(grad.device)).sum().item())
            edges.append({
                "src": n["node"],
                "dst": f"resid_{l_idx}",
                "src_layer": n["layer"],
                "dst_layer": l_idx,
                "score": score,
            })
            if injection_score is None:
                injection_score = score
        node_totals.append({
            "node": n["node"],
            "module": path,
            "layer": n["layer"],
            "type": n["type"],
            "score": injection_score if injection_score is not None else float("nan"),
        })

    def _abs_key(score: float) -> float:
        return abs(score) if score == score else -1.0  # NaN sorts last

    edges.sort(key=lambda e: _abs_key(e["score"]), reverse=True)
    for i, e in enumerate(edges):
        e["rank"] = i
    node_totals.sort(key=lambda n: _abs_key(n["score"]), reverse=True)

    all_edge_count = len(edges)
    if top_k_edges:
        edges = edges[:top_k_edges]

    result: dict[str, Any] = {
        "edges": edges,
        "nodes": node_totals,
        "meta": {
            "method": "eap-ig" if ig_steps > 0 else "eap",
            "ig_steps": ig_steps,
            "metric": metric,
            "n_edges": all_edge_count,
            "n_nodes": len(node_totals),
            "n_forward_passes": 2 + ig_steps,
            "n_backward_passes": n_backwards,
            "caveat": (
                "first-order approximation on the additive residual stream; "
                "verify selected circuits causally (find_circuit does this "
                "automatically)"
            ),
        },
    }

    if render:
        render_eap(result)
    return result
