"""Runtime forward-hook probes.

Structural detection that needs a short forward pass: finding the input
consumer / output producer, per-block residual + attention metadata, and
the model output shape. All built on :func:`interpkit.core.arch.tree._safe_forward`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from interpkit.core.arch.tree import _safe_forward, module_at_path


def _walker_find_input_consumer(model: nn.Module, sample_input: Any) -> nn.Module | None:
    """Find the first leaf module whose forward consumes the raw input.

    Used when HF accessors and conventions both fail to identify the
    embedding. Hooks every leaf module, runs one forward, and returns
    the first one whose first input tensor matches the model input by
    identity / shape.
    """
    captured: list[tuple[str, nn.Module]] = []

    def make_hook(name: str, mod: nn.Module):
        def hook(_m: nn.Module, _inp: Any, _out: Any) -> None:
            captured.append((name, mod))

        return hook

    handles = []
    for name, mod in model.named_modules():
        if name and len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(make_hook(name, mod)))

    try:
        _safe_forward(model, sample_input)
    finally:
        for h in handles:
            h.remove()

    return captured[0][1] if captured else None


def _walker_find_output_producer(
    model: nn.Module, sample_input: Any, *, num_classes: int | None = None,
) -> nn.Module | None:
    """Find the trailing Linear/Conv that produces the output logits.

    Walks named modules in reverse, returning the last ``nn.Linear`` (or
    ``nn.Conv2d`` for CNN classifiers) whose ``out_features`` matches
    ``num_classes`` if provided, otherwise simply the last weight-bearing
    Linear / Conv that's not inside the block stack.
    """
    candidates: list[nn.Module] = []
    for _name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            out = getattr(mod, "out_features", None) or getattr(mod, "out_channels", None)
            if num_classes is None or out == num_classes:
                candidates.append(mod)
    return candidates[-1] if candidates else None


def _runtime_detect_residual(block: nn.Module, sample_input: Any) -> bool:
    """Standalone residual-detection on an isolated block (rare path).

    Most callers should use :func:`_detect_block_metadata`, which detects
    residuality for many blocks during a single full-model forward. This
    standalone variant is used only when calling the block in isolation
    is feasible (e.g. unit tests with synthetic blocks).

    A block is residual iff ``output ≈ input + transform(input)``: the
    delta from input to output is small relative to the output norm.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_m: nn.Module, inp: tuple, _out: Any) -> None:
        if inp and isinstance(inp[0], torch.Tensor):
            captured["inp"] = inp[0].detach().clone()

    h = block.register_forward_pre_hook(hook)
    try:
        out = _safe_forward(block, sample_input)
    except Exception:  # noqa: BLE001 — conservatively assume non-residual
        return False
    finally:
        h.remove()

    return _is_residual_pair(captured.get("inp"), out)


def _is_residual_pair(inp: torch.Tensor | None, out: Any) -> bool:
    """Check if (input, output) tensor pair looks like a residual update."""
    if inp is None:
        return False
    if isinstance(out, tuple):
        out = out[0] if out else None
    if not isinstance(out, torch.Tensor):
        return False
    if out.shape != inp.shape:
        return False
    delta_norm = (out - inp).norm().item()
    out_norm = out.norm().item()
    if out_norm < 1e-12:
        return False
    return (delta_norm / out_norm) < 0.7


def _classify_block_mechanism(block: nn.Module) -> str:
    """Label a block's computational mechanism from its structure alone.

    Returns one of ``"attention" | "recurrent" | "ssm" | "conv" | "mlp" |
    "unknown"``. Detection is by structural fingerprint (module types +
    the canonical attribute/param vocabularies in
    :mod:`interpkit.core.arch.names`), NOT by model class name — so novel
    HF architectures are described without per-model code.

    Priority: attention (fused or separate Q/K/V children, or
    ``MultiheadAttention``) → ssm (Conv1d + state param) → recurrent
    (RG-LRU / time-mix submodule names) → conv (Conv2d) → mlp (any Linear)
    → unknown. The attention check uses the full QKV vocabulary (so it
    matches ``q_proj`` / ``q_lin`` / ``query`` / ``c_attn`` / ``qkv`` …) plus
    the attention module-name vocabulary, matching the per-layer attention
    resolution. ``BlockSpec.has_attention`` is derived from this (it is
    ``mechanism == "attention"``), so the two fields cannot disagree.
    """
    from interpkit.core.arch.names import (
        ATTN_NAMES,
        FUSED_QKV_NAMES,
        K_PROJ_NAMES,
        Q_PROJ_NAMES,
        RECURRENT_NAMES,
        SSM_STATE_PARAM_NAMES,
        V_PROJ_NAMES,
    )

    # Attention: fused QKV, or separate Q+K+V children, or MultiheadAttention,
    # or a submodule whose attribute name marks an attention module (catches
    # variants whose projections fall outside the Q/K/V vocab, e.g. DeBERTa's
    # DisentangledSelfAttention with query_proj/key_proj/value_proj).
    for name, sub in block.named_modules():
        if isinstance(sub, nn.MultiheadAttention):
            return "attention"
        if name and name.rsplit(".", 1)[-1] in ATTN_NAMES:
            return "attention"
        kids = {n for n, _ in sub.named_children()}
        if kids & FUSED_QKV_NAMES:
            return "attention"
        if (kids & Q_PROJ_NAMES) and (kids & K_PROJ_NAMES) and (kids & V_PROJ_NAMES):
            return "attention"

    # SSM (Mamba/Mamba2): a 1-D causal conv plus a learned state param.
    if any(isinstance(m, nn.Conv1d) for m in block.modules()):
        param_tails = {n.rsplit(".", 1)[-1] for n, _ in block.named_parameters()}
        if param_tails & SSM_STATE_PARAM_NAMES:
            return "ssm"

    # Recurrence (Griffin RG-LRU, RWKV time-mix): a submodule whose attribute
    # name marks a recurrence.
    for name, _sub in block.named_modules():
        if name and name.rsplit(".", 1)[-1] in RECURRENT_NAMES:
            return "recurrent"

    if any(isinstance(m, nn.Conv2d) for m in block.modules()):
        return "conv"
    if any(isinstance(m, nn.Linear) for m in block.modules()):
        return "mlp"
    return "unknown"


def _detect_block_metadata(
    model: nn.Module, block_paths: list[str], sample_input: Any,
) -> dict[str, dict[str, bool]]:
    """In a single full-model forward, detect ``has_residual``, ``has_attention``
    and ``mechanism`` for every block path. Avoids the cost (and correctness
    issues) of calling each block with a sample input it wasn't designed to
    receive.
    """
    metadata: dict[str, dict[str, Any]] = {p: {"has_residual": False, "has_attention": False, "mechanism": "unknown"} for p in block_paths}
    pairs: dict[str, tuple[torch.Tensor | None, Any]] = {p: (None, None) for p in block_paths}
    handles = []

    def make_pre_hook(path: str):
        def fn(_m: nn.Module, inp: tuple) -> None:
            if inp and isinstance(inp[0], torch.Tensor):
                pairs[path] = (inp[0].detach().clone(), pairs[path][1])

        return fn

    def make_post_hook(path: str):
        def fn(_m: nn.Module, _inp: Any, out: Any) -> None:
            pairs[path] = (pairs[path][0], out)

        return fn

    for path in block_paths:
        try:
            block = module_at_path(model, path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        handles.append(block.register_forward_pre_hook(make_pre_hook(path)))
        handles.append(block.register_forward_hook(make_post_hook(path)))
        # Mechanism detection is purely structural (no forward pass).
        # ``has_attention`` is DERIVED from the mechanism so the two fields
        # can never disagree — a single source of truth via the full
        # ``names.py`` vocabulary, instead of the old narrow QKV fast-path
        # that missed q_lin / query_proj-style attention (DistilBERT, T5,
        # DeBERTa).
        mech = _classify_block_mechanism(block)
        metadata[path]["mechanism"] = mech
        metadata[path]["has_attention"] = mech == "attention"

    try:
        _safe_forward(model, sample_input)
    except Exception:
        pass
    finally:
        for h in handles:
            h.remove()

    for path, (inp, out) in pairs.items():
        metadata[path]["has_residual"] = _is_residual_pair(inp, out)
    return metadata


def _probe_output_shape(model: nn.Module, sample_input: Any) -> tuple[int, ...] | None:
    """Run a forward pass and return the model output's shape (or None)."""
    try:
        out = _safe_forward(model, sample_input)
    except Exception:
        return None
    if hasattr(out, "logits"):
        out = out.logits
    if isinstance(out, torch.Tensor):
        return tuple(out.shape)
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return tuple(out[0].shape)
    return None
