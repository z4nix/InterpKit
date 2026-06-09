"""Per-layer structure resolution and module role assignment.

Resolves, for each repeated block, the attention/MLP/QKV/output-projection
submodule paths (``LayerInfo``) and tags every module with a semantic
role (``attention`` / ``mlp`` / ``head`` / ``embed`` / ``norm``).
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import cast

import torch
import torch.nn as nn

from interpkit.core.arch.heads import _HEAD_BASE_NAMES
from interpkit.core.arch.names import ALL_QKV_NAMES as _ALL_QKV_NAMES
from interpkit.core.arch.names import ATTN_RE as _ATTN_RE
from interpkit.core.arch.names import FUSED_QKV_NAMES as _FUSED_QKV_NAMES
from interpkit.core.arch.names import K_PROJ_NAMES as _K_PROJ_NAMES
from interpkit.core.arch.names import MLP_RE as _MLP_RE
from interpkit.core.arch.names import O_PROJ_NAMES as _O_PROJ_NAMES
from interpkit.core.arch.names import Q_PROJ_NAMES as _Q_PROJ_NAMES
from interpkit.core.arch.names import V_PROJ_NAMES as _V_PROJ_NAMES
from interpkit.core.arch.tree import _is_norm_module, module_at_path
from interpkit.core.arch.types import LayerInfo, ModuleInfo

logger = logging.getLogger(__name__)

_INTERLEAVED_QKV_CLASSES = frozenset({"GPTNeoXAttention"})


def _find_submodule_recursive(
    parent: nn.Module,
    parent_path: str,
    pattern: re.Pattern[str],
) -> tuple[str, nn.Module] | None:
    """BFS for the shallowest submodule whose base name matches *pattern*."""
    queue: deque[tuple[str, nn.Module]] = deque()
    for name, mod in parent.named_children():
        queue.append((name, mod))
    while queue:
        rel_name, mod = queue.popleft()
        base = rel_name.split(".")[-1]
        if pattern.search(base):
            return f"{parent_path}.{rel_name}", mod
        for child_name, child_mod in mod.named_children():
            queue.append((f"{rel_name}.{child_name}", child_mod))
    return None


def _resolve_projections(
    attn_mod: nn.Module,
    attn_path: str,
    info: LayerInfo,
) -> None:
    """Locate Q/K/V projections inside *attn_mod* and set fields on *info*."""
    for child_name, child_mod in attn_mod.named_modules():
        if not child_name or not hasattr(child_mod, "weight"):
            continue
        base = child_name.split(".")[-1]
        full = f"{attn_path}.{child_name}"
        if base in _FUSED_QKV_NAMES:
            info.qkv_style = "fused"
            info.qkv_proj_path = full
            info.qkv_layout = (
                "interleaved"
                if type(attn_mod).__name__ in _INTERLEAVED_QKV_CLASSES
                else "concatenated"
            )
        elif base in _Q_PROJ_NAMES:
            info.q_proj_path = full
            if info.qkv_style == "unknown":
                info.qkv_style = "separate"
        elif base in _K_PROJ_NAMES:
            info.k_proj_path = full
        elif base in _V_PROJ_NAMES:
            info.v_proj_path = full


def _resolve_output_proj(
    attn_mod: nn.Module | None,
    attn_path: str | None,
    layer_mod: nn.Module,
    layer_path: str,
    info: LayerInfo,
    *,
    hidden_size: int | None = None,
) -> None:
    """Locate the output projection, skipping modules already tagged as Q/K/V.

    N-005 / N-006 / NR-007: when ``hidden_size`` is known, accept any
    o_proj-named candidate whose **output dimension** equals
    ``hidden_size``. The INPUT dim may differ (Qwen3 GQA: inner_dim =
    num_heads*head_dim = 16*128 = 2048, hidden_size = 1024;
    Flan-T5: inner_dim = num_heads*d_kv = 6*64 = 384, d_model = 512),
    but every transformer's attention output projection by definition
    produces ``hidden_size``-dim vectors. This still rejects ALBERT's
    ``embedding_hidden_mapping_in`` (Linear(128, 768)) when used as an
    o_proj candidate on a model whose hidden_size is something else,
    and at any rate ``embedding_hidden_mapping_in`` doesn't match any
    name in ``_O_PROJ_NAMES`` so it never reaches this filter.
    """
    skip = {info.qkv_proj_path, info.q_proj_path, info.k_proj_path, info.v_proj_path}
    skip.discard(None)

    def _has_correct_out_dim(mod: nn.Module) -> bool:
        """Return True iff *mod*'s output dimension equals hidden_size."""
        if hidden_size is None or not hasattr(mod, "weight"):
            return True  # No hint to enforce — accept everything as before.
        w = mod.weight
        if w.dim() < 2:
            return False
        # nn.Linear weight is (out, in); GPT-2's Conv1D stores it
        # transposed as (in, out). Same canonical "out dim equals
        # hidden_size" rule applies — read the right axis.
        is_conv1d = type(mod).__name__ == "Conv1D"
        out_dim = w.shape[1] if is_conv1d else w.shape[0]
        return int(out_dim) == hidden_size

    # Two passes inside attn_mod: prefer o_proj-named children whose
    # out_dim matches hidden_size; only fall back to non-matching if
    # nothing else qualifies (rare; most architectures have at least
    # one such candidate).
    if attn_mod is not None and attn_path is not None:
        fallback: tuple[str, nn.Module] | None = None
        for name, mod in attn_mod.named_modules():
            if not name or not hasattr(mod, "weight"):
                continue
            full = f"{attn_path}.{name}"
            if full in skip:
                continue
            base = name.split(".")[-1]
            if base not in _O_PROJ_NAMES:
                continue
            if _has_correct_out_dim(mod):
                info.o_proj_path = full
                return
            if fallback is None:
                fallback = (full, mod)
        if fallback is not None and hidden_size is None:
            info.o_proj_path = fallback[0]
            return

    fallback = None
    for name, mod in layer_mod.named_modules():
        if not name or not hasattr(mod, "weight"):
            continue
        full = f"{layer_path}.{name}"
        if full in skip:
            continue
        base = name.split(".")[-1]
        if base not in _O_PROJ_NAMES:
            continue
        if _has_correct_out_dim(mod):
            info.o_proj_path = full
            return
        if fallback is None:
            fallback = (full, mod)
    if fallback is not None and hidden_size is None:
        info.o_proj_path = fallback[0]


def _has_qkv_children(mod: nn.Module) -> bool:
    """True if *mod*'s direct children include Q/K/V (or fused) projections."""
    names = {n for n, m in mod.named_children() if hasattr(m, "weight")}
    has_sep = (
        bool(names & _Q_PROJ_NAMES)
        and bool(names & _K_PROJ_NAMES)
        and bool(names & _V_PROJ_NAMES)
    )
    return has_sep or bool(names & _FUSED_QKV_NAMES)


def _probe_for_attention(
    layer_mod: nn.Module,
    layer_path: str,
) -> tuple[str, nn.Module] | None:
    """BFS for the shallowest submodule whose *direct* children are Q/K/V
    projections.  Unlike the old version (which returned the first coarse
    container that had Q/K/V *somewhere* inside), this drills through
    intermediate wrappers like ``nn.ModuleList`` to find the actual
    attention module."""
    queue: deque[tuple[str, nn.Module]] = deque()
    for name, mod in layer_mod.named_children():
        queue.append((f"{layer_path}.{name}", mod))
    while queue:
        path, mod = queue.popleft()
        if _has_qkv_children(mod):
            return path, mod
        for cname, cmod in mod.named_children():
            queue.append((f"{path}.{cname}", cmod))
    return None


def _probe_for_mlp(
    layer_mod: nn.Module,
    layer_path: str,
    attn_path: str | None,
) -> tuple[str, nn.Module] | None:
    """BFS for the shallowest submodule that looks like an MLP: contains 2+
    Linear-like weight modules, is not the attention module, and carries no
    Q/K/V projections.  Drills through intermediate containers like
    ``nn.ModuleList``.

    Fallback: if no container is found, detect "flat MLP" architectures
    (e.g. OPT) where fc1/fc2 sit directly under the layer.  Returns the
    last such Linear as the MLP anchor — hooking it captures the MLP output.
    """
    queue: deque[tuple[str, nn.Module]] = deque()
    for name, mod in layer_mod.named_children():
        queue.append((f"{layer_path}.{name}", mod))
    while queue:
        path, mod = queue.popleft()
        if attn_path and path == attn_path:
            continue
        if _is_norm_module(mod):
            continue
        linear_count = 0
        has_qkv = False
        for sub_name, sub_mod in mod.named_modules():
            if not sub_name or not hasattr(sub_mod, "weight"):
                continue
            if sub_mod.weight.dim() < 2:
                continue
            linear_count += 1
            if sub_name.split(".")[-1] in _ALL_QKV_NAMES:
                has_qkv = True
        if linear_count >= 2 and not has_qkv:
            return path, mod
        for cname, cmod in mod.named_children():
            queue.append((f"{path}.{cname}", cmod))

    # Flat-MLP fallback: collect direct-child Linear modules that are not
    # part of attention, not norms, and not QKV projections.
    flat_fcs: list[tuple[str, nn.Module]] = []
    for name, mod in layer_mod.named_children():
        full_path = f"{layer_path}.{name}"
        if attn_path and (full_path == attn_path or full_path.startswith(attn_path + ".")):
            continue
        if _is_norm_module(mod):
            continue
        if name in _ALL_QKV_NAMES:
            continue
        if not hasattr(mod, "weight"):
            continue
        if cast(torch.Tensor, mod.weight).ndim < 2:
            continue
        flat_fcs.append((full_path, mod))
    if len(flat_fcs) >= 2:
        return flat_fcs[-1]

    return None


def _find_mlp_output_sibling(
    model: nn.Module,
    mlp_in_path: str,
    hidden_size: int,
) -> tuple[str, nn.Module] | None:
    """Find the sibling Linear that outputs ``(hidden_size,)``.

    For ALBERT-style flat MLPs, the resolved ``mlp_path`` initially
    points at the input Linear (``ffn``: hidden→intermediate). The
    canonical MLP anchor (whose forward output is added to the residual
    stream) is the OUTPUT Linear (``ffn_output``: intermediate→hidden).

    Looks at direct siblings of *mlp_in_path* and returns the first
    Linear whose ``out_features == hidden_size`` and whose name is in
    the standard MLP-output suffix set.
    """
    parent_path, _, child_name = mlp_in_path.rpartition(".")
    if not parent_path:
        return None
    try:
        parent = module_at_path(model, parent_path)
    except AttributeError:
        return None
    output_suffixes = frozenset({
        "ffn_output", "fc_out", "out_lin", "fc2", "c_proj", "out_proj",
        "down_proj", "wo", "dense", "output_dense",
    })
    for name, child in parent.named_children():
        if name == child_name:
            continue
        if not isinstance(child, nn.Linear):
            continue
        if child.out_features != hidden_size:
            continue
        if name in output_suffixes:
            return f"{parent_path}.{name}", child
    return None


def _resolve_layer_info(
    model: nn.Module,
    layer_name: str,
    layer_idx: int,
    *,
    block_types: list[str] | None = None,
    hidden_size: int | None = None,
) -> LayerInfo:
    """Build a fully resolved :class:`LayerInfo` for one transformer layer.

    *hidden_size* is forwarded to ``_resolve_output_proj`` so it can
    reject non-square Linear candidates (N-005 / N-006). Defaults to
    ``None`` for callers that don't know the model dimension.
    """
    info = LayerInfo(name=layer_name, index=layer_idx)

    # If the config declares this layer recurrent, mark it early and skip
    # the attention probe (MLP may still exist).
    config_type = (
        block_types[layer_idx]
        if block_types and layer_idx < len(block_types)
        else None
    )

    try:
        layer_mod = module_at_path(model, layer_name)
    except AttributeError:
        logger.debug(
            "Could not resolve module path %r — marking layer %d as recurrent.",
            layer_name, layer_idx,
        )
        info.layer_type = "recurrent"
        return info

    # --- attention detection ---
    attn_result: tuple[str, nn.Module] | None = None
    if config_type not in ("recurrent",):
        attn_result = _find_submodule_recursive(layer_mod, layer_name, _ATTN_RE)
        if attn_result is None:
            attn_result = _probe_for_attention(layer_mod, layer_name)
    if attn_result is not None:
        info.attn_path = attn_result[0]
        _resolve_projections(attn_result[1], attn_result[0], info)

    # --- MLP detection ---
    mlp_result = _find_submodule_recursive(layer_mod, layer_name, _MLP_RE)
    # N-005: ALBERT names its first MLP Linear ``ffn`` (matches _MLP_RE),
    # but the OUTPUT of the MLP is the sibling ``ffn_output``. If the
    # regex match returned a single Linear whose out_features don't equal
    # the model's hidden_size, look for a sibling whose out_features ==
    # hidden_size (the canonical MLP output anchor).
    if mlp_result is not None and hidden_size is not None:
        mlp_path, mlp_mod = mlp_result
        if isinstance(mlp_mod, nn.Linear) and mlp_mod.out_features != hidden_size:
            sibling = _find_mlp_output_sibling(model, mlp_path, hidden_size)
            if sibling is not None:
                mlp_result = sibling
    if mlp_result is None:
        mlp_result = _probe_for_mlp(layer_mod, layer_name, info.attn_path)
    if mlp_result is not None:
        info.mlp_path = mlp_result[0]

    # --- output projection ---
    _resolve_output_proj(
        attn_result[1] if attn_result else None,
        attn_result[0] if attn_result else None,
        layer_mod,
        layer_name,
        info,
        hidden_size=hidden_size,
    )

    # --- N-007: derive attn_inner_path ---
    # The deepest "pre-residual, pre-LN" attention anchor is the immediate
    # parent of ``o_proj``. For BERT/RoBERTa/ELECTRA/ALBERT this drills into
    # ``BertSelfOutput``-style wrapper. For GPT/Llama where attn_path IS the
    # thin attention module already, the parent of o_proj == attn_path so
    # the field collapses to attn_path (no behaviour change).
    if info.o_proj_path is not None:
        parent_path = info.o_proj_path.rsplit(".", 1)[0]
        info.attn_inner_path = parent_path
    elif info.attn_path is not None:
        info.attn_inner_path = info.attn_path

    # --- classify layer type ---
    if config_type == "recurrent":
        info.layer_type = "recurrent"
    elif info.attn_path and info.mlp_path:
        info.layer_type = "standard"
    elif info.attn_path:
        info.layer_type = "attention_only"
    elif info.mlp_path:
        info.layer_type = "mlp_only"
    else:
        info.layer_type = "recurrent"

    return info


def _assign_roles(
    modules: list[ModuleInfo],
    model: nn.Module,
    layer_infos: list[LayerInfo],
    layer_names: list[str],
    unembed_name: str | None,
) -> None:
    """Assign semantic roles using resolved model structure and isinstance checks.

    Priority:
    1. Discovered unembedding → ``"head"``
    2. Structural containment under resolved attn_path → ``"attention"``
    3. Structural containment under resolved mlp_path → ``"mlp"``
    4. ``isinstance`` — ``nn.Embedding`` → ``"embed"``
    5. ``isinstance`` — norm modules → ``"norm"``
    6. Output heads outside the layer stack (base-name set lookup) → ``"head"``
    7. Fallback: class name contains "attention"/"mlp" — for architectures
       where structural resolution didn't find attn/mlp paths.
    """
    mod_map = dict(model.named_modules())

    attn_prefixes = [li.attn_path for li in layer_infos if li.attn_path]
    mlp_prefixes = [li.mlp_path for li in layer_infos if li.mlp_path]

    for mi in modules:
        mod = mod_map.get(mi.name)

        if unembed_name and mi.name == unembed_name:
            mi.role = "head"
            continue

        if any(mi.name == p or mi.name.startswith(p + ".") for p in attn_prefixes):
            mi.role = "attention"
            continue

        if any(mi.name == p or mi.name.startswith(p + ".") for p in mlp_prefixes):
            mi.role = "mlp"
            continue

        if mod is None:
            continue

        if isinstance(mod, nn.Embedding):
            mi.role = "embed"
            continue

        if _is_norm_module(mod):
            mi.role = "norm"
            continue

        base = mi.name.rsplit(".", 1)[-1].lower()
        if (base in _HEAD_BASE_NAMES
                and hasattr(mod, "weight")
                and not any(mi.name.startswith(lp + ".") for lp in layer_names)):
            mi.role = "head"
            continue

    # Fallback: class-name check for architectures where layer_infos
    # couldn't resolve attn/mlp paths (non-standard module tree).
    for mi in modules:
        if mi.role is not None:
            continue
        tn = mi.type_name.lower()
        if "attention" in tn or "selfattn" in tn:
            mi.role = "attention"
        elif "mlp" in tn or "feedforward" in tn:
            mi.role = "mlp"
