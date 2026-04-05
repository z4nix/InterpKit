"""circuits — residual stream decomposition, OV/QK analysis, and composition scores."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console

from interpkit.core.discovery import ModelArchInfo, _get_mod_by_path, _get_weight, extract_proj_weight
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def _nearest_attention_layer(arch: ModelArchInfo, layer: int) -> int | None:
    """Return the nearest attention layer index, preferring forward."""
    indices = arch.attention_layer_indices
    if not indices:
        return None
    forward = [i for i in indices if i >= layer]
    if forward:
        return forward[0]
    return indices[-1]


def _redirect_to_attention(arch: ModelArchInfo, layer: int, op_name: str) -> int:
    """Validate that *layer* has attention; if not, redirect with a warning."""
    li = arch.layer_infos[layer] if layer < len(arch.layer_infos) else None
    if li is not None and li.attn_path is not None:
        return layer

    alt = _nearest_attention_layer(arch, layer)
    if alt is None:
        raise ValueError(
            f"{op_name}: this model has no attention layers "
            f"(all layers are {li.layer_type if li else 'unknown'})."
        )

    reason = (
        f"layer_type='{li.layer_type}'" if li else "layer not found"
    )
    warnings.warn(
        f"{op_name}: Layer {layer} has no attention ({reason}). "
        f"Redirecting to nearest attention layer {alt}.",
        stacklevel=3,
    )
    return alt


# ------------------------------------------------------------------
# Residual stream decomposition
# ------------------------------------------------------------------


def run_decompose(
    model: Model,
    input_data: Any,
    *,
    position: int = -1,
) -> dict[str, Any]:
    """Decompose the residual stream into per-component contributions.

    For each layer, captures the output of the attention block and MLP
    block (which are added to the residual stream).  Returns the
    contribution of each component at the specified token position.

    Returns
    -------
    dict with:
        ``components`` — list of ``{"name", "layer", "type", "vector", "norm"}``
        ``residual`` — the final residual stream vector at the given position
        ``position`` — the analysed position index
    """
    from interpkit.core.render import render_decompose

    arch = model.arch_info
    if not arch.layer_names:
        raise ValueError("Decomposition requires detected layer structure.")

    model_input = model._prepare(input_data)

    component_outputs: dict[str, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    for li in arch.layer_infos:
        if li.attn_path:
            attn_mod = _get_module(model._model, li.attn_path)
            key = f"{li.name}::attn"

            def _make_hook(k: str):
                def hook_fn(_mod, _inp, output):
                    t = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                    )
                    if t is not None:
                        component_outputs[k] = t.detach().float()
                return hook_fn

            hooks.append(attn_mod.register_forward_hook(_make_hook(key)))

        if li.mlp_path:
            mlp_mod = _get_module(model._model, li.mlp_path)
            key = f"{li.name}::mlp"

            def _make_mlp_hook(k: str):
                def hook_fn(_mod, _inp, output):
                    t = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                    )
                    if t is not None:
                        component_outputs[k] = t.detach().float()
                return hook_fn

            hooks.append(mlp_mod.register_forward_hook(_make_mlp_hook(key)))

    # Also capture the final residual (output of last layer)
    last_layer = arch.layer_names[-1]
    residual_output: list[torch.Tensor] = []

    def _residual_hook(_mod, _inp, output):
        t = output if isinstance(output, torch.Tensor) else (
            output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
        )
        if t is not None:
            residual_output.append(t.detach().float())

    last_mod = _get_module(model._model, last_layer)
    hooks.append(last_mod.register_forward_hook(_residual_hook))

    with torch.no_grad():
        model._forward(model_input)

    for h in hooks:
        h.remove()

    components: list[dict[str, Any]] = []
    for key, tensor in component_outputs.items():
        layer_name, comp_type = key.rsplit("::", 1)
        layer_match = re.search(r"\.(\d+)", layer_name)
        layer_idx = int(layer_match.group(1)) if layer_match else 0

        if tensor.dim() == 3:
            vec = tensor[0, position, :]
        elif tensor.dim() == 2:
            vec = tensor[position, :]
        else:
            vec = tensor

        components.append({
            "name": f"L{layer_idx}.{'attn' if comp_type == 'attn' else 'mlp'}",
            "layer": layer_idx,
            "type": comp_type,
            "vector": vec,
            "norm": vec.norm().item(),
        })

    components.sort(key=lambda c: (c["layer"], 0 if c["type"] == "attn" else 1))

    residual = None
    if residual_output:
        r = residual_output[0]
        if r.dim() == 3:
            residual = r[0, position, :]
        elif r.dim() == 2:
            residual = r[position, :]
        else:
            residual = r

    result = {
        "components": components,
        "residual": residual,
        "position": position,
    }

    render_decompose(result)
    return result


# ------------------------------------------------------------------
# OV / QK matrix analysis
# ------------------------------------------------------------------


def run_ov_scores(
    model: Model,
    *,
    layer: int,
) -> dict[str, Any]:
    """Analyse the OV (output-value) circuit of each attention head.

    Computes the effective OV matrix ``W_OV = W_V @ W_O`` for each head
    and reports its top singular values, Frobenius norm, and the approximate
    rank (number of singular values > 1% of the largest).

    Returns
    -------
    dict with ``heads`` (list per head) and ``layer``.
    """
    arch = model.arch_info
    if not arch.layer_names or not arch.num_attention_heads:
        raise ValueError("OV analysis requires detected layer structure and head count.")

    layer = _redirect_to_attention(arch, layer, "ov_scores")
    num_heads = arch.num_attention_heads
    li = arch.layer_infos[layer]

    # Find V projection weight via centralised extractor
    w_v = extract_proj_weight(
        model._model, li, "v", num_heads, arch.num_key_value_heads,
    )

    # Find O projection weight
    if li.o_proj_path is None:
        raise ValueError(f"Could not find output projection weight in layer {layer}.")
    proj_mod = _get_mod_by_path(model._model, li.o_proj_path)

    if w_v is None:
        raise ValueError(f"Could not find V projection weight in layer {layer}.")
    if not hasattr(proj_mod, "weight"):
        raise ValueError(f"Could not find output projection weight in layer {layer}.")

    # Normalise W_O to (d_model, num_heads * head_dim)
    raw_w_o = _get_weight(proj_mod).float()
    is_conv1d = type(proj_mod).__name__ == "Conv1D"
    w_o = raw_w_o.T if is_conv1d else raw_w_o  # -> (d_model, H*D_h)

    w_v = w_v.float()  # (kv_heads*D_h, d_model)

    # GQA: V may have fewer head slices than O
    num_kv_heads = arch.num_key_value_heads or num_heads
    head_dim = int(w_o.shape[1]) // num_heads

    heads: list[dict[str, Any]] = []
    for h in range(num_heads):
        o_start = h * head_dim
        o_end = o_start + head_dim

        # Map Q head to its KV head group
        kv_idx = h * num_kv_heads // num_heads
        v_start = kv_idx * head_dim
        v_end = v_start + head_dim

        w_v_h = w_v[v_start:v_end, :]  # (head_dim, d_model)
        w_o_h = w_o[:, o_start:o_end]  # (d_model, head_dim)

        # W_OV = W_O_h @ W_V_h : (d_model, d_model)
        w_ov = w_o_h @ w_v_h  # (d_model, d_model)

        svd_vals = torch.linalg.svdvals(w_ov.cpu())
        fro_norm = w_ov.norm().item()
        approx_rank = int((svd_vals > 0.01 * svd_vals[0]).sum().item()) if svd_vals[0] > 0 else 0

        heads.append({
            "head": h,
            "frobenius_norm": fro_norm,
            "top_singular_values": svd_vals[:5].tolist(),
            "approx_rank": approx_rank,
            "w_ov": w_ov,
        })

    result = {"layer": layer, "heads": heads}
    _render_ov_qk(result, "OV")
    return result


def run_qk_scores(
    model: Model,
    *,
    layer: int,
) -> dict[str, Any]:
    """Analyse the QK (query-key) circuit of each attention head.

    Computes the effective QK matrix ``W_QK = W_Q^T @ W_K`` for each head.
    """
    arch = model.arch_info
    if not arch.layer_names or not arch.num_attention_heads:
        raise ValueError("QK analysis requires detected layer structure and head count.")

    layer = _redirect_to_attention(arch, layer, "qk_scores")
    num_heads = arch.num_attention_heads
    li = arch.layer_infos[layer]

    w_q = extract_proj_weight(
        model._model, li, "q", num_heads, arch.num_key_value_heads,
    )
    w_k = extract_proj_weight(
        model._model, li, "k", num_heads, arch.num_key_value_heads,
    )

    if w_q is None or w_k is None:
        raise ValueError(f"Could not find Q/K projection weights in layer {layer}.")

    w_q = w_q.float()  # (H*D_h, d_model)
    w_k = w_k.float()

    # GQA: K may have fewer head slices than Q
    num_kv_heads = arch.num_key_value_heads or num_heads
    head_dim = w_q.shape[0] // num_heads

    heads: list[dict[str, Any]] = []
    for h in range(num_heads):
        q_start = h * head_dim
        q_end = q_start + head_dim

        kv_idx = h * num_kv_heads // num_heads
        k_start = kv_idx * head_dim
        k_end = k_start + head_dim

        w_q_h = w_q[q_start:q_end, :]  # (head_dim, d_model)
        w_k_h = w_k[k_start:k_end, :]  # (head_dim, d_model)

        # W_QK = W_Q_h^T @ W_K_h : (d_model, d_model)
        w_qk = w_q_h.T @ w_k_h

        svd_vals = torch.linalg.svdvals(w_qk.cpu())
        fro_norm = w_qk.norm().item()
        approx_rank = int((svd_vals > 0.01 * svd_vals[0]).sum().item()) if svd_vals[0] > 0 else 0

        heads.append({
            "head": h,
            "frobenius_norm": fro_norm,
            "top_singular_values": svd_vals[:5].tolist(),
            "approx_rank": approx_rank,
            "w_qk": w_qk,
        })

    result = {"layer": layer, "heads": heads}
    _render_ov_qk(result, "QK")
    return result


# ------------------------------------------------------------------
# Composition scores
# ------------------------------------------------------------------


def run_composition(
    model: Model,
    *,
    src_layer: int,
    dst_layer: int,
    comp_type: str = "q",
) -> dict[str, Any]:
    """Compute composition scores between heads in two layers.

    Measures how much head *j* in *src_layer* composes with head *i* in
    *dst_layer* via the specified projection (Q, K, or V).

    Uses the full W_OV matrix (``W_O @ W_V``) for the source head per
    Elhage et al., "A Mathematical Framework for Transformer Circuits".

    Parameters
    ----------
    comp_type:
        ``"q"`` for Q-composition, ``"k"`` for K-composition,
        ``"v"`` for V-composition.

    Returns
    -------
    dict with ``scores`` (tensor of shape ``dst_heads x src_heads``),
    ``src_layer``, ``dst_layer``, ``comp_type``.
    """
    arch = model.arch_info
    if not arch.layer_names or not arch.num_attention_heads:
        raise ValueError("Composition analysis requires layer structure and head count.")

    src_layer = _redirect_to_attention(arch, src_layer, "composition (src)")
    dst_layer = _redirect_to_attention(arch, dst_layer, "composition (dst)")

    num_heads = arch.num_attention_heads
    num_kv_heads = arch.num_key_value_heads or num_heads

    # Source layer: build W_OV = W_O @ W_V for each head
    src_li = arch.layer_infos[src_layer]
    if src_li.o_proj_path is None:
        raise ValueError(f"No output projection in source layer {src_layer}.")

    src_proj = _get_mod_by_path(model._model, src_li.o_proj_path)
    if not hasattr(src_proj, "weight"):
        raise ValueError(f"No output projection in source layer {src_layer}.")

    raw_w_o_src = _get_weight(src_proj).float()
    is_conv1d = type(src_proj).__name__ == "Conv1D"
    w_o_src = raw_w_o_src.T if is_conv1d else raw_w_o_src  # -> (d_model, H*D_h)
    head_dim = int(w_o_src.shape[1]) // num_heads

    w_v_src = extract_proj_weight(
        model._model, src_li, "v", num_heads, num_kv_heads,
    )
    if w_v_src is None:
        raise ValueError(f"Could not find V projection in source layer {src_layer}.")
    w_v_src = w_v_src.float()

    # Destination layer: get the composition target projection
    dst_li = arch.layer_infos[dst_layer]

    if comp_type not in ("q", "k", "v"):
        raise ValueError(f"comp_type must be 'q', 'k', or 'v', got {comp_type!r}")

    w_dst = extract_proj_weight(
        model._model, dst_li, comp_type, num_heads, num_kv_heads,
    )
    if w_dst is None:
        raise ValueError(f"Could not find {comp_type.upper()} projection in destination layer {dst_layer}.")

    w_dst = w_dst.float()

    # Determine head counts for the destination projection (K/V may use kv_heads)
    dst_num_heads = num_heads
    if comp_type in ("k", "v"):
        dst_num_heads = num_kv_heads

    dst_head_dim = w_dst.shape[0] // dst_num_heads if dst_num_heads > 0 else head_dim

    # score(i,j) = || W_dst_i @ W_OV_j ||_F / (|| W_dst_i ||_F * || W_OV_j ||_F)
    scores = torch.zeros(dst_num_heads, num_heads)

    for i in range(dst_num_heads):
        dst_start = i * dst_head_dim
        dst_end = dst_start + dst_head_dim
        w_dst_h = w_dst[dst_start:dst_end, :]  # (head_dim, d_model)
        dst_norm = w_dst_h.norm()

        for j in range(num_heads):
            o_start = j * head_dim
            o_end = o_start + head_dim
            w_o_h = w_o_src[:, o_start:o_end]   # (d_model, head_dim)

            # V uses KV head grouping
            kv_idx = j * num_kv_heads // num_heads
            v_start = kv_idx * head_dim
            v_end = v_start + head_dim
            w_v_h = w_v_src[v_start:v_end, :]    # (head_dim, d_model)
            w_ov_h = w_o_h @ w_v_h               # (d_model, d_model)

            composition = w_dst_h @ w_ov_h  # (dst_head_dim, d_model)
            ov_norm = w_ov_h.norm()

            if dst_norm > 0 and ov_norm > 0:
                scores[i, j] = composition.norm() / (dst_norm * ov_norm)

    result = {
        "scores": scores,
        "src_layer": src_layer,
        "dst_layer": dst_layer,
        "comp_type": comp_type,
        "num_heads": num_heads,
    }

    _render_composition(result)
    return result


# ------------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------------


def _render_ov_qk(result: dict[str, Any], matrix_type: str) -> None:
    """Print OV or QK analysis results."""
    from rich.table import Table

    layer = result["layer"]
    heads = result["heads"]

    console.print(f"\n[bold]{matrix_type} Analysis — Layer {layer}[/bold]")

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Head", style="cyan", justify="right")
    table.add_column("Frobenius Norm", justify="right")
    table.add_column("Approx Rank", justify="right")
    table.add_column("Top Singular Values", style="dim")

    for h in heads:
        svs = ", ".join(f"{v:.2f}" for v in h["top_singular_values"][:3])
        table.add_row(
            str(h["head"]),
            f"{h['frobenius_norm']:.3f}",
            str(h["approx_rank"]),
            svs,
        )

    console.print(table)
    console.print()


def _render_composition(result: dict[str, Any]) -> None:
    """Print composition scores."""
    from rich.table import Table

    scores = result["scores"]
    src_layer = result["src_layer"]
    dst_layer = result["dst_layer"]
    comp_type = result["comp_type"].upper()
    num_heads = result["num_heads"]

    console.print(
        f"\n[bold]{comp_type}-Composition: L{src_layer} → L{dst_layer}[/bold]"
    )

    # Find top-5 pairs
    flat = scores.view(-1)
    top_k = min(10, flat.numel())
    top_vals, top_idxs = flat.topk(top_k)

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Dst Head", style="cyan", justify="right")
    table.add_column("Src Head", style="cyan", justify="right")
    table.add_column("Score", justify="right")

    for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
        i = idx // num_heads
        j = idx % num_heads
        table.add_row(
            f"L{dst_layer}.H{i}", f"L{src_layer}.H{j}", f"{val:.4f}",
        )

    console.print(table)
    console.print()
