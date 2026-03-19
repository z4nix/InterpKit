"""circuits — residual stream decomposition, OV/QK analysis, and composition scores."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console

from interpkit.ops.patch import _get_module
from interpkit.ops.dla import _find_attn_submodule, _find_mlp_submodule
from interpkit.ops.heads import _find_output_proj

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


# ------------------------------------------------------------------
# Residual stream decomposition
# ------------------------------------------------------------------


def run_decompose(
    model: "Model",
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
    hooks: list[torch.utils.hooks.RemovableHook] = []

    for layer_name in arch.layer_names:
        layer_mod = _get_module(model._model, layer_name)

        attn_child = _find_attn_submodule(layer_mod)
        if attn_child is not None:
            key = f"{layer_name}::attn"

            def _make_hook(k: str):
                def hook_fn(_mod, _inp, output):
                    t = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                    )
                    if t is not None:
                        component_outputs[k] = t.detach().float()
                return hook_fn

            hooks.append(attn_child[1].register_forward_hook(_make_hook(key)))

        mlp_child = _find_mlp_submodule(layer_mod)
        if mlp_child is not None:
            key = f"{layer_name}::mlp"

            def _make_mlp_hook(k: str):
                def hook_fn(_mod, _inp, output):
                    t = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                    )
                    if t is not None:
                        component_outputs[k] = t.detach().float()
                return hook_fn

            hooks.append(mlp_child[1].register_forward_hook(_make_mlp_hook(key)))

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
    model: "Model",
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

    num_heads = arch.num_attention_heads
    layer_name = arch.layer_names[layer]
    layer_mod = _get_module(model._model, layer_name)

    attn_child = _find_attn_submodule(layer_mod)
    if attn_child is None:
        raise ValueError(f"No attention submodule found in layer {layer}.")

    attn_mod = attn_child[1]

    # Find V projection and O projection weights
    w_v = _find_proj_weight(attn_mod, ("v_proj", "value", "c_attn"), "v")
    _, _, proj_mod = _find_output_proj(model._model, layer_name)

    if w_v is None:
        raise ValueError(f"Could not find V projection weight in layer {layer}.")
    if proj_mod is None or not hasattr(proj_mod, "weight"):
        raise ValueError(f"Could not find output projection weight in layer {layer}.")

    # Normalise W_O to (d_model, num_heads * head_dim)
    raw_w_o = proj_mod.weight.float()
    is_conv1d = type(proj_mod).__name__ == "Conv1D"
    w_o = raw_w_o.T if is_conv1d else raw_w_o  # -> (d_model, H*D_h)

    w_v = w_v.float()  # (H*D_h, d_model) from _find_proj_weight

    head_dim = w_v.shape[0] // num_heads
    d_model = w_o.shape[0]

    heads: list[dict[str, Any]] = []
    for h in range(num_heads):
        start = h * head_dim
        end = start + head_dim

        w_v_h = w_v[start:end, :]  # (head_dim, d_model)
        w_o_h = w_o[:, start:end]  # (d_model, head_dim)

        # W_OV = W_O_h @ W_V_h : (d_model, d_model)
        w_ov = w_o_h @ w_v_h  # (d_model, d_model)

        svd_vals = torch.linalg.svdvals(w_ov)
        fro_norm = w_ov.norm().item()
        approx_rank = int((svd_vals > 0.01 * svd_vals[0]).sum().item())

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
    model: "Model",
    *,
    layer: int,
) -> dict[str, Any]:
    """Analyse the QK (query-key) circuit of each attention head.

    Computes the effective QK matrix ``W_QK = W_Q^T @ W_K`` for each head.
    """
    arch = model.arch_info
    if not arch.layer_names or not arch.num_attention_heads:
        raise ValueError("QK analysis requires detected layer structure and head count.")

    num_heads = arch.num_attention_heads
    layer_name = arch.layer_names[layer]
    layer_mod = _get_module(model._model, layer_name)

    attn_child = _find_attn_submodule(layer_mod)
    if attn_child is None:
        raise ValueError(f"No attention submodule found in layer {layer}.")

    attn_mod = attn_child[1]

    w_q = _find_proj_weight(attn_mod, ("q_proj", "query", "c_attn"), "q")
    w_k = _find_proj_weight(attn_mod, ("k_proj", "key", "c_attn"), "k")

    if w_q is None or w_k is None:
        raise ValueError(f"Could not find Q/K projection weights in layer {layer}.")

    w_q = w_q.float()  # (H*D_h, d_model) from _find_proj_weight
    w_k = w_k.float()

    head_dim = w_q.shape[0] // num_heads

    heads: list[dict[str, Any]] = []
    for h in range(num_heads):
        start = h * head_dim
        end = start + head_dim

        w_q_h = w_q[start:end, :]  # (head_dim, d_model)
        w_k_h = w_k[start:end, :]  # (head_dim, d_model)

        # W_QK = W_Q_h^T @ W_K_h : (d_model, d_model)
        w_qk = w_q_h.T @ w_k_h

        svd_vals = torch.linalg.svdvals(w_qk)
        fro_norm = w_qk.norm().item()
        approx_rank = int((svd_vals > 0.01 * svd_vals[0]).sum().item())

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
    model: "Model",
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

    num_heads = arch.num_attention_heads

    # Source layer: build W_OV = W_O @ W_V for each head
    src_layer_name = arch.layer_names[src_layer]
    src_mod = _get_module(model._model, src_layer_name)
    src_attn = _find_attn_submodule(src_mod)
    if src_attn is None:
        raise ValueError(f"No attention in source layer {src_layer}.")

    _, _, src_proj = _find_output_proj(model._model, src_layer_name)
    if src_proj is None or not hasattr(src_proj, "weight"):
        raise ValueError(f"No output projection in source layer {src_layer}.")

    raw_w_o_src = src_proj.weight.float()
    is_conv1d = type(src_proj).__name__ == "Conv1D"
    w_o_src = raw_w_o_src.T if is_conv1d else raw_w_o_src  # -> (d_model, H*D_h)
    d_model = w_o_src.shape[0]
    head_dim = w_o_src.shape[1] // num_heads

    w_v_src = _find_proj_weight(src_attn[1], ("v_proj", "value", "c_attn"), "v")
    if w_v_src is None:
        raise ValueError(f"Could not find V projection in source layer {src_layer}.")
    w_v_src = w_v_src.float()  # (H*D_h, d_model)

    # Destination layer: get the composition target projection
    dst_layer_name = arch.layer_names[dst_layer]
    dst_mod = _get_module(model._model, dst_layer_name)
    dst_attn = _find_attn_submodule(dst_mod)
    if dst_attn is None:
        raise ValueError(f"No attention in destination layer {dst_layer}.")

    proj_names = {"q": ("q_proj", "query"), "k": ("k_proj", "key"), "v": ("v_proj", "value")}
    if comp_type not in proj_names:
        raise ValueError(f"comp_type must be 'q', 'k', or 'v', got {comp_type!r}")

    w_dst = _find_proj_weight(dst_attn[1], (*proj_names[comp_type], "c_attn"), comp_type)
    if w_dst is None:
        raise ValueError(f"Could not find {comp_type.upper()} projection in destination layer {dst_layer}.")

    w_dst = w_dst.float()

    # score(i,j) = || W_dst_i @ W_OV_j ||_F / (|| W_dst_i ||_F * || W_OV_j ||_F)
    scores = torch.zeros(num_heads, num_heads)

    for i in range(num_heads):
        dst_start = i * head_dim
        dst_end = dst_start + head_dim
        w_dst_h = w_dst[dst_start:dst_end, :]  # (head_dim, d_model)
        dst_norm = w_dst_h.norm()

        for j in range(num_heads):
            src_start = j * head_dim
            src_end = src_start + head_dim
            w_o_h = w_o_src[:, src_start:src_end]   # (d_model, head_dim)
            w_v_h = w_v_src[src_start:src_end, :]    # (head_dim, d_model)
            w_ov_h = w_o_h @ w_v_h                   # (d_model, d_model)

            composition = w_dst_h @ w_ov_h  # (head_dim, d_model)
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
# Weight extraction helpers
# ------------------------------------------------------------------


def _find_proj_weight(
    attn_mod: torch.nn.Module,
    names: tuple[str, ...],
    proj_type: str,
) -> torch.Tensor | None:
    """Find the weight matrix for a Q/K/V projection.

    Returns a weight tensor of shape ``(proj_dim, d_model)`` — i.e. the
    projection maps *from* residual-stream space *to* head space.

    Handles both ``nn.Linear`` (weight shape ``(out, in)``) and GPT-2's
    ``Conv1D`` (weight shape ``(in, out)``).
    """
    for child_name, child_mod in attn_mod.named_modules():
        if not child_name:
            continue
        base = child_name.split(".")[-1]

        if base in ("c_attn", "qkv") and hasattr(child_mod, "weight"):
            w = child_mod.weight
            is_conv1d = type(child_mod).__name__ == "Conv1D"

            if is_conv1d:
                # Conv1D weight: (d_model, 3*d_model) — split along dim=1
                third = w.shape[1] // 3
                if proj_type == "q":
                    return w[:, :third].T  # -> (proj_dim, d_model)
                elif proj_type == "k":
                    return w[:, third:2*third].T
                elif proj_type == "v":
                    return w[:, 2*third:].T
            else:
                # nn.Linear weight: (3*d_model, d_model) — split along dim=0
                third = w.shape[0] // 3
                if proj_type == "q":
                    return w[:third, :]
                elif proj_type == "k":
                    return w[third:2*third, :]
                elif proj_type == "v":
                    return w[2*third:, :]

        if base in names and hasattr(child_mod, "weight"):
            w = child_mod.weight
            is_conv1d = type(child_mod).__name__ == "Conv1D"
            if is_conv1d:
                return w.T  # -> (out_features, in_features)
            return w

    return None


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
