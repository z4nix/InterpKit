"""circuits — residual stream decomposition, OV/QK analysis, and composition scores."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console

from interpkit.core.arch import ArchInfo, extract_proj_weight, get_weight, module_at_path
from interpkit.core.inputs import input_seq_len
from interpkit.core.paths import validate_position
from interpkit.core.theme import ACCENT, MUTED

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def _require_attention_layer(arch: ArchInfo, layer: int, op_name: str) -> int:
    """Validate that *layer* exists and has attention. Raise loud on failure (F-020).

    Pre-1.0 ``circuits`` ops silently redirected out-of-range layers to
    the nearest attention layer (sometimes with a UserWarning, sometimes
    completely silent for ``composition``). In notebook sweeps this
    produced misleading data: a user iterating ``layer=0..31`` on a
    12-layer model got layer-11 data for layers 12-31 with no clear signal.

    The 1.0 fix is to fail loud on out-of-range or non-attention layers.
    No silent redirects, no fallbacks.
    """
    n_layers = len(arch.layer_infos)
    if layer < 0 or layer >= n_layers:
        raise IndexError(
            f"{op_name}: layer {layer} out of range. "
            f"This model has {n_layers} layers (valid: 0 to {n_layers - 1})."
        )
    li = arch.layer_infos[layer]
    if li.attn_path is None:
        raise ValueError(
            f"{op_name}: layer {layer} has no attention submodule "
            f"(layer_type={li.layer_type!r}). "
            f"Attention-bearing layers: {arch.attention_layer_indices}."
        )
    return layer


# Backwards-compat name used by older code paths in this module.
def _nearest_attention_layer(arch: ArchInfo, layer: int) -> int | None:
    """Return the nearest attention layer index. Retained only for legacy callers
    inside this module; new code should use :func:`_require_attention_layer`."""
    indices = arch.attention_layer_indices
    if not indices:
        return None
    forward = [i for i in indices if i >= layer]
    if forward:
        return forward[0]
    return indices[-1]


def _redirect_to_attention(arch: ArchInfo, layer: int, op_name: str) -> int:
    """Deprecated alias for :func:`_require_attention_layer` — kept so any
    forgotten internal call sites get the new fail-loud behaviour rather
    than the pre-1.0 silent redirect (F-020).
    """
    li = arch.layer_infos[layer] if 0 <= layer < len(arch.layer_infos) else None
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


# Post-LN architectures: each block applies sublayer + residual + LN, so the
# block's output is LN'd. The clean ``Σ components = residual`` invariant
# does NOT hold for these families because of the per-layer LN
# non-linearity — Σ-of-deltas ≈ pre-LN residual instead. Pre-LN families
# (everything else) get the exact invariant. Detection is by HF
# ``config.model_type`` so it stays robust across HF version bumps.
_POST_LN_MODEL_TYPES = frozenset({
    "bert", "roberta", "distilbert", "electra", "albert",
    "deberta", "deberta-v2", "deberta-v3",
    "xlm-roberta", "camembert", "mobilebert", "convbert",
    "bigbird", "ernie", "luke", "rembert",
})


def _is_post_ln(model: Model) -> bool:
    """Return True iff this model uses BERT-style post-LN blocks."""
    config = getattr(model._model, "config", None)
    if config is None:
        return False
    model_type = getattr(config, "model_type", None)
    return model_type in _POST_LN_MODEL_TYPES


def run_decompose(
    model: Model,
    input_data: Any,
    *,
    position: int = -1,
    exact: bool = False,
) -> dict[str, Any]:
    """Decompose the residual stream into per-component contributions.

    Uses the residual-schema dispatch from
    :mod:`interpkit.core.arch.residual` — components and their summation
    invariant are determined by ``(arch.residual_topology,
    arch.is_shared_layers)``.

    API contract per topology:

    - Pre-LN models (GPT-2, Llama, Qwen, Pythia, OPT-125m, BLOOM):
      ``c["type"] in {"embed", "attn", "mlp"}``. BLOOM uses a hook-
      target adjustment (subtract block input from each submodule
      output before treating it as a delta) so its components match
      the pre-LN shape.
    - Post-LN models (BERT, RoBERTa, DistilBERT, ELECTRA, ALBERT,
      OPT-350m): ``c["type"] in {"embed", "block_delta"}`` — one
      block-level delta per layer. Loses the ``attn``/``mlp`` split
      because the per-block LayerNorm prevents a clean algebraic
      decomposition (explicit tradeoff documented in
      :mod:`interpkit.core.arch.residual`).
    - Seq2seq (T5, Flan-T5, BART): Pre-LN shape rooted at the
      decoder stack only — encoder blocks are not on the residual
      path to ``lm_head``.

    For every supported topology, ``Σ components ≈ residual`` to fp32
    epsilon by construction.

    Parameters
    ----------
    position:
        Token position to analyse. Default ``-1`` (last token).
    exact:
        When ``True`` and the model is non-fp32, briefly cast the
        model to fp32 for the decomposition forward pass. Eliminates
        accumulation error at attention-sink positions but doubles
        peak memory.

    Returns
    -------
    dict with:
        ``components`` — list of
        ``{"name", "layer", "type", "vector", "norm"}`` starting with
        ``L-1.embed``, then per-layer entries whose ``type`` follows
        the per-topology contract above.
        ``residual`` — final residual stream vector at ``position``.
        ``position`` — analysed position index.
        ``precision_note`` — string describing the precision regime.
        ``post_ln`` — convenience bool: ``arch.residual_topology ==
        "post_ln"``.
    """
    from interpkit.core.arch import residual_schema_for
    from interpkit.core.render import render_decompose
    from interpkit.core.support_matrix import check_op_supported

    arch = model.arch_info
    # Gate DeBERTa-v3 (DisentangledSelfAttention) before any forward
    # hook fires — they trigger an unfixable HF transformers broadcast bug.
    check_op_supported("decompose", arch)

    schema = residual_schema_for(arch)
    if schema is None:
        from interpkit.core.exceptions import OperationNotSupportedForArchitecture
        raise OperationNotSupportedForArchitecture(
            f"`decompose` is unsupported for "
            f"(family={arch.family.value}, topology={arch.residual_topology}, "
            f"is_shared_layers={arch.is_shared_layers}). The residual schema "
            f"selector did not find a matching implementation."
        )

    model_input = model._prepare(input_data)

    _seq_len = input_seq_len(model_input)
    if _seq_len is not None:
        position = validate_position(position, _seq_len, op="decompose")

    model_dtype = next(model._model.parameters()).dtype if any(
        True for _ in model._model.parameters()
    ) else torch.float32

    used_exact = False
    original_dtype: torch.dtype | None = None
    if exact and model_dtype != torch.float32:
        original_dtype = model_dtype
        model._model.to(dtype=torch.float32)
        used_exact = True

    try:
        embed_vec, schema_components, residual_vec = schema.decompose(
            model, model_input, position=position,
        )
    finally:
        if used_exact and original_dtype is not None:
            model._model.to(dtype=original_dtype)

    # Translate Component dataclass entries into the dict shape the
    # rest of the codebase (render_decompose, tests, audit2) expects.
    components: list[dict[str, Any]] = [{
        "name": "L-1.embed",
        "layer": -1,
        "type": "embed",
        "vector": embed_vec,
        "norm": embed_vec.norm().item(),
    }]
    for comp in schema_components:
        components.append(comp.to_dict())

    components.sort(
        key=lambda c: (
            c["layer"],
            0 if c["type"] == "embed"
            else 1 if c["type"] == "attn"
            else 2 if c["type"] == "mlp"
            else 3,  # block_delta
        ),
    )

    post_ln = arch.residual_topology == "post_ln"

    if post_ln:
        ln_note = (
            " Σ components = residual (telescoping block deltas; "
            "post-LN architectures collapse attn/mlp into a single "
            "per-block delta)."
        )
    elif arch.residual_topology == "seq2seq":
        ln_note = (
            " Σ components = residual rooted at the decoder stack "
            "(encoder hidden states enter via cross-attention; not on "
            "the residual path to lm_head)."
        )
    else:
        ln_note = (
            " Σ components = residual (embed + Σ_l attn_l + mlp_l) "
            "within fp32 epsilon."
        )

    if used_exact:
        precision_note = (
            f"forward re-run in float32 (model native dtype: {original_dtype}); "
            "decomposition is exact within fp32 epsilon." + ln_note
        )
    elif model_dtype == torch.float32:
        precision_note = (
            "fp32 forward — decomposition exact within fp32 epsilon." + ln_note
        )
    else:
        precision_note = (
            f"forward in {model_dtype} — per-component contributions cast to "
            f"fp32 but the residual stream itself accumulated in {model_dtype}. "
            "Expect ~10% relative drift at attention-sink positions (e.g. pos=0). "
            "Pass exact=True for an exact (slower, higher memory) reconstruction."
            + ln_note
        )

    result = {
        "components": components,
        "residual": residual_vec,
        "position": position,
        "precision_note": precision_note,
        "post_ln": post_ln,
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
    from interpkit.core.support_matrix import check_op_supported

    arch = model.arch_info
    check_op_supported("ov_scores", arch)
    if not arch.layer_names or not arch.num_attention_heads:
        raise ValueError("OV analysis requires detected layer structure and head count.")

    layer = _require_attention_layer(arch, layer, "ov_scores")
    num_heads = arch.num_attention_heads
    li = arch.layer_infos[layer]

    # Find V projection weight via centralised extractor
    w_v = extract_proj_weight(
        model._model, li, "v", num_heads, arch.num_key_value_heads,
    )

    # Find O projection weight
    if li.o_proj_path is None:
        raise ValueError(f"Could not find output projection weight in layer {layer}.")
    proj_mod = module_at_path(model._model, li.o_proj_path)

    if w_v is None:
        raise ValueError(f"Could not find V projection weight in layer {layer}.")
    if not hasattr(proj_mod, "weight"):
        raise ValueError(f"Could not find output projection weight in layer {layer}.")

    # Normalise W_O to (d_model, num_heads * head_dim)
    raw_w_o = get_weight(proj_mod).float()
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
    from interpkit.core.support_matrix import check_op_supported

    arch = model.arch_info
    check_op_supported("qk_scores", arch)
    if not arch.layer_names or not arch.num_attention_heads:
        raise ValueError("QK analysis requires detected layer structure and head count.")

    layer = _require_attention_layer(arch, layer, "qk_scores")
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

    src_layer = _require_attention_layer(arch, src_layer, "composition (src)")
    dst_layer = _require_attention_layer(arch, dst_layer, "composition (dst)")

    num_heads = arch.num_attention_heads
    num_kv_heads = arch.num_key_value_heads or num_heads

    # Source layer: build W_OV = W_O @ W_V for each head
    src_li = arch.layer_infos[src_layer]
    if src_li.o_proj_path is None:
        raise ValueError(f"No output projection in source layer {src_layer}.")

    src_proj = module_at_path(model._model, src_li.o_proj_path)
    if not hasattr(src_proj, "weight"):
        raise ValueError(f"No output projection in source layer {src_layer}.")

    raw_w_o_src = get_weight(src_proj).float()
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
    table.add_column("Head", style=ACCENT, justify="right")
    table.add_column("Frobenius Norm", justify="right")
    table.add_column("Approx Rank", justify="right")
    table.add_column("Top Singular Values", style=MUTED)

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
    table.add_column("Dst Head", style=ACCENT, justify="right")
    table.add_column("Src Head", style=ACCENT, justify="right")
    table.add_column("Score", justify="right")

    for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
        i = idx // num_heads
        j = idx % num_heads
        table.add_row(
            f"L{dst_layer}.H{i}", f"L{src_layer}.H{j}", f"{val:.4f}",
        )

    console.print(table)
    console.print()
