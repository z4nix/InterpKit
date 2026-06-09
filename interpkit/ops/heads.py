"""heads — decompose attention module output into per-head contributions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from interpkit.core.arch import get_weight
from interpkit.core.paths import validate_module_path
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_head_activations(
    model: Model,
    input_data: Any,
    *,
    at: str,
    output_proj: bool = True,
) -> dict[str, Any]:
    """Extract per-head activation contributions at an attention module.

    Hooks the input to the output projection (``c_proj`` / ``out_proj`` /
    ``o_proj``) to capture concatenated head outputs, then reshapes into
    per-head tensors and (optionally) projects each through the corresponding
    slice of W_o.

    **Sum invariant (N-007).** When ``output_proj=True``, the exact
    invariant that holds across every transformer family is::

        sum_h(head_acts[h]) + W_O.bias  ==  o_proj.forward(concat_heads)

    in fp32 to ~1e-5. The right-hand side is the OUTPUT of the o_proj
    Linear (the canonical "pre-residual, pre-LN attention output"
    anchor), captured at the module path ``pre_residual_anchor_path``
    in the returned dict. Comparing instead against the wrapper
    attention module's output (e.g. ``BertAttention``) does NOT satisfy
    this invariant because the wrapper post-LN's its output and adds
    the residual stream — both position-dependent transforms that
    cannot be undone per-head.

    Parameters
    ----------
    at:
        Name of a layer or attention module (e.g. ``"transformer.h.8"``
        or ``"transformer.h.8.attn"``).
    output_proj:
        If *True* (default), each head's output is projected through its
        slice of W_o so the result lives in residual-stream space
        (shape ``(batch, seq, d_model)`` per head).  If *False*, returns
        the raw pre-projection head outputs (shape ``(batch, seq, head_dim)``).

    N-007 (verified, not a limitation):
        The sum invariant holds on **every** family — including ALiBi
        (BLOOM), T5 relative-position bias, the post-LN encoders (BERT,
        RoBERTa, ELECTRA) and shared-weight ALBERT — to fp32 epsilon
        (≤ ~1e-6 rel; see ``audit2/N007_VERIFICATION.md``). It is purely
        a statement about the linearity of the output projection:
        positional bias acts on attention *scores* (upstream of the
        concat → o_proj step) and therefore cannot contaminate the
        anchor sum. The one requirement is comparing against
        ``output(pre_residual_anchor_path)`` (the o_proj output), NOT
        against the wrapper attention module's output (which post-LN's
        and adds the residual — see ``has_wrapper_attention``).

    Returns
    -------
    dict with:
        ``head_acts``: tensor of shape ``(num_heads, batch, seq, dim)``.
            By convention this is the LAST invocation of the attention
            module. For non-shared models that is the only invocation; for
            shared-weight architectures (ALBERT) it is the FINAL logical
            layer's invocation — matching the convention
            :func:`run_decompose` uses for its residual hook. Per-layer
            decompositions on shared models live in
            ``head_acts_per_invocation``.
        ``head_acts_per_invocation``: ``None`` for non-shared models;
            always a ``list[Tensor]`` for shared-weight models (A4 — present
            whenever ``arch.is_shared_layers`` is True, regardless of how
            many times the block fired), with one entry per logical-layer
            invocation of the shared physical attention. Each entry has the
            same shape as ``head_acts``. Users wanting per-layer head
            decompositions on ALBERT read this list rather than re-running
            ``run_head_activations``.
        ``num_heads``: int
        ``head_dim``: int
        ``module``: str — the attention module name used
        ``pre_residual_anchor_path``: str — the dotted path of the
            o_proj submodule whose forward OUTPUT equals
            ``Σ_h head_acts[h] + W_O.bias``. Use this (not ``module``)
            as the comparison anchor in any sum-invariant check.
        ``has_wrapper_attention``: bool — True iff the resolved
            ``at``-targeted wrapper module also contains a per-layer
            LayerNorm + residual add (BERT-family). Documented so that
            audit harnesses know not to compare against the wrapper.
    """
    from interpkit.core.support_matrix import check_op_supported

    arch = model.arch_info
    # N-004: gate DeBERTa-v3 before any forward hook fires.
    check_op_supported("head_activations", arch)
    num_heads = arch.num_attention_heads
    if num_heads is None or num_heads == 0:
        raise ValueError(
            "Cannot decompose heads: num_attention_heads not detected. "
            "Make sure the model has an HF config with num_attention_heads."
        )

    # F-022: reject typo'd module paths up-front with a friendly KeyError.
    validate_module_path(at, arch)

    # Try pre-resolved layer_infos first, fall back to ad-hoc search
    proj_mod = None
    proj_path: str | None = None
    attn_mod_name = at
    proj_child_name = ""
    matched_li: Any | None = None
    for li in arch.layer_infos:
        if li.name == at or li.attn_path == at:
            matched_li = li
            if li.o_proj_path:
                proj_mod = _get_module(model._model, li.o_proj_path)
                proj_path = li.o_proj_path
                attn_mod_name = li.attn_path or at
                proj_child_name = li.o_proj_path.split(".")[-1]
            break
    if proj_mod is None:
        attn_mod_name, proj_child_name, proj_mod = _find_output_proj(model._model, at)
        if proj_mod is not None:
            # Reconstruct dotted path from attn_mod_name + child name.
            proj_path = (
                f"{attn_mod_name}.{proj_child_name}" if proj_child_name else attn_mod_name
            )

    if proj_mod is None:
        raise RuntimeError(
            f"Could not find output projection (c_proj / out_proj / o_proj / dense / out_lin) "
            f"inside '{at}'. Head decomposition requires an identifiable output projection."
        )

    captured_input: list[torch.Tensor] = []

    def _capture_input_hook(_mod: torch.nn.Module, inp: Any, _output: Any) -> None:
        # Capture EVERY invocation (shared-layer architectures fire the
        # same physical o_proj N times per forward — ALBERT). Non-shared
        # models simply produce a single-element list.
        t = inp[0] if isinstance(inp, tuple) else inp
        if isinstance(t, torch.Tensor):
            captured_input.append(t.detach().clone())

    model_input = model._prepare(input_data)
    handle = proj_mod.register_forward_hook(_capture_input_hook)
    try:
        with torch.no_grad():
            model._forward(model_input)
    finally:
        handle.remove()

    if not captured_input:
        raise RuntimeError(
            f"Output projection '{proj_child_name}' produced no captured input."
        )

    # Shared-layer architectures (ALBERT) re-invoke the same physical
    # attention N times per forward — ``captured_input`` then has N
    # entries. Build a per-invocation list of head decompositions and
    # default ``head_acts`` to the LAST invocation (matches the
    # convention ``run_decompose`` uses for its residual hook).
    is_shared = bool(getattr(arch, "is_shared_layers", False))
    head_acts_per_invocation: list[torch.Tensor] | None = None

    # Validate the layout once on the trailing capture (canonical FINAL
    # invocation for shared models; only entry for non-shared).
    head_layout_check = captured_input[-1].float()
    if head_layout_check.dim() == 2:
        head_layout_check = head_layout_check.unsqueeze(0)
    if head_layout_check.shape[-1] % num_heads != 0:
        raise ValueError(
            f"Pre-projection activation dim ({head_layout_check.shape[-1]}) is not "
            f"divisible by num_attention_heads ({num_heads}). The module may use "
            f"grouped-query attention or a different head layout."
        )
    head_dim = head_layout_check.shape[-1] // num_heads

    def _decompose_one(captured: torch.Tensor) -> torch.Tensor:
        """Reshape one captured pre-projection tensor into per-head outputs."""
        c = captured.float()
        if c.dim() == 2:
            c = c.unsqueeze(0)
        b, s, _ = c.shape
        ph = c.view(b, s, num_heads, head_dim).permute(2, 0, 1, 3)  # (H, B, S, D_h)
        if output_proj and hasattr(proj_mod, "weight"):
            raw_w_o = get_weight(proj_mod).float()
            is_conv1d = type(proj_mod).__name__ == "Conv1D"
            w_o = raw_w_o.T if is_conv1d else raw_w_o
            d_model = int(w_o.shape[0])
            w_o_heads = w_o.view(d_model, num_heads, head_dim)
            projected = torch.zeros(num_heads, b, s, d_model, device=c.device)
            for h in range(num_heads):
                projected[h] = ph[h] @ w_o_heads[:, h, :].T
            return projected
        return ph

    per_head = _decompose_one(captured_input[-1])

    # A4: always populate the per-invocation list on shared-layer models so
    # callers can rely on its presence (not its truthiness). Previously it was
    # only set when >1 invocation was captured, so a shared-layer model whose
    # forward happened to fire the block once left the field None despite the
    # model being shared — an inconsistent contract.
    if is_shared:
        head_acts_per_invocation = [_decompose_one(t) for t in captured_input]

    # N-007: surface the canonical sum-invariant anchor and a flag
    # indicating whether ``at`` resolved to a wrapper module that
    # post-LN's its output (BERT-family). Audit harnesses should compare
    # ``Σ head_acts + W_O.bias`` against ``output(pre_residual_anchor_path)``,
    # never against ``output(at)`` for wrapper architectures.
    inner_path = matched_li.attn_inner_path if matched_li is not None else None
    if inner_path is None and proj_path is not None:
        inner_path = proj_path.rsplit(".", 1)[0]
    has_wrapper = inner_path is not None and inner_path != attn_mod_name

    return {
        "head_acts": per_head,
        "head_acts_per_invocation": head_acts_per_invocation,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "module": attn_mod_name,
        "pre_residual_anchor_path": proj_path,
        "attn_inner_path": inner_path,
        "has_wrapper_attention": has_wrapper,
    }


def _find_output_proj(
    model: torch.nn.Module, at: str
) -> tuple[str, str, torch.nn.Module | None]:
    """Locate the output projection submodule inside an attention block.

    Returns ``(attn_module_name, proj_child_name, proj_module)``.
    """
    from interpkit.core.arch import ATTN_NAMES, O_PROJ_NAMES

    target = _get_module(model, at)

    proj_patterns = O_PROJ_NAMES

    # Direct children of the target module
    for child_name, child_mod in target.named_children():
        if child_name in proj_patterns and hasattr(child_mod, "weight"):
            return at, child_name, child_mod

    # If 'at' points to a layer (not attention), look deeper via named_modules.
    # Match attention submodules by canonical attribute name set, no regex.
    _attn_names = ATTN_NAMES
    for child_name, child_mod in target.named_modules():
        if not child_name:
            continue
        last_segment = child_name.rsplit(".", 1)[-1].lower()
        if last_segment in _attn_names:
            attn_full = f"{at}.{child_name}" if child_name else at
            for sub_name, sub_mod in child_mod.named_modules():
                if not sub_name:
                    continue
                base = sub_name.split(".")[-1]
                if base in proj_patterns and hasattr(sub_mod, "weight"):
                    return attn_full, sub_name, sub_mod

    # Last resort: search all descendants (handles deep nesting like
    # attention.output.dense in BERT/ViT models)
    for child_name, child_mod in target.named_modules():
        if not child_name:
            continue
        base = child_name.split(".")[-1]
        if base in proj_patterns and hasattr(child_mod, "weight"):
            return at, child_name, child_mod

    return at, "", None
