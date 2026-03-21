"""heads — decompose attention module output into per-head contributions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_head_activations(
    model: "Model",
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

    Returns
    -------
    dict with:
        ``head_acts``: tensor of shape ``(num_heads, batch, seq, dim)``
        ``num_heads``: int
        ``head_dim``: int
        ``module``: str — the attention module name used
    """
    arch = model.arch_info
    num_heads = arch.num_attention_heads
    if num_heads is None or num_heads == 0:
        raise ValueError(
            "Cannot decompose heads: num_attention_heads not detected. "
            "Make sure the model has an HF config with num_attention_heads."
        )

    # Try pre-resolved layer_infos first, fall back to ad-hoc search
    proj_mod = None
    attn_mod_name = at
    proj_child_name = ""
    for li in arch.layer_infos:
        if li.name == at or li.attn_path == at:
            if li.o_proj_path:
                proj_mod = _get_module(model._model, li.o_proj_path)
                attn_mod_name = li.attn_path or at
                proj_child_name = li.o_proj_path.split(".")[-1]
            break
    if proj_mod is None:
        attn_mod_name, proj_child_name, proj_mod = _find_output_proj(model._model, at)

    if proj_mod is None:
        raise RuntimeError(
            f"Could not find output projection (c_proj / out_proj / o_proj / dense / out_lin) "
            f"inside '{at}'. Head decomposition requires an identifiable output projection."
        )

    captured_input: list[torch.Tensor] = []

    def _capture_input_hook(_mod: torch.nn.Module, inp: Any, _output: Any) -> None:
        t = inp[0] if isinstance(inp, tuple) else inp
        if isinstance(t, torch.Tensor):
            captured_input.append(t.detach().clone())

    model_input = model._prepare(input_data)
    handle = proj_mod.register_forward_hook(_capture_input_hook)
    with torch.no_grad():
        model._forward(model_input)
    handle.remove()

    if not captured_input:
        raise RuntimeError(
            f"Output projection '{proj_child_name}' produced no captured input."
        )

    concat_heads = captured_input[0].float()  # (batch, seq, num_heads * head_dim)

    if concat_heads.dim() == 2:
        concat_heads = concat_heads.unsqueeze(0)

    head_dim = concat_heads.shape[-1] // num_heads
    batch, seq, _ = concat_heads.shape

    per_head = concat_heads.view(batch, seq, num_heads, head_dim)  # (B, S, H, D_h)
    per_head = per_head.permute(2, 0, 1, 3)  # (H, B, S, D_h)

    if output_proj and hasattr(proj_mod, "weight"):
        raw_w_o = proj_mod.weight.float()
        is_conv1d = type(proj_mod).__name__ == "Conv1D"
        w_o = raw_w_o.T if is_conv1d else raw_w_o  # -> (d_model, H*D_h)
        d_model = w_o.shape[0]
        w_o_heads = w_o.view(d_model, num_heads, head_dim)  # (d_model, H, D_h)

        projected = torch.zeros(num_heads, batch, seq, d_model, device=concat_heads.device)
        for h in range(num_heads):
            projected[h] = per_head[h] @ w_o_heads[:, h, :].T  # (B, S, D_h) @ (D_h, d_model)

        per_head = projected

    return {
        "head_acts": per_head,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "module": attn_mod_name,
    }


def _find_output_proj(
    model: torch.nn.Module, at: str
) -> tuple[str, str, torch.nn.Module | None]:
    """Locate the output projection submodule inside an attention block.

    Returns ``(attn_module_name, proj_child_name, proj_module)``.
    """
    target = _get_module(model, at)

    proj_patterns = ("c_proj", "out_proj", "o_proj", "dense", "out_lin", "o")

    # Direct children of the target module
    for child_name, child_mod in target.named_children():
        if child_name in proj_patterns and hasattr(child_mod, "weight"):
            return at, child_name, child_mod

    # If 'at' points to a layer (not attention), look deeper via named_modules
    for child_name, child_mod in target.named_modules():
        if not child_name:
            continue
        is_attn = re.search(r"(self_attn|attn|attention)", child_name, re.IGNORECASE)
        if is_attn:
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
