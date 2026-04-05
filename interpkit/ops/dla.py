"""dla — Direct Logit Attribution: decompose output logits by component contribution."""

from __future__ import annotations

import re
from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_dla(
    model: Model,
    input_data: Any,
    *,
    token: int | str | None = None,
    position: int = -1,
    top_k: int = 10,
    save: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Direct Logit Attribution: decompose output logits by component.

    For each layer, measures how much the attention heads and MLP contribute
    to the logit of a target token by projecting their outputs through the
    unembedding matrix.

    **Note on the LayerNorm approximation:** The true model logit is
    ``W_U @ LayerNorm(residual)``.  Because LayerNorm is nonlinear, the
    contribution of each component cannot be decomposed exactly through
    it.  This implementation projects each component's raw output directly
    through W_U, bypassing LayerNorm.  As a result, ``total_logit`` (the
    sum of all component contributions) will *not* exactly equal the
    model's actual logit for the target token.  Component *rankings* are
    still meaningful — this is the same approximation used by
    TransformerLens and other standard DLA implementations.

    Parameters
    ----------
    token:
        Target token to attribute.  If *None*, uses the model's top-1
        prediction.  Can be an int (token id) or a string (decoded to id).
    position:
        Token position to analyse (default ``-1`` = last).
    top_k:
        Number of top/bottom contributors to highlight in the rendering.
    save:
        Path to save a bar-chart figure.

    Returns
    -------
    dict with:
        ``target_token`` (str), ``target_id`` (int),
        ``contributions`` (list of ``{"component", "layer", "type", "logit_contribution"}``),
        ``total_logit`` (float),
        ``approximation_note`` (str) — explains the LayerNorm caveat.
    """
    from interpkit.core.render import render_dla

    arch = model.arch_info

    if not arch.is_language_model or arch.unembedding_name is None:
        raise ValueError(
            "DLA requires a language model with a detectable unembedding matrix."
        )
    if not arch.layer_names:
        raise ValueError("DLA requires detected layer structure.")
    if model._tokenizer is None:
        raise ValueError("DLA requires a tokenizer.")

    num_heads = arch.num_attention_heads
    if num_heads is None or num_heads == 0:
        raise ValueError("DLA requires num_attention_heads in the model config.")

    model_input = model._prepare(input_data)

    # Get unembedding direction for the target token
    unembed_mod = _get_module(model._model, arch.unembedding_name)
    unembed_weight = unembed_mod.weight.float()  # (vocab, embed_dim)

    # Handle models where embed_dim != hidden_size (e.g. OPT-350m)
    project_out_weight = None
    if arch.project_out_path:
        try:
            po_mod = _get_module(model._model, arch.project_out_path)
            project_out_weight = po_mod.weight.float()  # (embed_dim, hidden_size)
        except AttributeError:
            import warnings
            warnings.warn(
                f"project_out_path '{arch.project_out_path}' is set but its weight "
                f"could not be loaded. DLA results may have incorrect dimensionality.",
                stacklevel=2,
            )

    # Determine target token id
    if token is None:
        with torch.no_grad():
            logits = model._forward(model_input)
        if logits.dim() == 3:
            last_logits = logits[0, position, :]
        else:
            last_logits = logits[0]
        target_id = last_logits.argmax().item()
    elif isinstance(token, str):
        ids = model._tokenizer.encode(token, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Could not encode token: {token!r}")
        if len(ids) > 1:
            import warnings
            decoded_first = model._tokenizer.decode([ids[0]])
            warnings.warn(
                f"Token {token!r} encodes to {len(ids)} subwords; "
                f"using only the first subword ({decoded_first!r}, id={ids[0]}).",
                stacklevel=2,
            )
        target_id = ids[0]
    else:
        target_id = token

    target_token_str = model._tokenizer.decode([target_id])

    # Compute effective unembedding direction in residual-stream space.
    # For standard models: unembed_dir = W_U[target_id] with shape (d_model,).
    # For OPT-style models: hidden -> project_out -> lm_head, so the
    # effective direction is W_project_out^T @ W_U[target_id].
    raw_unembed_dir = unembed_weight[target_id]  # (embed_dim,)
    if project_out_weight is not None:
        unembed_dir = project_out_weight.T @ raw_unembed_dir  # (hidden_size,)
    else:
        unembed_dir = raw_unembed_dir  # (d_model,)

    # Capture outputs of each attention output-projection and each MLP
    component_outputs: dict[str, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHook] = []

    for li in arch.layer_infos:
        comp_key_attn = f"{li.name}::attn"
        if li.attn_path:
            attn_mod = _get_module(model._model, li.attn_path)

            def _make_attn_hook(key: str):
                def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
                    t = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                    )
                    if t is not None:
                        component_outputs[key] = t.detach().float()
                return hook_fn

            hooks.append(attn_mod.register_forward_hook(_make_attn_hook(comp_key_attn)))

        if li.mlp_path:
            mlp_mod = _get_module(model._model, li.mlp_path)
            comp_key_mlp = f"{li.name}::mlp"

            def _make_mlp_hook(key: str):
                def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
                    t = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                    )
                    if t is not None:
                        component_outputs[key] = t.detach().float()
                return hook_fn

            hooks.append(mlp_mod.register_forward_hook(_make_mlp_hook(comp_key_mlp)))

    model._forward(model_input)

    for h in hooks:
        h.remove()

    # Compute each component's contribution to the target logit
    contributions: list[dict[str, Any]] = []

    for comp_key, output_tensor in component_outputs.items():
        layer_name, comp_type = comp_key.rsplit("::", 1)

        if output_tensor.dim() == 3:
            vec = output_tensor[0, position, :]  # (d_model,)
        elif output_tensor.dim() == 2:
            vec = output_tensor[position, :]
        else:
            vec = output_tensor

        logit_contrib = (vec @ unembed_dir).item()

        layer_match = re.search(r"\.(\d+)", layer_name)
        layer_idx = int(layer_match.group(1)) if layer_match else 0

        contributions.append({
            "component": f"L{layer_idx}.{'attn' if comp_type == 'attn' else 'mlp'}",
            "layer": layer_idx,
            "type": comp_type,
            "logit_contribution": logit_contrib,
            "module": comp_key.split("::")[0],
        })

    # Per-head breakdown: capture all pre-projection inputs in a single forward pass
    head_contributions: list[dict[str, Any]] = []

    proj_info: list[tuple[str, torch.nn.Module]] = []
    pre_proj_captures: dict[str, list[torch.Tensor]] = {}

    for li in arch.layer_infos:
        if not li.o_proj_path or not li.attn_path:
            continue
        try:
            proj_mod = _get_module(model._model, li.o_proj_path)
        except AttributeError:
            continue
        if not hasattr(proj_mod, "weight"):
            continue
        comp_key = f"{li.name}::attn"
        if comp_key not in component_outputs:
            continue
        proj_info.append((li.name, proj_mod))
        pre_proj_captures[li.name] = []

    if proj_info:
        capture_hooks: list[torch.utils.hooks.RemovableHook] = []

        def _make_capture_hook(store: list[torch.Tensor]):
            def hook_fn(_mod: torch.nn.Module, inp: Any, _output: Any) -> None:
                t = inp[0] if isinstance(inp, tuple) else inp
                if isinstance(t, torch.Tensor):
                    store.append(t.detach().float())
            return hook_fn

        for layer_name, proj_mod in proj_info:
            capture_hooks.append(
                proj_mod.register_forward_hook(_make_capture_hook(pre_proj_captures[layer_name]))
            )

        model._forward(model_input)

        for ch in capture_hooks:
            ch.remove()

    for layer_name, proj_mod in proj_info:
        captured = pre_proj_captures[layer_name]
        if not captured:
            continue

        concat_heads = captured[0]
        if concat_heads.dim() == 2:
            concat_heads = concat_heads.unsqueeze(0)

        head_dim = concat_heads.shape[-1] // num_heads
        per_head = concat_heads[0, position, :].view(num_heads, head_dim)

        raw_w_o = proj_mod.weight.float()
        is_conv1d = type(proj_mod).__name__ == "Conv1D"
        w_o = raw_w_o.T if is_conv1d else raw_w_o
        d_model = w_o.shape[0]
        w_o_heads = w_o.view(d_model, num_heads, head_dim)

        layer_match = re.search(r"\.(\d+)", layer_name)
        layer_idx = int(layer_match.group(1)) if layer_match else 0

        for h in range(num_heads):
            head_resid = per_head[h] @ w_o_heads[:, h, :].T
            logit_contrib = (head_resid @ unembed_dir).item()
            head_contributions.append({
                "component": f"L{layer_idx}.H{h}",
                "layer": layer_idx,
                "head": h,
                "type": "head",
                "logit_contribution": logit_contrib,
            })

    contributions.sort(key=lambda c: c["logit_contribution"], reverse=True)
    head_contributions.sort(key=lambda c: c["logit_contribution"], reverse=True)

    total_logit = sum(c["logit_contribution"] for c in contributions)

    result = {
        "target_token": target_token_str,
        "target_id": target_id,
        "contributions": contributions,
        "head_contributions": head_contributions,
        "total_logit": total_logit,
        "approximation_note": (
            "total_logit is the sum of per-component contributions projected "
            "through the unembedding matrix, bypassing the final LayerNorm. "
            "Because LayerNorm is nonlinear, this sum will not exactly match "
            "the model's true logit. Component rankings remain valid."
        ),
    }

    render_dla(result, top_k=top_k)

    if save is not None:
        from interpkit.core.plot import plot_dla

        plot_dla(result, top_k=top_k, save_path=save)

    if html is not None:
        from interpkit.core.html import html_dla as gen_html_dla
        from interpkit.core.html import save_html

        save_html(gen_html_dla(result), html)

    return result


def _find_attn_submodule(
    layer_mod: torch.nn.Module,
) -> tuple[str, torch.nn.Module] | None:
    """Find the attention submodule inside a layer (recursive BFS)."""
    queue: deque[tuple[str, torch.nn.Module]] = deque()
    for name, mod in layer_mod.named_children():
        queue.append((name, mod))
    while queue:
        rel_name, mod = queue.popleft()
        base = rel_name.split(".")[-1]
        if re.search(r"(self_attn|attn|attention)", base, re.IGNORECASE):
            return rel_name, mod
        for child_name, child_mod in mod.named_children():
            queue.append((f"{rel_name}.{child_name}", child_mod))
    return None


def _find_mlp_submodule(
    layer_mod: torch.nn.Module,
) -> tuple[str, torch.nn.Module] | None:
    """Find the MLP submodule inside a layer (recursive BFS)."""
    queue: deque[tuple[str, torch.nn.Module]] = deque()
    for name, mod in layer_mod.named_children():
        queue.append((name, mod))
    while queue:
        rel_name, mod = queue.popleft()
        base = rel_name.split(".")[-1]
        if re.search(r"(mlp|ffn|feed_forward)", base, re.IGNORECASE):
            return rel_name, mod
        for child_name, child_mod in mod.named_children():
            queue.append((f"{rel_name}.{child_name}", child_mod))
    return None
