"""dla — Direct Logit Attribution: decompose output logits by component contribution."""

from __future__ import annotations

import re
from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from interpkit.core.discovery import _get_weight
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
    sae: Any | None = None,
    sae_at: str | None = None,
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
    sae:
        A pre-loaded :class:`interpkit.ops.sae.SAE` object.  When provided
        together with *sae_at*, the contribution of the specified component
        is further decomposed into per-feature logit attributions.
    sae_at:
        Module path of the component to decompose through the SAE (e.g.
        ``"transformer.h.11.attn"``).  Must match a module captured by
        DLA.  Required when *sae* is provided.

    Returns
    -------
    dict with:
        ``target_token`` (str), ``target_id`` (int),
        ``contributions`` (list of ``{"component", "layer", "type", "logit_contribution"}``),
        ``total_logit`` (float),
        ``approximation_note`` (str) — explains the LayerNorm caveat.
        Optionally ``feature_contributions`` when *sae* is provided.
    """
    from interpkit.core.render import render_dla

    # Validate sae / sae_at pairing
    if (sae is None) != (sae_at is None):
        raise ValueError(
            "Both 'sae' and 'sae_at' must be provided together. "
            "Pass sae=<SAE object> and sae_at=<module path> to decompose "
            "a component through the SAE."
        )

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
    unembed_weight = _get_weight(unembed_mod).float()  # (vocab, embed_dim)

    # Handle models where embed_dim != hidden_size (e.g. OPT-350m)
    project_out_weight = None
    if arch.project_out_path:
        try:
            po_mod = _get_module(model._model, arch.project_out_path)
            project_out_weight = _get_weight(po_mod).float()  # (embed_dim, hidden_size)
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
        target_id = int(last_logits.argmax().item())
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
    hooks: list[torch.utils.hooks.RemovableHandle] = []

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

    for hook in hooks:
        hook.remove()

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
        capture_hooks: list[torch.utils.hooks.RemovableHandle] = []

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

        raw_w_o = _get_weight(proj_mod).float()
        is_conv1d = type(proj_mod).__name__ == "Conv1D"
        w_o = raw_w_o.T if is_conv1d else raw_w_o
        d_model = int(w_o.shape[0])
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

    if sae is not None and sae_at is not None:
        result["feature_contributions"] = _compute_dla_features(
            sae, sae_at, component_outputs, unembed_dir, position, top_k,
        )

    render_dla(result, top_k=top_k)

    if save is not None:
        from interpkit.core.plot import plot_dla

        plot_dla(result, top_k=top_k, save_path=save)

    if html is not None:
        from interpkit.core.html import html_dla as gen_html_dla
        from interpkit.core.html import save_html

        save_html(gen_html_dla(result), html)

    return result


_ATTN_SUFFIXES = (".attn", ".self_attn", ".attention")
_MLP_SUFFIXES = (".mlp", ".ffn", ".feed_forward")


def _compute_dla_features(
    sae: Any,
    sae_at: str,
    component_outputs: dict[str, torch.Tensor],
    unembed_dir: torch.Tensor,
    position: int,
    top_k: int,
) -> dict[str, Any]:
    """Decompose a DLA component's contribution through an SAE into per-feature logit attributions."""
    act_tensor: torch.Tensor | None = None
    matched_key: str | None = None

    # DLA stores keys as "{layer_name}::attn" / "{layer_name}::mlp" where
    # layer_name is e.g. "transformer.h.0".  The user typically passes the
    # full submodule path (e.g. "transformer.h.0.attn") or a layer path.
    # Strategy: detect the component type from the suffix, strip it, and
    # construct the canonical key.
    comp_type: str | None = None
    layer_name = sae_at
    for sfx in _ATTN_SUFFIXES:
        if sae_at.endswith(sfx):
            comp_type = "attn"
            layer_name = sae_at[: -len(sfx)]
            break
    if comp_type is None:
        for sfx in _MLP_SUFFIXES:
            if sae_at.endswith(sfx):
                comp_type = "mlp"
                layer_name = sae_at[: -len(sfx)]
                break

    if comp_type is not None:
        key = f"{layer_name}::{comp_type}"
        if key in component_outputs:
            act_tensor = component_outputs[key]
            matched_key = key

    # Fall back: try sae_at as a layer name (match ::attn first, then ::mlp)
    if act_tensor is None:
        for suffix in ("::attn", "::mlp"):
            key = sae_at + suffix
            if key in component_outputs:
                act_tensor = component_outputs[key]
                matched_key = key
                break

    if act_tensor is None:
        valid_modules = sorted({k.split("::")[0] for k in component_outputs})
        valid_with_types = []
        for m in valid_modules:
            for suffix in ("::attn", "::mlp"):
                if m + suffix in component_outputs:
                    valid_with_types.append(m + suffix.replace("::", "."))
        raise ValueError(
            f"sae_at={sae_at!r} did not match any component captured by DLA. "
            f"Valid module paths: {valid_with_types}"
        )

    # Extract the activation vector at the target position
    if act_tensor.dim() == 3:
        vec = act_tensor[0, position, :].float()
    elif act_tensor.dim() == 2:
        vec = act_tensor[position, :].float()
    else:
        vec = act_tensor.float()

    if vec.shape[-1] != sae.d_in:
        raise ValueError(
            f"SAE input dimension ({sae.d_in}) does not match the activation "
            f"dimension ({vec.shape[-1]}) at {sae_at!r}. Make sure the SAE was "
            f"trained on the same layer/component."
        )

    # Encode through the SAE
    features = sae.encode(vec.unsqueeze(0)).squeeze(0)  # (d_sae,)

    active_mask = features > 0
    if not active_mask.any():
        return {
            "sae_at": sae_at,
            "matched_component": matched_key,
            "features": [],
            "num_active": 0,
            "total_features": sae.d_sae,
        }

    active_idxs = active_mask.nonzero(as_tuple=True)[0]
    active_acts = features[active_idxs]

    # Per-feature logit contribution: feat_act * (W_dec[feat_idx] @ unembed_dir)
    dec_rows = sae.W_dec[active_idxs].float()  # (n_active, d_model)
    logit_dirs = dec_rows @ unembed_dir  # (n_active,)
    logit_contribs = active_acts * logit_dirs

    # Sort by absolute contribution and take top_k
    abs_contribs = logit_contribs.abs()
    k = min(top_k, len(active_idxs))
    top_vals, top_local_idxs = abs_contribs.topk(k)

    feat_list = []
    for local_idx in top_local_idxs.tolist():
        feat_idx = active_idxs[local_idx].item()
        feat_list.append({
            "feature_idx": feat_idx,
            "activation": active_acts[local_idx].item(),
            "logit_contribution": logit_contribs[local_idx].item(),
        })

    return {
        "sae_at": sae_at,
        "matched_component": matched_key,
        "features": feat_list,
        "num_active": int(active_mask.sum().item()),
        "total_features": sae.d_sae,
    }


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
