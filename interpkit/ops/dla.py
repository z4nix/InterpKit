"""dla — Direct Logit Attribution: decompose output logits by component contribution."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from interpkit.core.arch import ATTN_NAMES as _ATTN_NAMES
from interpkit.core.arch import MLP_NAMES as _MLP_NAMES
from interpkit.core.arch import get_weight
from interpkit.core.inputs import input_seq_len
from interpkit.core.paths import validate_position
from interpkit.ops._hooks import first_tensor
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
    through W_U, *bypassing the final LayerNorm entirely* (it does not
    apply the per-position LN scale/centering, nor does it include the
    token/positional embedding contributions to the residual).  As a
    result, the sum of contributions (``total_logit_pre_ln``) is on a
    different scale from the model's actual logit, and ``ln_error`` can be
    comparable to or larger than the logit magnitude — this is expected.
    Component *rankings* remain meaningful (the omitted terms are shared
    across components at a position), which is the contract this op
    provides.  Note this differs from TransformerLens, which applies the
    cached final-LN scale (and centering) so its residual-decomposition
    sum reconstructs the logit to near-zero error; use ``ln_error`` here
    to see the gap rather than assuming reconstruction.

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

    _seq_len = input_seq_len(model_input)
    if _seq_len is not None:
        position = validate_position(position, _seq_len, op="dla")

    # Get unembedding direction for the target token
    unembed_mod = _get_module(model._model, arch.unembedding_name)
    unembed_weight = get_weight(unembed_mod).float()  # (vocab, embed_dim)

    # Handle models where embed_dim != hidden_size (e.g. OPT-350m)
    project_out_weight = None
    if arch.project_out_path:
        try:
            po_mod = _get_module(model._model, arch.project_out_path)
            project_out_weight = get_weight(po_mod).float()  # (embed_dim, hidden_size)
        except AttributeError:
            import warnings
            warnings.warn(
                f"project_out_path '{arch.project_out_path}' is set but its weight "
                f"could not be loaded. DLA results may have incorrect dimensionality.",
                stacklevel=2,
            )

    # Single instrumented forward — captures each component's output, each
    # attention layer's pre-projection per-head input, AND the model logits
    # in ONE pass. The pre-refactor DLA ran three or four separate forwards
    # for these; everything below reuses this one.
    shared_layers = bool(getattr(arch, "is_shared_layers", False))
    component_outputs, pre_proj_captures, proj_info, logits = _capture_dla_activations(
        model, arch, model_input, shared_layers,
    )

    # Determine target token id (model top-1 reuses the forward above).
    target_id = _resolve_target_id(model, token, position, logits)
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

    contributions = _component_contributions(
        component_outputs, unembed_dir, position, arch,
    )
    head_contributions = _head_contributions(
        proj_info, pre_proj_captures, unembed_dir, position, num_heads,
        shared_layers, arch,
    )

    contributions.sort(key=lambda c: c["logit_contribution"], reverse=True)
    head_contributions.sort(key=lambda c: c["logit_contribution"], reverse=True)

    # F-006: split the misleading single ``total_logit`` field into three
    # explicit fields. The pre-1.0 API exposed only the sum-of-contributions
    # value but called it ``total_logit``, leading users to treat it as
    # the actual model logit. Per the audit, this routinely deviated by
    # 3.5–12.1 nats from the true logit (>20% relative error).
    #
    # The sum of per-component contributions bypasses the final LayerNorm —
    # because LayerNorm is non-linear, this sum cannot equal the actual logit.
    # We now report all three values explicitly so users can see both the
    # decomposition rankings (still valid) and the LN-induced gap.
    total_logit_pre_ln = float(sum(c["logit_contribution"] for c in contributions))

    # Actual model logit at the target — read from the single capture forward.
    if logits.dim() == 3:
        model_logit = float(logits[0, position, target_id].item())
    elif logits.dim() == 2:
        model_logit = float(logits[0, target_id].item())
    else:
        model_logit = float(logits[target_id].item())
    ln_error = float(model_logit - total_logit_pre_ln)

    result = {
        "target_token": target_token_str,
        "target_id": target_id,
        "contributions": contributions,
        "head_contributions": head_contributions,
        # F-006: three explicit fields, never the misleading single total_logit
        "total_logit_pre_ln": total_logit_pre_ln,
        "model_logit": model_logit,
        "ln_error": ln_error,
        "approximation_note": (
            "total_logit_pre_ln is the sum of per-component contributions "
            "projected through the unembedding matrix, BYPASSING the final "
            "LayerNorm. model_logit is the actual model output at the target "
            "token. ln_error = model_logit - total_logit_pre_ln captures the "
            "LayerNorm non-linearity gap. Component rankings remain valid; "
            "the sum is an approximation, not the model's prediction."
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


# ---------------------------------------------------------------------------
# DLA phase helpers (run_dla orchestrates these)
# ---------------------------------------------------------------------------


def _resolve_target_id(
    model: Model, token: int | str | None, position: int, logits: torch.Tensor,
) -> int:
    """Resolve the target token id.

    ``None`` → the model's top-1 at *position* (reusing *logits*, no extra
    forward); an ``int`` → used directly; a ``str`` → decoded to its first
    sub-token, warning (with a leading-space tip) when it spans several.
    """
    if token is None:
        last_logits = logits[0, position, :] if logits.dim() == 3 else logits[0]
        return int(last_logits.argmax().item())
    if isinstance(token, str):
        ids = model._tokenizer.encode(token, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Could not encode token: {token!r}")
        if len(ids) > 1:
            import warnings
            decoded_first = model._tokenizer.decode([ids[0]])

            tip = ""
            if not token.startswith((" ", "\t", "\n")):
                try:
                    spaced_ids = model._tokenizer.encode(" " + token, add_special_tokens=False)
                except (TypeError, ValueError, RuntimeError):
                    spaced_ids = []
                if len(spaced_ids) == 1:
                    tip = (
                        f" Tip: pass token={(' ' + token)!r} (with a leading "
                        f"space) — that is a single token in this vocabulary "
                        f"(id={spaced_ids[0]}) and matches what the model "
                        f"actually predicts mid-sentence."
                    )

            warnings.warn(
                f"Token {token!r} encodes to {len(ids)} subwords; "
                f"using only the first subword ({decoded_first!r}, id={ids[0]})." + tip,
                stacklevel=3,
            )
        return ids[0]
    return token


def _make_component_hook(
    store: dict[str, torch.Tensor],
    layer_name: str,
    comp_type: str,
    counter: list[int] | None,
):
    """Forward hook capturing a component's output into *store* (fp32).

    Non-shared models key by ``"{layer}::{type}"``. Shared-weight models
    (ALBERT, *counter* given) write a distinct ``"{layer}#{i}::{type}"`` key
    per invocation so per-layer contributions don't collapse into one entry.
    """
    def hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
        t = first_tensor(output)
        if t is None:
            return
        if counter is None:
            store[f"{layer_name}::{comp_type}"] = t.detach().float()
        else:
            store[f"{layer_name}#{counter[0]}::{comp_type}"] = t.detach().float()
            counter[0] += 1

    return hook


def _make_preproj_hook(store: list[torch.Tensor]):
    """Forward hook capturing a projection module's INPUT (the per-head,
    pre-output-projection activations) — appended per invocation."""
    def hook(_mod: torch.nn.Module, inp: Any, _output: Any) -> None:
        t = inp[0] if isinstance(inp, tuple) else inp
        if isinstance(t, torch.Tensor):
            store.append(t.detach().float())

    return hook


def _capture_dla_activations(
    model: Model,
    arch: Any,
    model_input: Any,
    shared_layers: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]], list[tuple[str, torch.nn.Module]], torch.Tensor]:
    """Single forward capturing everything DLA needs in one pass.

    Returns ``(component_outputs, pre_proj_captures, proj_info, logits)``:
    each component's output (attn / mlp), each attention layer's
    pre-projection per-head input, and the model logits. This replaces the
    three separate forward passes the pre-refactor DLA ran (component
    capture, per-head capture, and the final logit read). Shared-weight
    models register one hook per physical module and rely on per-call
    ordering, which a single forward makes self-consistent.
    """
    component_outputs: dict[str, torch.Tensor] = {}
    pre_proj_captures: dict[str, list[torch.Tensor]] = {}
    proj_info: list[tuple[str, torch.nn.Module]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    seen_attn: set[int] = set()
    seen_mlp: set[int] = set()
    seen_proj: set[int] = set()

    for li in arch.layer_infos:
        if li.attn_path:
            mod = _get_module(model._model, li.attn_path)
            if not (shared_layers and id(mod) in seen_attn):
                seen_attn.add(id(mod))
                handles.append(mod.register_forward_hook(_make_component_hook(
                    component_outputs, li.name, "attn", [0] if shared_layers else None)))
        if li.mlp_path:
            mod = _get_module(model._model, li.mlp_path)
            if not (shared_layers and id(mod) in seen_mlp):
                seen_mlp.add(id(mod))
                handles.append(mod.register_forward_hook(_make_component_hook(
                    component_outputs, li.name, "mlp", [0] if shared_layers else None)))
        if li.o_proj_path and li.attn_path:
            try:
                proj_mod = _get_module(model._model, li.o_proj_path)
            except AttributeError:
                proj_mod = None
            if (proj_mod is not None and hasattr(proj_mod, "weight")
                    and not (shared_layers and id(proj_mod) in seen_proj)):
                seen_proj.add(id(proj_mod))
                store: list[torch.Tensor] = []
                pre_proj_captures[li.name] = store
                proj_info.append((li.name, proj_mod))
                handles.append(proj_mod.register_forward_hook(_make_preproj_hook(store)))

    try:
        logits = model._forward(model_input)
    finally:
        for h in handles:
            h.remove()
    return component_outputs, pre_proj_captures, proj_info, logits


def _component_contributions(
    component_outputs: dict[str, torch.Tensor],
    unembed_dir: torch.Tensor,
    position: int,
    arch: Any,
) -> list[dict[str, Any]]:
    """Project each captured component output through the unembedding direction."""
    out: list[dict[str, Any]] = []
    for comp_key, output_tensor in component_outputs.items():
        # Shared-layer keys carry a "#N" call-index suffix; strip it to
        # recover the module path and use the suffix as the logical layer.
        layer_part, comp_type = comp_key.rsplit("::", 1)
        if "#" in layer_part:
            layer_name, idx_str = layer_part.rsplit("#", 1)
            try:
                layer_idx = int(idx_str)
            except ValueError:
                layer_idx = 0
        else:
            layer_name = layer_part
            layer_idx_opt = arch.layer_of(layer_name) if hasattr(arch, "layer_of") else None
            layer_idx = layer_idx_opt if layer_idx_opt is not None else 0

        if output_tensor.dim() == 3:
            vec = output_tensor[0, position, :]
        elif output_tensor.dim() == 2:
            vec = output_tensor[position, :]
        else:
            vec = output_tensor

        out.append({
            "component": f"L{layer_idx}.{'attn' if comp_type == 'attn' else 'mlp'}",
            "layer": layer_idx,
            "type": comp_type,
            "logit_contribution": (vec @ unembed_dir).item(),
            "module": layer_name,
        })
    return out


def _head_contributions(
    proj_info: list[tuple[str, torch.nn.Module]],
    pre_proj_captures: dict[str, list[torch.Tensor]],
    unembed_dir: torch.Tensor,
    position: int,
    num_heads: int,
    shared_layers: bool,
    arch: Any,
) -> list[dict[str, Any]]:
    """Per-head logit attribution: split each captured pre-projection input
    into heads and project ``head_act @ W_O[head]`` through the unembedding."""
    out: list[dict[str, Any]] = []
    for layer_name, proj_mod in proj_info:
        captured = pre_proj_captures.get(layer_name, [])
        if not captured:
            continue

        # Shared-weight o_proj fires N times → each capture is one logical
        # layer. Non-shared layers capture exactly once.
        captures_to_process = captured if shared_layers else [captured[0]]

        raw_w_o = get_weight(proj_mod).float()
        is_conv1d = type(proj_mod).__name__ == "Conv1D"
        w_o = raw_w_o.T if is_conv1d else raw_w_o
        d_model = int(w_o.shape[0])

        for invocation_idx, concat_heads in enumerate(captures_to_process):
            if concat_heads.dim() == 2:
                concat_heads = concat_heads.unsqueeze(0)
            head_dim = concat_heads.shape[-1] // num_heads
            per_head = concat_heads[0, position, :].view(num_heads, head_dim)
            w_o_heads = w_o.view(d_model, num_heads, head_dim)

            if shared_layers:
                layer_idx = invocation_idx
            else:
                layer_idx_opt = arch.layer_of(layer_name) if hasattr(arch, "layer_of") else None
                layer_idx = layer_idx_opt if layer_idx_opt is not None else 0

            for h in range(num_heads):
                head_resid = per_head[h] @ w_o_heads[:, h, :].T
                out.append({
                    "component": f"L{layer_idx}.H{h}",
                    "layer": layer_idx,
                    "head": h,
                    "type": "head",
                    "logit_contribution": (head_resid @ unembed_dir).item(),
                })
    return out


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

    from interpkit.ops.sae import _ensure_sae_on_device

    sae = _ensure_sae_on_device(sae, vec.device)

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


# _ATTN_NAMES / _MLP_NAMES imported from interpkit.core.arch at module top.


def _find_attn_submodule(
    layer_mod: torch.nn.Module,
) -> tuple[str, torch.nn.Module] | None:
    """Find the attention submodule inside a layer (recursive BFS).

    Identifies attention submodules by canonical attribute name set
    (no regex). Same set used across the codebase for consistency.
    """
    queue: deque[tuple[str, torch.nn.Module]] = deque()
    for name, mod in layer_mod.named_children():
        queue.append((name, mod))
    while queue:
        rel_name, mod = queue.popleft()
        base = rel_name.rsplit(".", 1)[-1].lower()
        if base in _ATTN_NAMES:
            return rel_name, mod
        for child_name, child_mod in mod.named_children():
            queue.append((f"{rel_name}.{child_name}", child_mod))
    return None


def _find_mlp_submodule(
    layer_mod: torch.nn.Module,
) -> tuple[str, torch.nn.Module] | None:
    """Find the MLP submodule inside a layer (recursive BFS).

    Identifies MLP submodules by canonical attribute name set (no regex).
    """
    queue: deque[tuple[str, torch.nn.Module]] = deque()
    for name, mod in layer_mod.named_children():
        queue.append((name, mod))
    while queue:
        rel_name, mod = queue.popleft()
        base = rel_name.rsplit(".", 1)[-1].lower()
        if base in _MLP_NAMES:
            return rel_name, mod
        for child_name, child_mod in mod.named_children():
            queue.append((f"{rel_name}.{child_name}", child_mod))
    return None
