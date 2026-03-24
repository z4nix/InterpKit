"""lens — logit lens: project each layer's output to vocabulary space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_lens(
    model: "Model",
    text: Any,
    *,
    save: str | None = None,
    html: str | None = None,
    position: int | None = None,
) -> list[dict[str, Any]] | None:
    """Project each layer's hidden state through the unembedding matrix.

    Analyses **all** token positions by default, producing the classic logit-lens
    heatmap (layers x positions).  Pass ``position=N`` to analyse a single
    token position (negative indices supported, e.g. ``position=-1`` for the
    last token — matching the original behaviour).

    Each entry in the returned list includes:

    - ``layer_name``, ``top1_token``, ``top1_prob``, ``top5_tokens``, ``top5_probs``
      for the *last* position (backward-compatible).
    - ``positions``: a list of per-position dicts, each containing
      ``pos``, ``top1_token``, ``top1_prob``, ``top5_tokens``, ``top5_probs``.

    The return value also has a ``tokens`` key on each entry when all
    positions are analysed, listing the input tokens.
    """
    from interpkit.core.render import render_lens

    arch = model.arch_info

    if not arch.is_language_model or arch.unembedding_name is None:
        console.print(
            f"\n  [yellow]lens not available:[/yellow] no unembedding matrix detected"
            f" for {arch.arch_family or 'this model'}.\n"
        )
        return None

    if not arch.layer_names:
        console.print(
            "\n  [yellow]lens not available:[/yellow] no layer structure detected.\n"
        )
        return None

    if model._tokenizer is None:
        console.print(
            "\n  [yellow]lens not available:[/yellow] no tokenizer loaded.\n"
        )
        return None

    text_input = model._prepare(text)

    unembed_mod = _get_module(model._model, arch.unembedding_name)
    unembed_weight = unembed_mod.weight  # (vocab_size, embed_dim) or (embed_dim, vocab_size) for Conv1D
    if type(unembed_mod).__name__ == "Conv1D":
        unembed_weight = unembed_weight.T  # normalize to (vocab_size, embed_dim)
    unembed_bias = getattr(unembed_mod, "bias", None)

    # Handle models where embed_dim != hidden_size (e.g. OPT-350m)
    project_out_mod = None
    if arch.project_out_path:
        try:
            project_out_mod = _get_module(model._model, arch.project_out_path)
        except AttributeError:
            pass

    layer_outputs: dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
            t = output if isinstance(output, torch.Tensor) else (
                output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
            )
            if t is not None:
                layer_outputs[name] = t.detach()
        return hook_fn

    hooks = []
    for layer_name in arch.layer_names:
        try:
            mod = _get_module(model._model, layer_name)
            hooks.append(mod.register_forward_hook(_make_hook(layer_name)))
        except AttributeError:
            continue

    try:
        with torch.no_grad():
            model._forward(text_input)
    finally:
        for h in hooks:
            h.remove()

    if not layer_outputs:
        console.print("\n  [yellow]lens:[/yellow] no layer outputs captured.\n")
        return None

    final_norm = _find_final_norm(model._model, arch)

    # Recover input tokens for labelling
    input_tokens: list[str] | None = None
    if isinstance(text, str) and model._tokenizer is not None:
        encoded = model._tokenizer(text, return_tensors="pt")
        input_tokens = model._tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0].tolist()
        )

    predictions: list[dict[str, Any]] = []

    for layer_name in arch.layer_names:
        if layer_name not in layer_outputs:
            continue

        hidden = layer_outputs[layer_name].float()  # (batch, seq, hidden) or (seq, hidden)

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)  # -> (1, seq, hidden)

        seq_len = hidden.shape[1]

        if final_norm is not None:
            norm_dtype = next(final_norm.parameters()).dtype
            hidden = final_norm(hidden.to(norm_dtype)).float()

        # Project through project_out if the model has embed_dim != hidden_size
        projected = hidden
        if project_out_mod is not None:
            proj_dtype = next(project_out_mod.parameters()).dtype
            projected = project_out_mod(projected.to(proj_dtype)).float()

        logits = projected @ unembed_weight.float().T
        if unembed_bias is not None:
            logits = logits + unembed_bias.float()
        probs = torch.softmax(logits, dim=-1)  # (batch, seq, vocab)

        # Determine which positions to report
        if position is not None:
            pos_idx = position if position >= 0 else seq_len + position
            pos_indices = [pos_idx]
        else:
            pos_indices = list(range(seq_len))

        per_position: list[dict[str, Any]] = []
        for pos in pos_indices:
            if pos < 0 or pos >= seq_len:
                continue
            top5_probs_t, top5_ids_t = probs[0, pos].topk(min(5, probs.shape[-1]))
            top5_tokens = [model._tokenizer.decode([tid]) for tid in top5_ids_t.tolist()]
            top5_probs_list = top5_probs_t.tolist()
            per_position.append({
                "pos": pos,
                "top1_token": top5_tokens[0],
                "top1_prob": top5_probs_list[0],
                "top5_tokens": top5_tokens,
                "top5_probs": top5_probs_list,
            })

        # Backward-compatible top-level fields use the last position
        last_pos_data = per_position[-1] if per_position else {
            "top1_token": "", "top1_prob": 0.0, "top5_tokens": [], "top5_probs": [],
        }

        entry: dict[str, Any] = {
            "layer_name": layer_name,
            "top1_token": last_pos_data["top1_token"],
            "top1_prob": last_pos_data["top1_prob"],
            "top5_tokens": last_pos_data["top5_tokens"],
            "top5_probs": last_pos_data["top5_probs"],
            "positions": per_position,
        }
        if input_tokens is not None:
            entry["tokens"] = input_tokens
        predictions.append(entry)

    model_name = arch.arch_family or "model"
    render_lens(predictions, model_name)

    if save is not None:
        from interpkit.core.plot import plot_lens

        plot_lens(predictions, save_path=save, input_tokens=input_tokens)

    if html is not None:
        import re as _re_html

        from interpkit.core.html import html_lens as gen_html_lens, save_html

        flat_preds = []
        for pred in predictions:
            _lm = _re_html.search(r"\.(\d+)", pred.get("layer_name", ""))
            li = int(_lm.group(1)) if _lm else 0
            for pos_data in pred.get("positions", []):
                flat_preds.append({
                    "layer": li,
                    "position": pos_data.get("pos", 0),
                    "prediction": pos_data.get("top1_token", "?"),
                    "prob": pos_data.get("top1_prob", 0.0),
                })
        save_html(gen_html_lens(flat_preds, input_tokens), html)

    return predictions


def _find_final_norm(model: torch.nn.Module, arch: Any) -> torch.nn.Module | None:
    """Try to find the final layer norm applied before the LM head."""
    import re

    norm_pattern = re.compile(
        r"^(model\.norm|transformer\.ln_f|gpt_neox\.final_layer_norm|"
        r"model\.final_layernorm|backbone\.norm_f)$",
        re.IGNORECASE,
    )
    for name, mod in model.named_modules():
        if norm_pattern.match(name):
            return mod

    # Generic fallback: look for a top-level norm module (LayerNorm or RMSNorm variants)
    for name, mod in model.named_modules():
        if name.count(".") <= 1:
            cls_name = type(mod).__name__.lower()
            is_norm = isinstance(mod, torch.nn.LayerNorm) or "rmsnorm" in cls_name or "layernorm" in cls_name
            if is_norm and ("norm" in name.lower() or "Norm" in type(mod).__name__):
                return mod

    return None
