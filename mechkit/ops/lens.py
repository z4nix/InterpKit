"""lens — logit lens: project each layer's output to vocabulary space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console

from mechkit.ops.patch import _get_module

if TYPE_CHECKING:
    from mechkit.core.model import Model

console = Console()


def run_lens(model: "Model", text: Any, *, save: str | None = None) -> list[dict[str, Any]] | None:
    """Project each layer's hidden state through the unembedding matrix.

    Only works for language models with a detectable output head.
    """
    from mechkit.core.render import render_lens

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

    # Get the unembedding weight matrix
    unembed_mod = _get_module(model._model, arch.unembedding_name)
    unembed_weight = unembed_mod.weight  # shape: (vocab_size, hidden_size)

    # Capture hidden states at the output of each layer
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

    with torch.no_grad():
        model._forward(text_input)

    for h in hooks:
        h.remove()

    if not layer_outputs:
        console.print("\n  [yellow]lens:[/yellow] no layer outputs captured.\n")
        return None

    # Apply layer norm if the model has a final layer norm before the head
    # (common pattern: ln_f, model.norm, etc.)
    final_norm = _find_final_norm(model._model, arch)

    predictions: list[dict[str, Any]] = []

    for layer_name in arch.layer_names:
        if layer_name not in layer_outputs:
            continue

        hidden = layer_outputs[layer_name].float()

        # Take the last token position
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]  # (batch, hidden)
        elif hidden.dim() == 2:
            hidden = hidden[-1:, :]

        # Apply final norm if found
        if final_norm is not None:
            hidden = final_norm(hidden)

        # Project through unembedding: logits = hidden @ W^T
        logits = hidden @ unembed_weight.float().T  # (batch, vocab)
        probs = torch.softmax(logits, dim=-1)

        top5_probs, top5_ids = probs[0].topk(5)
        top5_tokens = [model._tokenizer.decode([tid]) for tid in top5_ids.tolist()]
        top5_probs_list = top5_probs.tolist()

        predictions.append({
            "layer_name": layer_name,
            "top1_token": top5_tokens[0],
            "top1_prob": top5_probs_list[0],
            "top5_tokens": top5_tokens,
            "top5_probs": top5_probs_list,
        })

    model_name = arch.arch_family or "model"
    render_lens(predictions, model_name)

    if save is not None:
        from mechkit.core.plot import plot_lens

        plot_lens(predictions, save_path=save)

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

    # Generic fallback: look for a top-level norm module
    for name, mod in model.named_modules():
        if name.count(".") <= 1 and isinstance(mod, (torch.nn.LayerNorm,)):
            type_name = type(mod).__name__
            if "norm" in name.lower() or "Norm" in type_name:
                return mod

    return None
