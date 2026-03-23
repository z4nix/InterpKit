"""activations — extract raw activation tensors at any named module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_activations(
    model: "Model",
    input_data: Any,
    *,
    at: str | list[str],
    print_stats: bool = True,
) -> dict[str, torch.Tensor] | torch.Tensor:
    """Extract activations at one or more named modules.

    Returns a single tensor if *at* is a string, or a dict if *at* is a list.
    """
    model_input = model._prepare(input_data)
    single = isinstance(at, str)
    module_names = [at] if single else list(at)

    # Check activation cache first (pass prepared input to avoid re-tokenizing)
    cached = model._get_cached(input_data, module_names, _prepared_input=model_input)
    if cached is not None:
        cache = cached
    else:
        cache = {}

        def _make_hook(name: str):
            def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
                t = output if isinstance(output, torch.Tensor) else (
                    output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                )
                if t is not None:
                    cache[name] = t.detach().clone()
            return hook_fn

        hooks = []
        for name in module_names:
            mod = _get_module(model._model, name)
            hooks.append(mod.register_forward_hook(_make_hook(name)))

        try:
            with torch.no_grad():
                model._forward(model_input)
        finally:
            for h in hooks:
                h.remove()

    if print_stats:
        from interpkit.core.render import render_activations

        render_activations(cache)

    if single:
        if at not in cache:
            raise RuntimeError(f"Module '{at}' produced no tensor output.")
        return cache[at]

    return cache
