"""diff — compare activations between two models on the same input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mechkit.core.model import Model


def run_diff(
    model_a: "Model",
    model_b: "Model",
    input_data: Any,
    *,
    save: str | None = None,
) -> list[dict[str, Any]]:
    """Compare activations between two models at all discovered layers.

    Returns a list of dicts sorted by cosine distance (highest change first).
    """
    from mechkit.core.render import render_diff
    from mechkit.ops.activations import run_activations

    # Find shared layer-like modules
    layers_a = set(model_a.arch_info.layer_names or [])
    layers_b = set(model_b.arch_info.layer_names or [])

    # If no layers detected, fall back to all named modules
    if not layers_a:
        layers_a = {m.name for m in model_a.arch_info.modules if m.param_count > 0}
    if not layers_b:
        layers_b = {m.name for m in model_b.arch_info.modules if m.param_count > 0}

    shared_layers = sorted(layers_a & layers_b)

    if not shared_layers:
        from rich.console import Console
        Console().print("\n  [yellow]diff:[/yellow] no shared modules found between the two models.\n")
        return []

    acts_a = run_activations(model_a, input_data, at=shared_layers, print_stats=False)
    acts_b = run_activations(model_b, input_data, at=shared_layers, print_stats=False)

    results: list[dict[str, Any]] = []
    for name in shared_layers:
        if name not in acts_a or name not in acts_b:
            continue

        a = acts_a[name].float().view(-1)
        b = acts_b[name].float().view(-1)

        if a.shape != b.shape:
            min_size = min(a.numel(), b.numel())
            a = a[:min_size]
            b = b[:min_size]

        cosine_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)
        distance = (1.0 - cosine_sim.item())

        results.append({
            "module": name,
            "distance": distance,
        })

    results.sort(key=lambda r: r["distance"], reverse=True)

    model_a_name = model_a.arch_info.arch_family or "model_a"
    model_b_name = model_b.arch_info.arch_family or "model_b"
    render_diff(results, model_a_name, model_b_name)

    if save is not None:
        from mechkit.core.plot import plot_diff

        plot_diff(results, model_a_name=model_a_name, model_b_name=model_b_name, save_path=save)

    return results
