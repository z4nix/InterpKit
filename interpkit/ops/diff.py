"""diff — compare activations between two models on the same input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_diff(
    model_a: "Model",
    model_b: "Model",
    input_data: Any,
    *,
    save: str | None = None,
) -> dict[str, Any]:
    """Compare activations between two models at all discovered layers.

    Returns a dict with ``"results"`` (list sorted by cosine distance,
    highest first), ``"skipped_a"`` (count of modules only in model A),
    and ``"skipped_b"`` (count of modules only in model B).
    """
    from interpkit.core.render import render_diff
    from interpkit.ops.activations import run_activations

    layers_a = set(model_a.arch_info.layer_names or [])
    layers_b = set(model_b.arch_info.layer_names or [])

    if not layers_a:
        layers_a = {m.name for m in model_a.arch_info.modules if m.param_count > 0}
    if not layers_b:
        layers_b = {m.name for m in model_b.arch_info.modules if m.param_count > 0}

    only_in_a = layers_a - layers_b
    only_in_b = layers_b - layers_a
    shared_layers = sorted(layers_a & layers_b)

    model_a_name = model_a.arch_info.arch_family or "model_a"
    model_b_name = model_b.arch_info.arch_family or "model_b"

    if only_in_a or only_in_b:
        parts = []
        if only_in_a:
            parts.append(f"{len(only_in_a)} modules only in {model_a_name}")
        if only_in_b:
            parts.append(f"{len(only_in_b)} modules only in {model_b_name}")
        console.print(
            f"\n  [yellow]diff:[/yellow] skipped {', '.join(parts)}. "
            f"Comparing {len(shared_layers)} shared modules."
        )

    if not shared_layers:
        console.print("\n  [yellow]diff:[/yellow] no shared modules found between the two models.\n")
        return {"results": [], "skipped_a": len(only_in_a), "skipped_b": len(only_in_b)}

    acts_a = run_activations(model_a, input_data, at=shared_layers, print_stats=False)
    acts_b = run_activations(model_b, input_data, at=shared_layers, print_stats=False)

    results: list[dict[str, Any]] = []
    for name in shared_layers:
        if name not in acts_a or name not in acts_b:
            continue

        a = acts_a[name].float().cpu().view(-1)
        b = acts_b[name].float().cpu().view(-1)

        if a.numel() == 0 or b.numel() == 0:
            continue

        if a.shape != b.shape:
            min_size = min(a.numel(), b.numel())
            a = a[:min_size]
            b = b[:min_size]

        if a.norm() == 0 and b.norm() == 0:
            distance = 0.0
        else:
            cosine_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)
            distance = (1.0 - cosine_sim.item())

        results.append({
            "module": name,
            "distance": distance,
        })

    results.sort(key=lambda r: r["distance"], reverse=True)

    render_diff(results, model_a_name, model_b_name)

    if save is not None:
        from interpkit.core.plot import plot_diff

        plot_diff(results, model_a_name=model_a_name, model_b_name=model_b_name, save_path=save)

    return {"results": results, "skipped_a": len(only_in_a), "skipped_b": len(only_in_b)}
