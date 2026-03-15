"""find_circuit — automated circuit discovery via iterative ablation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.ops.patch import _compute_effect, _get_module
from interpkit.ops.dla import _find_attn_submodule, _find_mlp_submodule

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def _make_zero_hook():
    """Forward hook that replaces tensor output with zeros."""
    def hook_fn(_mod, _inp, output):
        t = output if isinstance(output, torch.Tensor) else (
            output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
        )
        if t is None:
            return output
        replacement = torch.zeros_like(t)
        if isinstance(output, torch.Tensor):
            return replacement
        return (replacement,) + tuple(output[1:])
    return hook_fn


def run_find_circuit(
    model: "Model",
    clean: Any,
    corrupted: Any,
    *,
    threshold: float = 0.01,
    method: str = "ablation",
) -> dict[str, Any]:
    """Discover the minimal circuit that explains a behaviour.

    Identifies which attention heads and MLPs are necessary for the model's
    output on the *clean* input to differ from the *corrupted* input.

    Algorithm (``method="ablation"``):
      1. Compute the clean vs corrupted effect for each component individually.
      2. Keep only components whose ablation changes the output by more than
         *threshold* (normalised effect, 0-1 scale).
      3. Verify the discovered circuit by ablating all non-circuit components
         simultaneously and checking that the output is preserved.

    Parameters
    ----------
    threshold:
        Minimum ablation effect for a component to be included in the
        circuit (0-1 scale).  Lower values include more components.
    method:
        ``"ablation"`` — zero-ablate each component and measure effect.

    Returns
    -------
    dict with:
        ``circuit`` — list of ``{"component", "layer", "type", "effect"}``
            for components in the circuit, sorted by effect.
        ``excluded`` — list of excluded components.
        ``verification`` — dict with ``circuit_effect`` (how much the
            output changes when ALL non-circuit components are ablated)
            and ``faithfulness`` (1 = perfect preservation).
        ``threshold`` — the threshold used.
    """
    arch = model.arch_info
    if not arch.layer_names:
        raise ValueError("Circuit discovery requires detected layer structure.")

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    clean_logits = model._forward(clean_input)
    corrupted_logits = model._forward(corrupted_input)

    # Enumerate all attention and MLP components
    components: list[dict[str, Any]] = []
    for layer_name in arch.layer_names:
        layer_mod = _get_module(model._model, layer_name)
        layer_match = re.search(r"\.(\d+)", layer_name)
        layer_idx = int(layer_match.group(1)) if layer_match else 0

        attn_child = _find_attn_submodule(layer_mod)
        if attn_child is not None:
            components.append({
                "component": f"L{layer_idx}.attn",
                "layer": layer_idx,
                "type": "attn",
                "module_name": f"{layer_name}.{attn_child[0]}",
                "module": attn_child[1],
            })

        mlp_child = _find_mlp_submodule(layer_mod)
        if mlp_child is not None:
            components.append({
                "component": f"L{layer_idx}.mlp",
                "layer": layer_idx,
                "type": "mlp",
                "module_name": f"{layer_name}.{mlp_child[0]}",
                "module": mlp_child[1],
            })

    if not components:
        raise ValueError("No attention or MLP components found for circuit discovery.")

    # Phase 1: individual ablation — measure each component's importance
    component_effects: list[dict[str, Any]] = []

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Evaluating components", total=len(components))
        for comp in components:
            handle = comp["module"].register_forward_hook(_make_zero_hook())
            ablated_logits = model._forward(clean_input)
            handle.remove()

            effect = _compute_effect(clean_logits, corrupted_logits, ablated_logits)
            # Here "effect" measures how much ablating this component moves
            # output toward corrupted (i.e. how much it contributes to the clean behavior)
            ablation_effect = 1.0 - effect

            component_effects.append({
                "component": comp["component"],
                "layer": comp["layer"],
                "type": comp["type"],
                "effect": ablation_effect,
                "module_name": comp["module_name"],
                "module": comp["module"],
            })
            progress.advance(task)

    # Phase 2: threshold to select circuit
    circuit = [c for c in component_effects if c["effect"] >= threshold]
    excluded = [c for c in component_effects if c["effect"] < threshold]

    circuit.sort(key=lambda c: c["effect"], reverse=True)
    excluded.sort(key=lambda c: c["effect"], reverse=True)

    # Phase 3: verification — ablate all excluded components simultaneously
    verification = {"circuit_effect": 0.0, "faithfulness": 0.0}

    if excluded:
        hooks = []
        for comp in excluded:
            hooks.append(comp["module"].register_forward_hook(_make_zero_hook()))

        circuit_only_logits = model._forward(clean_input)

        for h in hooks:
            h.remove()

        faithfulness = _compute_effect(
            clean_logits, corrupted_logits, circuit_only_logits
        )
        verification["faithfulness"] = faithfulness
        verification["circuit_effect"] = 1.0 - faithfulness
    else:
        verification["faithfulness"] = 1.0
        verification["circuit_effect"] = 0.0

    # Clean up module refs from output
    clean_circuit = [
        {k: v for k, v in c.items() if k not in ("module", "module_name")}
        for c in circuit
    ]
    clean_excluded = [
        {k: v for k, v in c.items() if k not in ("module", "module_name")}
        for c in excluded
    ]

    result = {
        "circuit": clean_circuit,
        "excluded": clean_excluded,
        "verification": verification,
        "threshold": threshold,
        "total_components": len(component_effects),
    }

    _render_circuit(result)
    return result


def _render_circuit(result: dict[str, Any]) -> None:
    """Print circuit discovery results."""
    from rich.table import Table
    from rich.panel import Panel

    circuit = result["circuit"]
    excluded = result["excluded"]
    verification = result["verification"]

    console.print(f"\n[bold]Circuit Discovery[/bold]")
    console.print(
        f"  Threshold: {result['threshold']}  |  "
        f"Circuit: {len(circuit)}/{result['total_components']} components  |  "
        f"Faithfulness: {verification['faithfulness']:.1%}\n"
    )

    if circuit:
        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("Component", style="cyan")
        table.add_column("Type", style="dim")
        table.add_column("Effect", justify="right", style="bold")
        table.add_column("", min_width=15)

        max_eff = max(c["effect"] for c in circuit) or 1.0
        for c in circuit[:20]:
            bar_len = int(c["effect"] / max_eff * 12)
            bar = f"[green]{'█' * bar_len}[/green]"
            table.add_row(c["component"], c["type"], f"{c['effect']:.3f}", bar)

        if len(circuit) > 20:
            table.add_row("...", "", "", f"({len(circuit) - 20} more)")

        console.print(table)

    console.print(
        f"\n  [dim]Verification: ablating {len(excluded)} non-circuit components "
        f"preserves {verification['faithfulness']:.1%} of the clean→corrupted distinction.[/dim]\n"
    )
