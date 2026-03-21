"""find_circuit — automated circuit discovery via iterative ablation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.ops.patch import _compute_effect, _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def _make_ablation_hook(method: str = "mean", *, resample_act: torch.Tensor | None = None):
    """Forward hook that replaces tensor output according to *method*.

    ``"zero"``     — replace with zeros.
    ``"mean"``     — replace with the mean across the sequence dimension.
    ``"resample"`` — replace with *resample_act* (e.g. corrupted activation).
    """
    def hook_fn(_mod, _inp, output):
        t = output if isinstance(output, torch.Tensor) else (
            output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
        )
        if t is None:
            return output

        if method == "zero":
            replacement = torch.zeros_like(t)
        elif method == "mean":
            if t.dim() >= 3:
                replacement = t.mean(dim=-2, keepdim=True).expand_as(t)
            else:
                replacement = t.mean(dim=0, keepdim=True).expand_as(t)
        elif method == "resample":
            if resample_act is not None:
                replacement = resample_act.to(t.device)
            else:
                replacement = torch.zeros_like(t)
        else:
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
    method: str = "mean",
    metric: str = "logit_diff",
) -> dict[str, Any]:
    """Discover the minimal circuit that explains a behaviour.

    Identifies which attention heads and MLPs are necessary for the model's
    output on the *clean* input to differ from the *corrupted* input.

    Algorithm:
      1. Ablate each component individually and measure effect.
      2. Keep only components whose ablation changes the output by more than
         *threshold*.
      3. Verify the discovered circuit by ablating all non-circuit components
         simultaneously and checking that the output is preserved.

    Parameters
    ----------
    threshold:
        Minimum ablation effect for a component to be included in the
        circuit.  Lower values include more components.
    method:
        Ablation strategy: ``"mean"`` (default — recommended),
        ``"zero"``, or ``"resample"`` (replace with corrupted-input
        activations).
    metric:
        Effect metric passed to ``_compute_effect``.

    Returns
    -------
    dict with:
        ``circuit`` — list of ``{"component", "layer", "type", "effect"}``
            for components in the circuit, sorted by effect.
        ``excluded`` — list of excluded components.
        ``verification`` — dict with ``circuit_effect`` and ``faithfulness``.
        ``threshold`` — the threshold used.
    """
    arch = model.arch_info
    if not arch.layer_names:
        raise ValueError("Circuit discovery requires detected layer structure.")

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    with torch.no_grad():
        clean_logits = model._forward(clean_input)
        corrupted_logits = model._forward(corrupted_input)

    # Enumerate all attention and MLP components
    components: list[dict[str, Any]] = []
    for li in arch.layer_infos:
        if li.attn_path:
            attn_mod = _get_module(model._model, li.attn_path)
            components.append({
                "component": f"L{li.index}.attn",
                "layer": li.index,
                "type": "attn",
                "module_name": li.attn_path,
                "module": attn_mod,
            })
        if li.mlp_path:
            mlp_mod = _get_module(model._model, li.mlp_path)
            components.append({
                "component": f"L{li.index}.mlp",
                "layer": li.index,
                "type": "mlp",
                "module_name": li.mlp_path,
                "module": mlp_mod,
            })

    if not components:
        raise ValueError("No attention or MLP components found for circuit discovery.")

    ablation_method = method if method in ("zero", "mean", "resample") else "mean"

    # For resample ablation, cache each component's corrupted-input activation
    corrupted_acts: dict[str, torch.Tensor] = {}
    if ablation_method == "resample":
        cache_hooks: list[torch.utils.hooks.RemovableHook] = []
        for comp in components:
            key = comp["module_name"]

            def _cache(k: str):
                def fn(_mod, _inp, output):
                    t = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                    )
                    if t is not None:
                        corrupted_acts[k] = t.detach().clone()
                return fn

            cache_hooks.append(comp["module"].register_forward_hook(_cache(key)))

        with torch.no_grad():
            model._forward(corrupted_input)
        for h in cache_hooks:
            h.remove()

    # Phase 1: individual ablation — measure each component's importance
    component_effects: list[dict[str, Any]] = []

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Evaluating components", total=len(components))
        for comp in components:
            resample_act = corrupted_acts.get(comp["module_name"]) if ablation_method == "resample" else None
            handle = comp["module"].register_forward_hook(
                _make_ablation_hook(ablation_method, resample_act=resample_act)
            )
            with torch.no_grad():
                ablated_logits = model._forward(clean_input)
            handle.remove()

            effect = _compute_effect(clean_logits, corrupted_logits, ablated_logits, metric=metric)
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
            resample_act = corrupted_acts.get(comp["module_name"]) if ablation_method == "resample" else None
            hooks.append(comp["module"].register_forward_hook(
                _make_ablation_hook(ablation_method, resample_act=resample_act)
            ))

        with torch.no_grad():
            circuit_only_logits = model._forward(clean_input)

        for h in hooks:
            h.remove()

        faithfulness = _compute_effect(
            clean_logits, corrupted_logits, circuit_only_logits, metric=metric
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
