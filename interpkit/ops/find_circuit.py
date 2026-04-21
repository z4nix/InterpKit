"""find_circuit — automated circuit discovery via iterative ablation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.core.theme import ACCENT
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
            output[0] if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor) else None
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
    model: Model,
    clean: Any | list[Any],
    corrupted: Any | list[Any],
    *,
    threshold: float = 0.01,
    method: str = "mean",
    metric: str = "logit_diff",
) -> dict[str, Any]:
    """Discover the minimal circuit that explains a behaviour.

    Identifies which attention heads and MLPs are necessary for the model's
    output on the *clean* input to differ from the *corrupted* input.

    *clean* and *corrupted* may each be a single input or parallel lists.
    When lists are provided, ablation effects are averaged across all pairs
    to produce a more robust circuit.

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

    # Normalise to lists of pairs.  Use normalize_input_group so a chat-message
    # list (also a Python list) is treated as a single example, not as
    # "one example per message dict".
    from interpkit.core.inputs import normalize_input_group

    cleans = normalize_input_group(clean)
    corrupteds = normalize_input_group(corrupted)
    if len(cleans) != len(corrupteds):
        raise ValueError(
            f"clean ({len(cleans)} examples) and corrupted ({len(corrupteds)} examples) "
            f"must have the same number of entries."
        )
    n_pairs = len(cleans)
    if n_pairs == 0:
        raise ValueError("At least one clean/corrupted pair is required.")

    # Prepare all pairs and cache baseline logits
    pairs: list[tuple[Any, Any, torch.Tensor, torch.Tensor]] = []
    if n_pairs > 1:
        with Progress(console=console, transient=True) as progress:
            task = progress.add_task("Preparing baselines", total=n_pairs)
            for c, r in zip(cleans, corrupteds):
                ci, ri = model._prepare_pair(c, r)
                with torch.no_grad():
                    cl = model._forward(ci)
                    rl = model._forward(ri)
                pairs.append((ci, ri, cl, rl))
                progress.advance(task)
    else:
        for c, r in zip(cleans, corrupteds):
            ci, ri = model._prepare_pair(c, r)
            with torch.no_grad():
                cl = model._forward(ci)
                rl = model._forward(ri)
            pairs.append((ci, ri, cl, rl))

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

    if method not in ("zero", "mean", "resample"):
        raise ValueError(
            f"method must be one of 'zero', 'mean', 'resample'; got {method!r}"
        )
    ablation_method = method

    # For resample ablation, cache each component's corrupted-input activations
    # per pair so we can swap them in during ablation.
    all_corrupted_acts: list[dict[str, torch.Tensor]] = []
    if ablation_method == "resample":
        _resample_progress = n_pairs > 1
        _resample_ctx = Progress(console=console, transient=True) if _resample_progress else None
        _resample_task = None
        if _resample_ctx is not None:
            _resample_ctx.start()
            _resample_task = _resample_ctx.add_task("Caching corrupted activations", total=n_pairs)
        try:
            for _ci, ri, _cl, _rl in pairs:
                corrupted_acts: dict[str, torch.Tensor] = {}
                cache_hooks: list = []
                for comp in components:
                    key = comp["module_name"]

                    def _cache(k: str, _acts=corrupted_acts):
                        def fn(_mod, _inp, output):
                            t = output if isinstance(output, torch.Tensor) else (
                                output[0] if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor) else None
                            )
                            if t is not None:
                                _acts[k] = t.detach().clone()
                        return fn

                    cache_hooks.append(comp["module"].register_forward_hook(_cache(key)))

                with torch.no_grad():
                    model._forward(ri)
                for h in cache_hooks:
                    h.remove()
                all_corrupted_acts.append(corrupted_acts)
                if _resample_ctx is not None and _resample_task is not None:
                    _resample_ctx.advance(_resample_task)
        finally:
            if _resample_ctx is not None:
                _resample_ctx.stop()

    # Phase 1: individual ablation — measure each component's importance
    component_effects: list[dict[str, Any]] = []

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Evaluating components", total=len(components))
        for comp in components:
            effect_sum = 0.0
            for pi, (ci, _ri, cl, rl) in enumerate(pairs):
                resample_act = (
                    all_corrupted_acts[pi].get(comp["module_name"])
                    if ablation_method == "resample" else None
                )
                handle = comp["module"].register_forward_hook(
                    _make_ablation_hook(ablation_method, resample_act=resample_act)
                )
                try:
                    with torch.no_grad():
                        ablated_logits = model._forward(ci)
                finally:
                    handle.remove()

                effect = _compute_effect(cl, rl, ablated_logits, metric=metric)
                effect_sum += 1.0 - effect

            ablation_effect = effect_sum / n_pairs

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
        faith_sum = 0.0
        _verify_progress = n_pairs > 1
        _verify_ctx = Progress(console=console, transient=True) if _verify_progress else None
        _verify_task = None
        if _verify_ctx is not None:
            _verify_ctx.start()
            _verify_task = _verify_ctx.add_task("Verifying circuit", total=n_pairs)
        try:
            for pi, (ci, _ri, cl, rl) in enumerate(pairs):
                hooks = []
                for comp in excluded:
                    resample_act = (
                        all_corrupted_acts[pi].get(comp["module_name"])
                        if ablation_method == "resample" else None
                    )
                    hooks.append(comp["module"].register_forward_hook(
                        _make_ablation_hook(ablation_method, resample_act=resample_act)
                    ))

                try:
                    with torch.no_grad():
                        circuit_only_logits = model._forward(ci)
                finally:
                    for h in hooks:
                        h.remove()

                faith_sum += _compute_effect(cl, rl, circuit_only_logits, metric=metric)
                if _verify_ctx is not None and _verify_task is not None:
                    _verify_ctx.advance(_verify_task)
        finally:
            if _verify_ctx is not None:
                _verify_ctx.stop()

        faithfulness = faith_sum / n_pairs
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
        "num_pairs": n_pairs,
    }

    _render_circuit(result)
    return result


def _render_circuit(result: dict[str, Any]) -> None:
    """Print circuit discovery results."""
    from rich.table import Table

    circuit = result["circuit"]
    excluded = result["excluded"]
    verification = result["verification"]

    n_pairs = result.get("num_pairs", 1)
    pairs_label = f"  |  Pairs: {n_pairs}" if n_pairs > 1 else ""

    console.print("\n[bold]Circuit Discovery[/bold]")
    console.print(
        f"  Threshold: {result['threshold']}  |  "
        f"Circuit: {len(circuit)}/{result['total_components']} components  |  "
        f"Faithfulness: {verification['faithfulness']:.1%}{pairs_label}\n"
    )

    if circuit:
        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("Component", style=ACCENT)
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
