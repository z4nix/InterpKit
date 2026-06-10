"""find_circuit — automated circuit discovery via iterative ablation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.core.theme import ACCENT, MUTED
from interpkit.ops._hooks import register_capture_hook
from interpkit.ops.patch import _compute_effect_value as _compute_effect
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def _make_ablation_hook(
    at: str, method: str = "mean", *, resample_act: torch.Tensor | None = None,
):
    """Compile an ablation forward hook via :class:`AblateIntervention`.

    The replacement math (zeros / sequence mean / resampled corrupted
    activation) lives in :mod:`interpkit.core.interventions` — the single
    canonical implementation shared with ``ops/ablate.py``.
    """
    from interpkit.core.interventions import AblateIntervention

    return AblateIntervention(
        at, method=method, replacement=resample_act,
    ).build_hook(None)


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
        Component-selection strategy.

        Ablation methods (one forward per component per pair):
        ``"mean"`` (default — recommended), ``"zero"``, or
        ``"resample"`` (replace with corrupted-input activations).

        Gradient methods (a handful of passes total, phase 2):
        ``"eap"`` ranks components by Edge Attribution Patching scores;
        ``"eap-ig"`` uses 5 interpolated backward passes for more
        faithful scores. For both, *threshold* is interpreted as a
        fraction of the top component's absolute score, and the
        discovered circuit is still verified **causally** in phase 3
        via mean ablation of the excluded components.
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

    from interpkit.core.enums import VALID_FIND_CIRCUIT_METHODS, _validate_enum

    _validate_enum(method, VALID_FIND_CIRCUIT_METHODS, "method")
    use_eap = method in ("eap", "eap-ig")
    # EAP methods select components by gradient scores but verify the
    # discovered circuit *causally* — the phase-3 ablation below always
    # runs, using mean ablation for the EAP path.
    ablation_method = "mean" if use_eap else method

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
                    cache_hooks.append(
                        register_capture_hook(comp["module"], corrupted_acts, key)
                    )

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

    # Phase 1: measure each component's importance.
    # - Ablation methods: ablate each component individually (one forward
    #   per component per pair).
    # - EAP methods: one clean forward + one corrupted backward per pair
    #   scores all components at once; "effect" is the absolute EAP node
    #   score normalised by the largest component score, so the same
    #   threshold semantics apply.
    component_effects: list[dict[str, Any]] = []
    eap_edges: list[dict[str, Any]] | None = None
    eap_ig_steps = 0

    if use_eap:
        from interpkit.ops.eap import run_eap

        eap_ig_steps = 5 if method == "eap-ig" else 0
        node_scores: dict[str, list[float]] = {}
        edge_scores: dict[tuple[str, str], list[float]] = {}
        for c, r in zip(cleans, corrupteds):
            eap_result = run_eap(
                model, c, r,
                ig_steps=eap_ig_steps, top_k_edges=None, render=False,
            )
            for nd in eap_result["nodes"]:
                if nd["type"] in ("attn", "mlp") and nd["score"] == nd["score"]:
                    node_scores.setdefault(nd["node"], []).append(nd["score"])
            for ed in eap_result["edges"]:
                if ed["score"] == ed["score"]:
                    edge_scores.setdefault((ed["src"], ed["dst"]), []).append(
                        ed["score"]
                    )

        mean_scores = {
            name: sum(vals) / len(vals) for name, vals in node_scores.items()
        }
        max_abs = max((abs(v) for v in mean_scores.values()), default=1.0) or 1.0
        for comp in components:
            raw = mean_scores.get(comp["component"])
            effect = abs(raw) / max_abs if raw is not None else 0.0
            component_effects.append({
                "component": comp["component"],
                "layer": comp["layer"],
                "type": comp["type"],
                "effect": effect,
                "eap_score": raw,
                "module_name": comp["module_name"],
                "module": comp["module"],
            })

        eap_edges = [
            {"src": src, "dst": dst, "score": sum(vals) / len(vals)}
            for (src, dst), vals in edge_scores.items()
        ]
        eap_edges.sort(key=lambda e: abs(e["score"]), reverse=True)
        for i, e in enumerate(eap_edges):
            e["rank"] = i

    if not use_eap:
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
                        _make_ablation_hook(
                            comp["module_name"], ablation_method, resample_act=resample_act,
                        )
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
                        _make_ablation_hook(
                            comp["module_name"], ablation_method, resample_act=resample_act,
                        )
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
    if use_eap:
        # Additive keys — the legacy schema above is unchanged.
        result["edges"] = eap_edges
        result["meta"] = {
            "method": method,
            "ig_steps": eap_ig_steps,
            "selection": "eap",
            "verification_ablation": ablation_method,
            "effect_definition": (
                "absolute EAP node score normalised by the largest component "
                "score (threshold is a fraction of the top component)"
            ),
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
        table.add_column("Type", style=MUTED)
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
