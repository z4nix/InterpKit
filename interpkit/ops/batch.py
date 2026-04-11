"""batch — run any operation over a dataset with result aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.progress import Progress

from interpkit.core.theme import ACCENT

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_batch(
    model: Model,
    operation: str,
    dataset: list[dict[str, Any]],
    *,
    op_kwargs: dict[str, Any] | None = None,
    aggregate: bool = True,
) -> dict[str, Any]:
    """Run an InterpKit operation over a dataset of examples.

    Parameters
    ----------
    operation:
        Name of the Model method to call: ``"trace"``, ``"patch"``,
        ``"lens"``, ``"attribute"``, ``"ablate"``, ``"activations"``,
        ``"dla"``, etc.
    dataset:
        List of dicts.  Each dict is unpacked as keyword arguments to the
        operation.  Common keys: ``"input_data"``, ``"clean"``,
        ``"corrupted"``, ``"text"``, ``"at"``.

        For paired operations (trace, patch), supply ``"clean"`` and
        ``"corrupted"``.  For single-input ops, supply ``"input_data"``
        or ``"text"``.
    op_kwargs:
        Extra keyword arguments applied to every call (e.g. ``at``,
        ``top_k``).
    aggregate:
        If *True* (default), compute summary statistics across all results.

    Returns
    -------
    dict with:
        ``results`` — list of per-example results
        ``summary`` — aggregated statistics (when *aggregate* is True)
        ``count`` — number of examples processed
    """
    op_kwargs = op_kwargs or {}
    method = getattr(model, operation, None)
    if method is None:
        raise ValueError(f"Unknown operation: {operation!r}")

    results: list[Any] = []
    errors: list[dict[str, Any]] = []

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task(f"Batch {operation}", total=len(dataset))
        for i, example in enumerate(dataset):
            try:
                merged = {**op_kwargs, **example}
                result = method(**merged)
                results.append(result)
            except Exception as e:
                errors.append({
                    "index": i,
                    "error": str(e),
                    "type": type(e).__name__,
                })
                console.print(
                    f"  [dim]batch: example {i} failed "
                    f"({type(e).__name__}: {e})[/dim]"
                )
            progress.advance(task)

    output: dict[str, Any] = {
        "results": results,
        "count": len(results),
        "total": len(dataset),
        "errors": errors,
    }

    if aggregate and results:
        output["summary"] = _aggregate(results, operation)

    if errors:
        console.print(f"\n  [yellow]batch:[/yellow] {len(errors)} errors out of {len(dataset)} examples.")

    _render_summary(output, operation)
    return output


def run_trace_batch(
    model: Model,
    dataset: list[dict[str, str]],
    *,
    clean_col: str = "clean",
    corrupted_col: str = "corrupted",
    top_k: int | None = 20,
    mode: str = "module",
) -> dict[str, Any]:
    """Convenience: run causal tracing over a dataset.

    Each entry in *dataset* should have keys matching *clean_col* and
    *corrupted_col*.
    """
    examples = [
        {"clean": ex[clean_col], "corrupted": ex[corrupted_col]}
        for ex in dataset
    ]
    return run_batch(
        model, "trace", examples,
        op_kwargs={"top_k": top_k, "mode": mode},
    )


def run_dla_batch(
    model: Model,
    texts: list[str],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Convenience: run DLA over a list of texts."""
    examples = [{"input_data": t} for t in texts]
    return run_batch(model, "dla", examples, op_kwargs={"top_k": top_k})


def _aggregate(results: list[Any], operation: str) -> dict[str, Any]:
    """Compute summary statistics over batch results."""
    summary: dict[str, Any] = {}

    if operation in ("trace",):
        # Module-level trace: aggregate effect scores per module
        # Position-mode trace returns dicts — skip aggregation for those
        if isinstance(results[0], dict) and "effects" in results[0]:
            summary["note"] = "Position-mode trace results; per-module aggregation not applicable."
        elif isinstance(results[0], list):
            module_effects: dict[str, list[float]] = {}
            for result in results:
                for entry in result:
                    name = entry.get("module", "")
                    module_effects.setdefault(name, []).append(entry.get("effect", 0.0))

            ranked: list[dict[str, Any]] = []
            for module, effects in module_effects.items():
                ranked.append({
                    "module": module,
                    "mean_effect": sum(effects) / len(effects),
                    "max_effect": max(effects),
                    "min_effect": min(effects),
                    "count": len(effects),
                })
            ranked.sort(key=lambda x: float(x["mean_effect"]), reverse=True)
            summary["ranked_modules"] = ranked

    elif operation in ("patch", "ablate"):
        effects = [r.get("effect", 0.0) for r in results if isinstance(r, dict)]
        if effects:
            summary["mean_effect"] = sum(effects) / len(effects)
            summary["max_effect"] = max(effects)
            summary["min_effect"] = min(effects)
            summary["std_effect"] = (
                sum((e - summary["mean_effect"]) ** 2 for e in effects) / len(effects)
            ) ** 0.5

    elif operation == "dla":
        # Aggregate which components contribute most across examples
        component_contribs: dict[str, list[float]] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            for c in r.get("contributions", []):
                name = c.get("component", "")
                component_contribs.setdefault(name, []).append(
                    c.get("logit_contribution", 0.0)
                )
        ranked_comps: list[dict[str, Any]] = []
        for comp, vals in component_contribs.items():
            ranked_comps.append({
                "component": comp,
                "mean_contribution": sum(vals) / len(vals),
                "count": len(vals),
            })
        ranked_comps.sort(key=lambda x: abs(float(x["mean_contribution"])), reverse=True)
        summary["ranked_components"] = ranked_comps

    elif operation == "attribute":
        # Aggregate token scores across examples
        all_scores: dict[str, list[float]] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            for tok, score in zip(r.get("tokens", []), r.get("scores", [])):
                all_scores.setdefault(tok, []).append(score)
        summary["token_mean_scores"] = {
            tok: sum(vals) / len(vals) for tok, vals in all_scores.items()
        }

    return summary


def _render_summary(output: dict[str, Any], operation: str) -> None:
    """Print a brief batch summary."""
    from rich.table import Table

    summary = output.get("summary")
    count = output["count"]

    console.print(f"\n[bold]Batch {operation}: {count} examples[/bold]")

    if not summary:
        return

    if "ranked_modules" in summary:
        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("Module", style=ACCENT)
        table.add_column("Mean Effect", justify="right")
        table.add_column("Max", justify="right", style="dim")
        table.add_column("Count", justify="right", style="dim")

        for entry in summary["ranked_modules"][:10]:
            table.add_row(
                entry["module"],
                f"{entry['mean_effect']:.3f}",
                f"{entry['max_effect']:.3f}",
                str(entry["count"]),
            )
        console.print(table)

    elif "mean_effect" in summary:
        console.print(
            f"  Mean effect: {summary['mean_effect']:.3f}  "
            f"(std: {summary.get('std_effect', 0):.3f}, "
            f"range: [{summary['min_effect']:.3f}, {summary['max_effect']:.3f}])"
        )

    elif "ranked_components" in summary:
        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("Component", style=ACCENT)
        table.add_column("Mean Contribution", justify="right")
        table.add_column("Count", justify="right", style="dim")

        for entry in summary["ranked_components"][:10]:
            table.add_row(
                entry["component"],
                f"{entry['mean_contribution']:+.4f}",
                str(entry["count"]),
            )
        console.print(table)

    console.print()
