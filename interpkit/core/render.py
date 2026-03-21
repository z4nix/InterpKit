"""Terminal rendering — rich tables, trees, unicode bar charts, heatmap export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

if TYPE_CHECKING:
    from interpkit.core.discovery import ModelArchInfo, ModuleInfo

console = Console()

_ROLE_TAGS = {
    "attention": "[attn]",
    "mlp": "[mlp]",
    "head": "[head]",
    "norm": "[norm]",
    "embed": "[embed]",
}


# ------------------------------------------------------------------
# Inspect rendering
# ------------------------------------------------------------------


def render_inspect(arch_info: "ModelArchInfo", nn_model: "torch.nn.Module | None" = None) -> None:
    """Print a module tree with types, param counts, and detected roles."""
    header_parts = []
    if arch_info.arch_family:
        header_parts.append(arch_info.arch_family)
    if arch_info.num_layers is not None:
        header_parts.append(f"{arch_info.num_layers} layers")
    if arch_info.hidden_size is not None:
        header_parts.append(f"hidden={arch_info.hidden_size}")
    if arch_info.vocab_size is not None:
        header_parts.append(f"vocab={arch_info.vocab_size}")

    if nn_model is not None:
        seen: set[int] = set()
        total_params = 0
        for p in nn_model.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total_params += p.numel()
    else:
        total_params = sum(m.param_count for m in arch_info.modules)
    header_parts.append(f"{_format_params(total_params)} params total")

    console.print(f"\n[bold]{' | '.join(header_parts)}[/bold]")

    table = Table(show_header=True, header_style="bold", show_lines=False, pad_edge=False)
    table.add_column("Module", style="cyan", no_wrap=True)
    table.add_column("Type", style="dim")
    table.add_column("Params", justify="right")
    table.add_column("Output Shape", style="dim")
    table.add_column("Role", style="bold yellow")

    for m in arch_info.modules:
        role_tag = _ROLE_TAGS.get(m.role or "", "")
        shape_str = str(m.output_shape) if m.output_shape else ""
        param_str = _format_params(m.param_count) if m.param_count > 0 else ""
        table.add_row(m.name, m.type_name, param_str, shape_str, role_tag)

    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Causal trace rendering
# ------------------------------------------------------------------


def render_trace(
    results: list[dict[str, Any]],
    model_name: str,
    total_modules: int,
    top_k: int | None,
) -> None:
    """Print a ranked bar chart of causal tracing results."""
    scanned = len(results)
    if top_k is not None:
        title = f"Causal Trace: {model_name}  (top {scanned} of {total_modules} modules)"
    else:
        title = f"Causal Trace: {model_name}  ({total_modules} modules)"

    console.print(f"\n[bold]{title}[/bold]")

    if not results:
        console.print("  No significant causal effects found.")
        return

    max_effect = max(r["effect"] for r in results) if results else 1.0
    bar_width = 30

    table = Table(show_header=False, show_lines=False, pad_edge=False, box=None)
    table.add_column("Module", style="cyan", no_wrap=True, min_width=35)
    table.add_column("Role", style="yellow", min_width=8)
    table.add_column("Bar", no_wrap=True)
    table.add_column("Effect", justify="right", style="bold")

    for r in results:
        fill = int(bar_width * r["effect"] / max_effect) if max_effect > 0 else 0
        bar = "█" * fill
        role = _ROLE_TAGS.get(r.get("role") or "", "")
        table.add_row(r["module"], role, f"[green]{bar}[/green]", f"{r['effect']:.3f}")

    console.print(table)

    if results:
        best = results[0]
        console.print(f"\n  Top component: [bold cyan]{best['module']}[/bold cyan] (effect: {best['effect']:.3f})")

    if top_k is not None and scanned < total_modules:
        console.print(
            f"  Run with --top-k 0 to scan all {total_modules} modules.\n"
        )
    else:
        console.print()


def render_position_trace(result: dict[str, Any]) -> None:
    """Print a summary table for position-aware causal tracing."""
    effects = result["effects"]  # (num_layers, seq_len)
    layer_names = result["layer_names"]
    tokens = result.get("tokens")

    if not isinstance(effects, torch.Tensor):
        effects = torch.tensor(effects)

    console.print("\n[bold]Position-Aware Causal Trace[/bold]")

    num_layers, seq_len = effects.shape

    # Find top-5 (layer, position) pairs by effect
    flat = effects.view(-1)
    top_k = min(10, flat.numel())
    top_vals, top_idxs = flat.topk(top_k)

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Layer", style="cyan")
    table.add_column("Position", justify="right")
    table.add_column("Token", style="yellow")
    table.add_column("Effect", justify="right", style="bold")

    for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist()), 1):
        li = idx // seq_len
        pi = idx % seq_len
        tok = tokens[pi] if tokens and pi < len(tokens) else str(pi)
        table.add_row(str(rank), layer_names[li], str(pi), tok, f"{val:.3f}")

    console.print(table)
    console.print(
        f"\n  Heatmap: {num_layers} layers x {seq_len} positions. "
        f"Use save= to export the full (layer, position) heatmap.\n"
    )


# ------------------------------------------------------------------
# Logit lens rendering
# ------------------------------------------------------------------


def render_lens(
    predictions: list[dict[str, Any]],
    model_name: str,
) -> None:
    """Print logit lens top predictions per layer."""
    console.print(f"\n[bold]Logit Lens: {model_name}[/bold]")

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Layer", style="cyan")
    table.add_column("Top-1 Token", style="bold")
    table.add_column("Prob", justify="right")
    table.add_column("Top-5 Tokens", style="dim")

    for pred in predictions:
        top5_str = ", ".join(
            f"{tok} ({prob:.2f})" for tok, prob in zip(pred["top5_tokens"], pred["top5_probs"])
        )
        table.add_row(
            pred["layer_name"],
            pred["top1_token"],
            f"{pred['top1_prob']:.3f}",
            top5_str,
        )

    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Attribution rendering
# ------------------------------------------------------------------


def render_attribution_tokens(
    tokens: list[str],
    scores: list[float],
) -> None:
    """Print tokens coloured by attribution score (terminal)."""
    console.print("\n[bold]Attribution (gradient saliency)[/bold]")

    if not scores:
        console.print("  No attribution scores computed.")
        return

    max_score = max(abs(s) for s in scores) if scores else 1.0

    parts: list[str] = []
    for tok, score in zip(tokens, scores):
        intensity = abs(score) / max_score if max_score > 0 else 0
        if intensity > 0.7:
            parts.append(f"[bold red]{tok}[/bold red]")
        elif intensity > 0.4:
            parts.append(f"[yellow]{tok}[/yellow]")
        elif intensity > 0.15:
            parts.append(f"[dim]{tok}[/dim]")
        else:
            parts.append(tok)

    console.print("  " + "".join(parts))

    # Also show ranked list
    ranked = sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True)
    console.print()
    for tok, score in ranked[:10]:
        bar_len = int(20 * abs(score) / max_score) if max_score > 0 else 0
        bar = "█" * bar_len
        console.print(f"  {tok:>15s}  [green]{bar}[/green]  {score:.4f}")

    console.print()


def render_attribution_heatmap(
    attribution: torch.Tensor,
    output_path: str = "attribution_heatmap.png",
) -> None:
    """Save a vision attribution heatmap to a file."""
    import matplotlib.pyplot as plt
    import numpy as np

    attr_np = attribution.detach().cpu().numpy()

    # Collapse channel dim if present
    if attr_np.ndim == 3:
        attr_np = attr_np.mean(axis=0)
    elif attr_np.ndim == 4:
        attr_np = attr_np[0].mean(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(np.abs(attr_np), cmap="hot", interpolation="bilinear")
    ax.set_title("Gradient Attribution")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    console.print(f"\n  Attribution heatmap saved to [bold]{output_path}[/bold]\n")


# ------------------------------------------------------------------
# Patch rendering
# ------------------------------------------------------------------


def render_patch(result: dict[str, Any]) -> None:
    """Print the result of a single activation patch."""
    console.print(f"\n[bold]Activation Patch at: {result['module']}[/bold]")
    console.print(f"  Normalised effect: [bold]{result['effect']:.4f}[/bold]")
    console.print()


# ------------------------------------------------------------------
# Activations rendering
# ------------------------------------------------------------------


def render_activations(cache: dict[str, torch.Tensor]) -> None:
    """Print a summary table of extracted activations."""
    console.print("\n[bold]Activations[/bold]")

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Module", style="cyan", no_wrap=True)
    table.add_column("Shape", style="dim")
    table.add_column("Norm", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for name, tensor in cache.items():
        t = tensor.float()
        table.add_row(
            name,
            str(tuple(tensor.shape)),
            f"{t.norm():.3f}",
            f"{t.mean():.4f}",
            f"{t.std():.4f}",
            f"{t.min():.4f}",
            f"{t.max():.4f}",
        )

    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Ablation rendering
# ------------------------------------------------------------------


def render_ablate(result: dict[str, Any]) -> None:
    """Print the result of an ablation."""
    method = result.get("method", "zero")
    console.print(f"\n[bold]Ablation ({method}) at: {result['module']}[/bold]")
    console.print(f"  Effect on output: [bold]{result['effect']:.4f}[/bold]")
    console.print()


# ------------------------------------------------------------------
# Attention rendering
# ------------------------------------------------------------------


def render_attention(
    attention_data: list[dict[str, Any]],
    tokens: list[str] | None,
    model_name: str,
) -> None:
    """Print attention summary per layer/head."""
    console.print(f"\n[bold]Attention Patterns: {model_name}[/bold]")

    if not attention_data:
        console.print("  No attention data captured.")
        return

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Layer", style="cyan")
    table.add_column("Head", style="cyan", justify="right")
    table.add_column("Top Attention", style="dim")
    table.add_column("Entropy", justify="right")

    for entry in attention_data:
        top_attn_parts = []
        for src, tgt, score in entry.get("top_pairs", [])[:3]:
            src_tok = tokens[src] if tokens and src < len(tokens) else str(src)
            tgt_tok = tokens[tgt] if tokens and tgt < len(tokens) else str(tgt)
            top_attn_parts.append(f"{src_tok}->{tgt_tok} ({score:.2f})")
        table.add_row(
            str(entry["layer"]),
            str(entry["head"]),
            ", ".join(top_attn_parts),
            f"{entry.get('entropy', 0.0):.2f}",
        )

    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Steering rendering
# ------------------------------------------------------------------


def render_steer(
    original_tokens: list[tuple[str, float]],
    steered_tokens: list[tuple[str, float]],
    module_name: str,
    scale: float,
) -> None:
    """Print side-by-side comparison of top tokens with and without steering."""
    console.print(f"\n[bold]Steering at: {module_name} (scale={scale})[/bold]")

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Original Token", style="cyan")
    table.add_column("Prob", justify="right")
    table.add_column("Steered Token", style="green")
    table.add_column("Prob", justify="right")

    n = max(len(original_tokens), len(steered_tokens))
    for i in range(min(n, 10)):
        orig_tok = original_tokens[i][0] if i < len(original_tokens) else ""
        orig_prob = f"{original_tokens[i][1]:.3f}" if i < len(original_tokens) else ""
        steer_tok = steered_tokens[i][0] if i < len(steered_tokens) else ""
        steer_prob = f"{steered_tokens[i][1]:.3f}" if i < len(steered_tokens) else ""
        table.add_row(str(i + 1), orig_tok, orig_prob, steer_tok, steer_prob)

    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Probe rendering
# ------------------------------------------------------------------


def render_probe(result: dict[str, Any]) -> None:
    """Print probe results — accuracy and top features."""
    console.print(f"\n[bold]Linear Probe at: {result['module']}[/bold]")

    eval_method = result.get("eval_method", "")
    if eval_method == "holdout":
        console.print(f"  Test accuracy (holdout 20%): [bold]{result['accuracy']:.3f}[/bold]")
        if result.get("cv_accuracy") is not None:
            console.print(f"  CV accuracy (train split):   {result['cv_accuracy']:.3f}")
        if result.get("train_accuracy") is not None:
            console.print(f"  Train accuracy:              {result['train_accuracy']:.3f}")
    elif eval_method == "cv_only":
        console.print(f"  CV accuracy: [bold]{result['accuracy']:.3f}[/bold]")
        console.print("  [dim](too few samples for holdout split)[/dim]")
    elif eval_method == "train_only":
        console.print(f"  Train accuracy: [bold]{result['accuracy']:.3f}[/bold]  [dim](no holdout — <10 samples)[/dim]")
    else:
        console.print(f"  Accuracy: [bold]{result['accuracy']:.3f}[/bold]")
        if result.get("train_accuracy") is not None:
            console.print(f"  Train accuracy: {result['train_accuracy']:.3f}")

    if result.get("top_features"):
        console.print("\n  Top features by weight magnitude:")
        for idx, weight in result["top_features"][:10]:
            bar_len = int(20 * abs(weight) / abs(result["top_features"][0][1])) if result["top_features"][0][1] != 0 else 0
            bar = "█" * bar_len
            console.print(f"    dim {idx:>5d}  [green]{bar}[/green]  {weight:.4f}")

    console.print()


# ------------------------------------------------------------------
# Diff rendering
# ------------------------------------------------------------------


def render_diff(
    results: list[dict[str, Any]],
    model_a_name: str,
    model_b_name: str,
) -> None:
    """Print per-module activation distance between two models."""
    console.print(f"\n[bold]Model Diff: {model_a_name} vs {model_b_name}[/bold]")

    if not results:
        console.print("  No differences computed.")
        return

    max_dist = max(r["distance"] for r in results) if results else 1.0
    bar_width = 30

    table = Table(show_header=False, show_lines=False, pad_edge=False, box=None)
    table.add_column("Module", style="cyan", no_wrap=True, min_width=35)
    table.add_column("Bar", no_wrap=True)
    table.add_column("Cosine Dist", justify="right", style="bold")

    for r in results:
        fill = int(bar_width * r["distance"] / max_dist) if max_dist > 0 else 0
        bar = "█" * fill
        table.add_row(r["module"], f"[green]{bar}[/green]", f"{r['distance']:.4f}")

    console.print(table)

    if results:
        best = results[0]
        console.print(f"\n  Most changed: [bold cyan]{best['module']}[/bold cyan] (distance: {best['distance']:.4f})")

    console.print()


# ------------------------------------------------------------------
# SAE features rendering
# ------------------------------------------------------------------


def render_features(result: dict[str, Any]) -> None:
    """Print SAE feature decomposition results."""
    console.print(f"\n[bold]SAE Features at: {result['module']}[/bold]")
    console.print(
        f"  Active features: [bold]{result['num_active_features']}[/bold] / {result['total_features']}  "
        f"| Sparsity: {result['sparsity']:.2%}  "
        f"| Reconstruction error: {result['reconstruction_error']:.4f}"
    )

    top = result.get("top_features", [])
    if not top:
        console.print("  No active features found.")
        console.print()
        return

    max_val = max(abs(v) for _, v in top) if top else 1.0

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Feature", style="cyan", justify="right")
    table.add_column("Activation", justify="right")
    table.add_column("Bar", no_wrap=True)

    bar_width = 25
    for rank, (idx, val) in enumerate(top, 1):
        fill = int(bar_width * abs(val) / max_val) if max_val > 0 else 0
        bar = "█" * fill
        table.add_row(str(rank), str(idx), f"{val:.4f}", f"[green]{bar}[/green]")

    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Direct Logit Attribution rendering
# ------------------------------------------------------------------


def render_decompose(result: dict[str, Any]) -> None:
    """Print residual stream decomposition."""
    components = result["components"]
    position = result["position"]

    console.print(f"\n[bold]Residual Stream Decomposition (position {position})[/bold]")

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Component", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Norm", justify="right")
    table.add_column("", min_width=20)

    max_norm = max((c["norm"] for c in components), default=1.0) or 1.0

    for c in components:
        bar_len = int(c["norm"] / max_norm * 15)
        bar = f"[green]{'█' * bar_len}[/green]"
        table.add_row(c["name"], c["type"], f"{c['norm']:.3f}", bar)

    console.print(table)

    if result.get("residual") is not None:
        console.print(f"  Final residual norm: {result['residual'].norm().item():.3f}")
    console.print()


def render_dla(result: dict[str, Any], *, top_k: int = 10) -> None:
    """Print DLA results — top contributors to the target logit."""
    target = result["target_token"]
    contributions = result["contributions"]
    head_contribs = result.get("head_contributions", [])

    console.print(f"\n[bold]Direct Logit Attribution → \"{target}\"[/bold]")
    console.print(f"  Total component logit sum: {result['total_logit']:.3f}\n")

    # Component-level table (attn + mlp per layer)
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Component", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Contribution", justify="right")
    table.add_column("", min_width=20)

    max_abs = max((abs(c["logit_contribution"]) for c in contributions), default=1.0) or 1.0

    for c in contributions[:top_k]:
        val = c["logit_contribution"]
        bar_len = int(abs(val) / max_abs * 15)
        if val >= 0:
            bar = f"[green]{'█' * bar_len}[/green]"
        else:
            bar = f"[red]{'█' * bar_len}[/red]"
        table.add_row(c["component"], c["type"], f"{val:+.4f}", bar)

    if len(contributions) > top_k:
        table.add_row("...", "", "", f"({len(contributions) - top_k} more)")

    console.print(table)

    # Per-head table
    if head_contribs:
        console.print(f"\n[bold]  Per-Head Breakdown (top {top_k})[/bold]")
        htable = Table(show_header=True, header_style="bold", show_lines=False)
        htable.add_column("Head", style="cyan")
        htable.add_column("Contribution", justify="right")
        htable.add_column("", min_width=20)

        max_abs_h = max((abs(c["logit_contribution"]) for c in head_contribs), default=1.0) or 1.0
        shown = head_contribs[:top_k] + head_contribs[-top_k:] if len(head_contribs) > 2 * top_k else head_contribs
        seen = set()
        for c in shown:
            key = c["component"]
            if key in seen:
                continue
            seen.add(key)
            val = c["logit_contribution"]
            bar_len = int(abs(val) / max_abs_h * 15)
            if val >= 0:
                bar = f"[green]{'█' * bar_len}[/green]"
            else:
                bar = f"[red]{'█' * bar_len}[/red]"
            htable.add_row(key, f"{val:+.4f}", bar)

        console.print(htable)

    console.print()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
