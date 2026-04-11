"""Terminal rendering — rich tables, trees, unicode bar charts, heatmap export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich_gradient import Rule as GradientRule

from interpkit.core.theme import ACCENT, BRAND_COLORS, ROLE_PILL

if TYPE_CHECKING:
    from interpkit.core.discovery import ModelArchInfo

console = Console()

_BAR_WIDTH = 24
_TABLE_BOX = box.SIMPLE_HEAD


# ── Helpers ──────────────────────────────────────────────────────


def _section(title: str, subtitle: str = "") -> None:
    """Print a styled section header using a gradient Rule."""
    label = title
    if subtitle:
        label += f"  [dim]{subtitle}[/dim]"
    console.print()
    console.print(GradientRule(label, colors=BRAND_COLORS, align="left"))


def _bar(
    value: float,
    max_val: float,
    *,
    width: int = _BAR_WIDTH,
    positive: bool | None = None,
) -> str:
    """Render a smooth unicode bar with half-block precision."""
    if max_val <= 0:
        return ""
    ratio = min(abs(value) / max_val, 1.0)
    full = int(ratio * width)
    frac = ratio * width - full

    blocks = "\u2588" * full
    if frac >= 0.5:
        blocks += "\u258c"

    if not blocks:
        return ""

    if positive is None:
        positive = value >= 0

    color = "green" if positive else "red"
    return f"[{color}]{blocks}[/{color}]"


def _role_tag(role: str | None) -> str:
    """Format a module role as a pill-style tag with background."""
    if not role:
        return ""
    label, style = ROLE_PILL.get(role, (role, "dim"))
    return f"[{style}] {label} [/{style}]"


def _callout(label: str, value: str, style: str = ACCENT) -> None:
    """Print a highlighted callout line with a marker."""
    console.print(f"  [{style}]\u25b8[/{style}] {label}: [bold]{value}[/bold]")


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ------------------------------------------------------------------
# Inspect rendering
# ------------------------------------------------------------------


def render_inspect(arch_info: ModelArchInfo, nn_model: torch.nn.Module | None = None) -> None:
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

    _section("Model Architecture")
    details = " \u00b7 ".join(header_parts)
    console.print(f"  [dim]{details}[/dim]")

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX, pad_edge=False)
    table.add_column("Module", style=ACCENT, no_wrap=True)
    table.add_column("Type", style="dim")
    table.add_column("Params", justify="right")
    table.add_column("Output Shape", style="dim")
    table.add_column("Role")

    for m in arch_info.modules:
        tag = _role_tag(m.role)
        shape_str = str(m.output_shape) if m.output_shape else ""
        param_str = _format_params(m.param_count) if m.param_count > 0 else ""
        table.add_row(m.name, m.type_name, param_str, shape_str, tag)

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
    if top_k is not None:
        subtitle = f"top {len(results)} of {total_modules} modules"
    else:
        subtitle = f"{total_modules} modules"

    _section(f"Causal Trace \u2014 {model_name}", subtitle)

    if not results:
        console.print("  [dim]No significant causal effects found.[/dim]")
        return

    max_effect = max(r["effect"] for r in results) if results else 1.0

    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    table.add_column("Module", style=ACCENT, no_wrap=True, min_width=35)
    table.add_column("Role", min_width=8)
    table.add_column("Bar", no_wrap=True, min_width=_BAR_WIDTH + 2)
    table.add_column("Effect", justify="right", style="bold")

    for r in results:
        bar = _bar(r["effect"], max_effect)
        role = _role_tag(r.get("role"))
        table.add_row(r["module"], role, bar, f"{r['effect']:.3f}")

    console.print(table)

    if results:
        best = results[0]
        console.print()
        console.print(Panel(
            f"[bold {ACCENT}]{best['module']}[/bold {ACCENT}]  effect [bold]{best['effect']:.3f}[/bold]",
            title="[bold]Top component[/bold]",
            border_style=ACCENT,
            padding=(0, 2),
            expand=False,
        ))

    if top_k is not None and len(results) < total_modules:
        console.print(f"  [dim]Run with --top-k 0 to scan all {total_modules} modules.[/dim]")
    console.print()


def render_position_trace(result: dict[str, Any]) -> None:
    """Print a summary table for position-aware causal tracing."""
    effects = result["effects"]
    layer_names = result["layer_names"]
    tokens = result.get("tokens")

    if not isinstance(effects, torch.Tensor):
        effects = torch.tensor(effects)

    _section("Position-Aware Causal Trace")

    num_layers, seq_len = effects.shape

    flat = effects.view(-1)
    top_k = min(10, flat.numel())
    top_vals, top_idxs = flat.topk(top_k)

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Layer", style=ACCENT)
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
        f"\n  [dim]Heatmap: {num_layers} layers \u00d7 {seq_len} positions. "
        f"Use --save to export the full heatmap.[/dim]\n"
    )


# ------------------------------------------------------------------
# Logit lens rendering
# ------------------------------------------------------------------


def render_lens(
    predictions: list[dict[str, Any]],
    model_name: str,
) -> None:
    """Print logit lens top predictions per layer."""
    _section(f"Logit Lens \u2014 {model_name}")

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Layer", style=ACCENT)
    table.add_column("Top-1 Token", style="bold")
    table.add_column("Prob", justify="right")
    table.add_column("", no_wrap=True, min_width=12)
    table.add_column("Top-5 Tokens", style="dim")

    for pred in predictions:
        prob = pred["top1_prob"]
        prob_bar = _bar(prob, 1.0, width=10)
        top5_str = ", ".join(
            f"{tok} ({p:.2f})" for tok, p in zip(pred["top5_tokens"], pred["top5_probs"])
        )
        table.add_row(
            pred["layer_name"],
            pred["top1_token"],
            f"{prob:.3f}",
            prob_bar,
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
    _section("Attribution", "gradient saliency")

    if not scores:
        console.print("  [dim]No attribution scores computed.[/dim]")
        return

    max_score = max(abs(s) for s in scores) if scores else 1.0

    text = Text("  ")
    for tok, score in zip(tokens, scores):
        intensity = abs(score) / max_score if max_score > 0 else 0
        if intensity > 0.7:
            text.append(tok, style="bold red")
        elif intensity > 0.4:
            text.append(tok, style="yellow")
        elif intensity > 0.15:
            text.append(tok, style="dim")
        else:
            text.append(tok)

    console.print(text)
    console.print()

    ranked = sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True)
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    table.add_column("Token", justify="right", style="bold", min_width=15)
    table.add_column("Bar", no_wrap=True, min_width=_BAR_WIDTH + 2)
    table.add_column("Score", justify="right", style="dim")

    for tok, score in ranked[:10]:
        bar = _bar(score, max_score, positive=score >= 0)
        table.add_row(tok, bar, f"{score:.4f}")

    console.print(table)
    console.print()


def render_attribution_heatmap(
    attribution: torch.Tensor,
    output_path: str = "attribution_heatmap.png",
) -> None:
    """Save a vision attribution heatmap to a file."""
    import matplotlib.pyplot as plt
    import numpy as np

    attr_np = attribution.detach().cpu().numpy()

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

    console.print(f"\n  Heatmap saved to [bold]{output_path}[/bold]\n")


# ------------------------------------------------------------------
# Patch rendering
# ------------------------------------------------------------------


def render_patch(result: dict[str, Any]) -> None:
    """Print the result of a single activation patch."""
    _section(f"Activation Patch \u2014 {result['module']}")

    effect = result["effect"]
    bar = _bar(effect, max(abs(effect), 0.001), width=16)
    console.print(f"  Normalised effect: [bold]{effect:.4f}[/bold]  {bar}")
    console.print()


# ------------------------------------------------------------------
# Activations rendering
# ------------------------------------------------------------------


def render_activations(cache: dict[str, torch.Tensor]) -> None:
    """Print a summary table of extracted activations."""
    _section("Activations")

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Module", style=ACCENT, no_wrap=True)
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
    _section(f"Ablation ({method}) \u2014 {result['module']}")

    effect = result["effect"]
    bar = _bar(effect, max(abs(effect), 0.001), width=16)
    console.print(f"  Effect on output: [bold]{effect:.4f}[/bold]  {bar}")
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
    _section(f"Attention Patterns \u2014 {model_name}")

    if not attention_data:
        console.print("  [dim]No attention data captured.[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Layer", style=ACCENT)
    table.add_column("Head", style=ACCENT, justify="right")
    table.add_column("Top Attention Pairs")
    table.add_column("Entropy", justify="right", style="dim")

    for entry in attention_data:
        top_attn_parts = []
        for src, tgt, score in entry.get("top_pairs", [])[:3]:
            src_tok = tokens[src] if tokens and src < len(tokens) else str(src)
            tgt_tok = tokens[tgt] if tokens and tgt < len(tokens) else str(tgt)
            top_attn_parts.append(f"[bold]{src_tok}[/bold]\u2192{tgt_tok} [dim]({score:.2f})[/dim]")
        table.add_row(
            str(entry["layer"]),
            str(entry["head"]),
            "  ".join(top_attn_parts),
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
    _section(f"Steering \u2014 {module_name}", f"scale={scale}")

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Original Token", style=ACCENT)
    table.add_column("Prob", justify="right")
    table.add_column("", justify="center", style="dim", width=3)
    table.add_column("Steered Token", style="green")
    table.add_column("Prob", justify="right")

    n = max(len(original_tokens), len(steered_tokens))
    for i in range(min(n, 10)):
        orig_tok = original_tokens[i][0] if i < len(original_tokens) else ""
        orig_prob = f"{original_tokens[i][1]:.3f}" if i < len(original_tokens) else ""
        steer_tok = steered_tokens[i][0] if i < len(steered_tokens) else ""
        steer_prob = f"{steered_tokens[i][1]:.3f}" if i < len(steered_tokens) else ""
        changed = orig_tok != steer_tok
        arrow = "\u2192" if changed else "\u00b7"
        steer_style = "[bold green]" if changed else ""
        steer_end = "[/bold green]" if changed else ""
        table.add_row(str(i + 1), orig_tok, orig_prob, arrow, f"{steer_style}{steer_tok}{steer_end}", steer_prob)

    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Probe rendering
# ------------------------------------------------------------------


def render_probe(result: dict[str, Any]) -> None:
    """Print probe results — accuracy and top features."""
    _section(f"Linear Probe \u2014 {result['module']}")

    eval_method = result.get("eval_method", "")
    accuracy = result["accuracy"]

    if accuracy >= 0.9:
        acc_style = "bold green"
    elif accuracy >= 0.7:
        acc_style = "bold yellow"
    else:
        acc_style = "bold red"

    if eval_method == "holdout":
        console.print(f"  Test accuracy (holdout 20%): [{acc_style}]{accuracy:.3f}[/{acc_style}]")
        if result.get("cv_accuracy") is not None:
            console.print(f"  CV accuracy (train split):   {result['cv_accuracy']:.3f}")
        if result.get("train_accuracy") is not None:
            console.print(f"  Train accuracy:              {result['train_accuracy']:.3f}")
    elif eval_method == "cv_only":
        console.print(f"  CV accuracy: [{acc_style}]{accuracy:.3f}[/{acc_style}]")
        console.print("  [dim](too few samples for holdout split)[/dim]")
    elif eval_method == "train_only":
        console.print(f"  Train accuracy: [{acc_style}]{accuracy:.3f}[/{acc_style}]  [dim](no holdout \u2014 <10 samples)[/dim]")
    else:
        console.print(f"  Accuracy: [{acc_style}]{accuracy:.3f}[/{acc_style}]")
        if result.get("train_accuracy") is not None:
            console.print(f"  Train accuracy: {result['train_accuracy']:.3f}")

    if result.get("top_features"):
        console.print()
        console.print("  [bold]Top features by weight magnitude:[/bold]")
        max_weight = abs(result["top_features"][0][1]) if result["top_features"][0][1] != 0 else 1.0
        for idx, weight in result["top_features"][:10]:
            bar = _bar(weight, max_weight, width=16)
            console.print(f"    dim [{ACCENT}]{idx:>5d}[/{ACCENT}]  {bar}  {weight:.4f}")

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
    _section(f"Model Diff \u2014 {model_a_name} vs {model_b_name}")

    if not results:
        console.print("  [dim]No differences computed.[/dim]")
        return

    max_dist = max(r["distance"] for r in results) if results else 1.0

    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    table.add_column("Module", style=ACCENT, no_wrap=True, min_width=35)
    table.add_column("Bar", no_wrap=True, min_width=_BAR_WIDTH + 2)
    table.add_column("Cosine Dist", justify="right", style="bold")

    for r in results:
        bar = _bar(r["distance"], max_dist)
        table.add_row(r["module"], bar, f"{r['distance']:.4f}")

    console.print(table)

    if results:
        best = results[0]
        console.print()
        console.print(Panel(
            f"[bold {ACCENT}]{best['module']}[/bold {ACCENT}]  distance [bold]{best['distance']:.4f}[/bold]",
            title="[bold]Most changed[/bold]",
            border_style=ACCENT,
            padding=(0, 2),
            expand=False,
        ))

    console.print()


# ------------------------------------------------------------------
# SAE features rendering
# ------------------------------------------------------------------


def render_features(result: dict[str, Any]) -> None:
    """Print SAE feature decomposition results."""
    _section(f"SAE Features \u2014 {result['module']}")
    console.print(
        f"  Active: [bold]{result['num_active_features']}[/bold] / {result['total_features']}  "
        f"[dim]\u2502[/dim]  Sparsity: [bold]{result['sparsity']:.2%}[/bold]  "
        f"[dim]\u2502[/dim]  Recon error: {result['reconstruction_error']:.4f}"
    )

    top = result.get("top_features", [])
    if not top:
        console.print("  [dim]No active features found.[/dim]")
        console.print()
        return

    max_val = max(abs(v) for _, v in top) if top else 1.0
    console.print()

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Feature", style=ACCENT, justify="right")
    table.add_column("Activation", justify="right")
    table.add_column("", no_wrap=True, min_width=_BAR_WIDTH + 2)

    for rank, (idx, val) in enumerate(top, 1):
        bar = _bar(val, max_val)
        table.add_row(str(rank), str(idx), f"{val:.4f}", bar)

    console.print(table)

    top_feat = top[0]
    console.print(Panel(
        f"Feature [bold {ACCENT}]{top_feat[0]}[/bold {ACCENT}]  activation [bold]{top_feat[1]:.4f}[/bold]",
        title="[bold]Top feature[/bold]",
        border_style=ACCENT,
        padding=(0, 2),
        expand=False,
    ))
    console.print()


def render_contrastive_features(result: dict[str, Any]) -> None:
    """Print contrastive SAE feature analysis results."""
    _section(f"Contrastive SAE Features \u2014 {result['module']}")
    console.print(
        f"  Positive: [green]{result['num_positive']}[/green]  [dim]\u2502[/dim]  "
        f"Negative: [red]{result['num_negative']}[/red]  [dim]\u2502[/dim]  "
        f"Total features: {result['total_features']}"
    )

    top = result.get("top_differential_features", [])
    if not top:
        console.print("  [dim]No differential features found.[/dim]")
        console.print()
        return

    max_diff = max(abs(f["diff"]) for f in top) if top else 1.0
    console.print()

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Feature", style=ACCENT, justify="right")
    table.add_column("Pos Mean", justify="right")
    table.add_column("Neg Mean", justify="right")
    table.add_column("Diff", justify="right", style="bold")
    table.add_column("", no_wrap=True, min_width=_BAR_WIDTH + 2)

    for rank, feat in enumerate(top, 1):
        diff = feat["diff"]
        bar = _bar(diff, max_diff, positive=diff > 0)
        sign = "+" if diff > 0 else ""
        table.add_row(
            str(rank),
            str(feat["feature_idx"]),
            f"{feat['positive_mean']:.4f}",
            f"{feat['negative_mean']:.4f}",
            f"{sign}{diff:.4f}",
            bar,
        )

    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Direct Logit Attribution rendering
# ------------------------------------------------------------------


def render_decompose(result: dict[str, Any]) -> None:
    """Print residual stream decomposition."""
    components = result["components"]
    position = result["position"]

    _section("Residual Stream Decomposition", f"position {position}")

    max_norm = max((c["norm"] for c in components), default=1.0) or 1.0

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Component", style=ACCENT)
    table.add_column("Type", style="dim")
    table.add_column("Norm", justify="right")
    table.add_column("", min_width=_BAR_WIDTH + 2, no_wrap=True)

    for c in components:
        bar = _bar(c["norm"], max_norm)
        table.add_row(c["name"], c["type"], f"{c['norm']:.3f}", bar)

    console.print(table)

    if result.get("residual") is not None:
        console.print(f"  [dim]Final residual norm: {result['residual'].norm().item():.3f}[/dim]")
    console.print()


def render_dla(result: dict[str, Any], *, top_k: int = 10) -> None:
    """Print DLA results — top contributors to the target logit."""
    target = result["target_token"]
    contributions = result["contributions"]
    head_contribs = result.get("head_contributions", [])

    _section("Direct Logit Attribution")

    console.print(Panel(
        f'Target token: [bold]"{target}"[/bold]  |  Total logit sum: [bold]{result["total_logit"]:.3f}[/bold]',
        border_style=ACCENT,
        padding=(0, 2),
        expand=False,
    ))

    max_abs = max((abs(c["logit_contribution"]) for c in contributions), default=1.0) or 1.0

    table = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
    table.add_column("Component", style=ACCENT)
    table.add_column("Type", style="dim")
    table.add_column("Contribution", justify="right")
    table.add_column("", min_width=_BAR_WIDTH + 2, no_wrap=True)

    for c in contributions[:top_k]:
        val = c["logit_contribution"]
        bar = _bar(val, max_abs, positive=val >= 0)
        table.add_row(c["component"], c["type"], f"{val:+.4f}", bar)

    if len(contributions) > top_k:
        table.add_row("[dim]...[/dim]", "", "", f"[dim]({len(contributions) - top_k} more)[/dim]")

    console.print(table)

    if head_contribs:
        console.print()
        console.print(f"  [bold]Per-Head Breakdown[/bold]  [dim](top {top_k})[/dim]")

        max_abs_h = max((abs(c["logit_contribution"]) for c in head_contribs), default=1.0) or 1.0
        shown = head_contribs[:top_k] + head_contribs[-top_k:] if len(head_contribs) > 2 * top_k else head_contribs
        seen: set[str] = set()

        htable = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
        htable.add_column("Head", style=ACCENT)
        htable.add_column("Contribution", justify="right")
        htable.add_column("", min_width=_BAR_WIDTH + 2, no_wrap=True)

        for c in shown:
            key = c["component"]
            if key in seen:
                continue
            seen.add(key)
            val = c["logit_contribution"]
            bar = _bar(val, max_abs_h, positive=val >= 0)
            htable.add_row(key, f"{val:+.4f}", bar)

        console.print(htable)

    feat_info = result.get("feature_contributions")
    if feat_info and feat_info.get("features"):
        feats = feat_info["features"]
        sae_at = feat_info.get("sae_at", "?")
        n_active = feat_info.get("num_active", 0)
        n_total = feat_info.get("total_features", 0)

        console.print()
        console.print(
            f"  [bold]Feature-Level Breakdown[/bold]  [dim]({sae_at} via SAE — "
            f"{n_active}/{n_total} active)[/dim]"
        )

        max_abs_f = max((abs(f["logit_contribution"]) for f in feats), default=1.0) or 1.0

        ftable = Table(show_header=True, header_style="bold", box=_TABLE_BOX)
        ftable.add_column("Feature", style=ACCENT, justify="right")
        ftable.add_column("Activation", justify="right")
        ftable.add_column("Logit Contrib", justify="right")
        ftable.add_column("", min_width=_BAR_WIDTH + 2, no_wrap=True)

        for f in feats:
            val = f["logit_contribution"]
            bar = _bar(val, max_abs_f, positive=val >= 0)
            ftable.add_row(
                str(f["feature_idx"]),
                f"{f['activation']:.4f}",
                f"{val:+.4f}",
                bar,
            )

        console.print(ftable)

    console.print()
