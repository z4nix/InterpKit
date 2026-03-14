"""Matplotlib visualizations — publication-quality figures for mech interp results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rich.console import Console

console = Console()

# ── Shared style ─────────────────────────────────────────────────

_PALETTE = {
    "bg": "#1a1a2e",
    "surface": "#16213e",
    "primary": "#0f3460",
    "accent": "#e94560",
    "text": "#eaeaea",
    "muted": "#888888",
    "grid": "#2a2a4a",
}

plt.rcParams.update({
    "figure.facecolor": _PALETTE["bg"],
    "axes.facecolor": _PALETTE["surface"],
    "axes.edgecolor": _PALETTE["grid"],
    "axes.labelcolor": _PALETTE["text"],
    "text.color": _PALETTE["text"],
    "xtick.color": _PALETTE["muted"],
    "ytick.color": _PALETTE["muted"],
    "grid.color": _PALETTE["grid"],
    "grid.alpha": 0.3,
    "font.family": "monospace",
    "font.size": 10,
})


def _save_and_show(fig: plt.Figure, path: str | None, default_name: str) -> str:
    out = path or default_name
    fig.savefig(out, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    console.print(f"  Saved to [bold]{out}[/bold]")
    return out


# ── Attention heatmap ────────────────────────────────────────────


def plot_attention(
    weights: torch.Tensor,
    tokens: list[str] | None = None,
    layer: int = 0,
    head: int = 0,
    save_path: str | None = None,
) -> str:
    """Plot a single attention head as a heatmap.

    weights: (seq_len, seq_len) attention matrix
    """
    attn = weights.detach().cpu().float().numpy()
    seq_len = attn.shape[0]

    fig, ax = plt.subplots(figsize=(max(4, seq_len * 0.6), max(4, seq_len * 0.6)))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mechkit", ["#1a1a2e", "#0f3460", "#e94560", "#ffdd57"]
    )
    im = ax.imshow(attn, cmap=cmap, aspect="equal", vmin=0, vmax=1)

    if tokens:
        labels = [t[:12] for t in tokens[:seq_len]]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Key (attends to)", fontsize=9)
    ax.set_ylabel("Query (from)", fontsize=9)
    ax.set_title(f"Attention — Layer {layer}, Head {head}", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Attention weight", fontsize=9)

    fig.tight_layout()
    default = f"attention_L{layer}_H{head}.png"
    return _save_and_show(fig, save_path, default)


def plot_attention_multi(
    attention_data: list[dict[str, Any]],
    tokens: list[str] | None = None,
    save_path: str | None = None,
) -> str:
    """Plot a grid of attention heads."""
    if not attention_data:
        return ""

    layers = sorted(set(d["layer"] for d in attention_data))
    heads_per_layer = max(
        sum(1 for d in attention_data if d["layer"] == l) for l in layers
    )
    n_layers = len(layers)
    n_heads = min(heads_per_layer, 8)  # cap grid width

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(n_heads * 2.5, n_layers * 2.5),
        squeeze=False,
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mechkit", ["#1a1a2e", "#0f3460", "#e94560", "#ffdd57"]
    )

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    layer_to_idx = {l: i for i, l in enumerate(layers)}
    head_counts: dict[int, int] = {}

    for entry in attention_data:
        l = entry["layer"]
        row = layer_to_idx[l]
        col = head_counts.get(l, 0)
        head_counts[l] = col + 1

        if col >= n_heads:
            continue

        ax = axes[row][col]
        attn = entry["weights"].detach().cpu().float().numpy()
        ax.imshow(attn, cmap=cmap, aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"L{l} H{entry['head']}", fontsize=7, pad=2)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_color(_PALETTE["grid"])

    fig.suptitle("Attention Patterns", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_and_show(fig, save_path, "attention_grid.png")


# ── Causal trace bar chart ───────────────────────────────────────


def plot_trace(
    results: list[dict[str, Any]],
    model_name: str = "",
    save_path: str | None = None,
) -> str:
    """Horizontal bar chart of causal tracing results."""
    if not results:
        return ""

    top = results[:25]
    modules = [r["module"].split(".")[-2:] for r in reversed(top)]
    labels = [".".join(m) for m in modules]
    effects = [r["effect"] for r in reversed(top)]

    fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.35)))

    colors = [
        _PALETTE["accent"] if e == max(effects) else "#0f3460"
        for e in effects
    ]
    bars = ax.barh(range(len(labels)), effects, color=colors, height=0.7, edgecolor="none")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Patching Effect", fontsize=10)
    ax.set_title(f"Causal Trace{f': {model_name}' if model_name else ''}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(effects) * 1.15 if effects else 1)
    ax.grid(axis="x", alpha=0.2)

    for bar, val in zip(bars, effects):
        ax.text(bar.get_width() + max(effects) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7, color=_PALETTE["muted"])

    fig.tight_layout()
    return _save_and_show(fig, save_path, "causal_trace.png")


# ── Logit lens heatmap ───────────────────────────────────────────


def plot_lens(
    predictions: list[dict[str, Any]],
    save_path: str | None = None,
) -> str:
    """Heatmap: layers on y-axis, top-1 token confidence as color, token text annotated."""
    if not predictions:
        return ""

    layers = [p["layer_name"] for p in predictions]
    probs = [p["top1_prob"] for p in predictions]
    token_labels = [p["top1_token"] for p in predictions]

    fig, ax = plt.subplots(figsize=(6, max(3, len(layers) * 0.4)))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mechkit_lens", ["#1a1a2e", "#0f3460", "#28a745", "#ffdd57"]
    )

    # Single-column heatmap
    data = np.array(probs).reshape(-1, 1)
    im = ax.imshow(data, cmap=cmap, aspect=0.3, vmin=0, vmax=1)

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=8)
    ax.set_xticks([])

    for i, (tok, prob) in enumerate(zip(token_labels, probs)):
        text_color = _PALETTE["bg"] if prob > 0.5 else _PALETTE["text"]
        ax.text(0, i, f" {tok} ({prob:.2f})", ha="center", va="center",
                fontsize=8, fontweight="bold", color=text_color)

    ax.set_title("Logit Lens — Top-1 per Layer", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Probability", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return _save_and_show(fig, save_path, "logit_lens.png")


# ── Steering comparison ─────────────────────────────────────────


def plot_steer(
    original_tokens: list[tuple[str, float]],
    steered_tokens: list[tuple[str, float]],
    module_name: str = "",
    scale: float = 1.0,
    save_path: str | None = None,
) -> str:
    """Grouped bar chart comparing original vs steered top token probabilities."""
    n = min(10, len(original_tokens), len(steered_tokens))
    if n == 0:
        return ""

    labels_orig = [t[0].strip() for t in original_tokens[:n]]
    probs_orig = [t[1] for t in original_tokens[:n]]
    labels_steer = [t[0].strip() for t in steered_tokens[:n]]
    probs_steer = [t[1] for t in steered_tokens[:n]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(3, n * 0.4)), sharey=False)

    # Original
    ax1.barh(range(n), probs_orig, color="#0f3460", height=0.6)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels_orig, fontsize=9)
    ax1.set_xlabel("Probability", fontsize=9)
    ax1.set_title("Original", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(probs_orig + probs_steer) * 1.2)
    ax1.grid(axis="x", alpha=0.2)

    # Steered
    ax2.barh(range(n), probs_steer, color=_PALETTE["accent"], height=0.6)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels_steer, fontsize=9)
    ax2.set_xlabel("Probability", fontsize=9)
    ax2.set_title(f"Steered (scale={scale})", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(probs_orig + probs_steer) * 1.2)
    ax2.grid(axis="x", alpha=0.2)

    fig.suptitle(f"Steering at {module_name}", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save_and_show(fig, save_path, "steering.png")


# ── Diff bar chart ───────────────────────────────────────────────


def plot_diff(
    results: list[dict[str, Any]],
    model_a_name: str = "A",
    model_b_name: str = "B",
    save_path: str | None = None,
) -> str:
    """Horizontal bar chart of per-layer activation distance between two models."""
    if not results:
        return ""

    top = results[:25]
    modules = [r["module"].split(".")[-2:] for r in reversed(top)]
    labels = [".".join(m) for m in modules]
    distances = [r["distance"] for r in reversed(top)]

    fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.35)))

    colors = [
        _PALETTE["accent"] if d == max(distances) else "#0f3460"
        for d in distances
    ]
    bars = ax.barh(range(len(labels)), distances, color=colors, height=0.7)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cosine Distance", fontsize=10)
    ax.set_title(f"Model Diff: {model_a_name} vs {model_b_name}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(distances) * 1.15 if distances else 1)
    ax.grid(axis="x", alpha=0.2)

    for bar, val in zip(bars, distances):
        ax.text(bar.get_width() + max(distances) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7, color=_PALETTE["muted"])

    try:
        fig.tight_layout()
    except ValueError:
        pass
    return _save_and_show(fig, save_path, "model_diff.png")


# ── Attribution (text) bar chart ─────────────────────────────────


def plot_attribution(
    tokens: list[str],
    scores: list[float],
    save_path: str | None = None,
) -> str:
    """Bar chart of token attribution scores."""
    if not scores:
        return ""

    fig, ax = plt.subplots(figsize=(max(4, len(tokens) * 0.6), 4))

    max_score = max(abs(s) for s in scores)
    norm_scores = [s / max_score if max_score > 0 else 0 for s in scores]
    colors = [_PALETTE["accent"] if ns > 0.5 else "#0f3460" for ns in [abs(n) for n in norm_scores]]

    bars = ax.bar(range(len(tokens)), [abs(s) for s in scores], color=colors, width=0.7)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Attribution Score", fontsize=10)
    ax.set_title("Gradient Attribution", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    return _save_and_show(fig, save_path, "attribution.png")
