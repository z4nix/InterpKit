"""Matplotlib visualizations — publication-quality figures for mech interp results."""

from __future__ import annotations

import re
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

_INTERPKIT_RC = {
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
}


def _save_and_show(fig: plt.Figure, path: str | None, default_name: str) -> str:
    """Save figure to *path* (or *default_name*).

    Supports ``.png``, ``.svg``, and ``.pdf`` output — the format is
    detected from the file extension.
    """
    out = path or default_name
    ext = out.rsplit(".", 1)[-1].lower() if "." in out else "png"
    dpi = 300 if ext in ("svg", "pdf") else 150
    fig.savefig(out, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
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
    with plt.rc_context(_INTERPKIT_RC):
        attn = weights.detach().cpu().float().numpy()
        seq_len = attn.shape[0]

        fig, ax = plt.subplots(figsize=(max(4, seq_len * 0.6), max(4, seq_len * 0.6)))

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "interpkit", ["#1a1a2e", "#0f3460", "#e94560", "#ffdd57"]
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

    with plt.rc_context(_INTERPKIT_RC):
        layers = sorted(set(d["layer"] for d in attention_data))
        heads_per_layer = max(
            sum(1 for d in attention_data if d["layer"] == l) for l in layers
        )
        n_layers = len(layers)
        n_heads = min(heads_per_layer, 8)

        fig, axes = plt.subplots(
            n_layers, n_heads,
            figsize=(n_heads * 2.5, n_layers * 2.5),
            squeeze=False,
        )

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "interpkit", ["#1a1a2e", "#0f3460", "#e94560", "#ffdd57"]
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

    with plt.rc_context(_INTERPKIT_RC):
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
        max_eff = max(effects) if effects else 1
        ax.set_xlim(0, max(max_eff * 1.15, 0.001))
        ax.grid(axis="x", alpha=0.2)

        for bar, val in zip(bars, effects):
            ax.text(bar.get_width() + max(max_eff, 0.001) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7, color=_PALETTE["muted"])

        fig.tight_layout()
        return _save_and_show(fig, save_path, "causal_trace.png")


def plot_position_trace(
    result: dict[str, Any],
    save_path: str | None = None,
) -> str:
    """2D heatmap of position-aware causal tracing (Meng et al. style)."""
    effects = result["effects"]  # tensor (num_layers, seq_len)
    layer_names = result["layer_names"]
    tokens = result.get("tokens")

    if not isinstance(effects, torch.Tensor):
        effects = torch.tensor(effects)

    data = effects.detach().cpu().float().numpy()
    num_layers, seq_len = data.shape

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "interpkit_trace", ["#1a1a2e", "#0f3460", "#e94560", "#ffdd57"]
    )

    with plt.rc_context(_INTERPKIT_RC):
        fig_w = max(6, seq_len * 0.7)
        fig_h = max(4, num_layers * 0.4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=max(1.0, data.max()))

        ax.set_yticks(range(num_layers))
        short_names = []
        for ln in layer_names:
            m = re.search(r"\.(\d+)$", ln)
            short_names.append(f"L{m.group(1)}" if m else ln.split(".")[-1])
        ax.set_yticklabels(short_names, fontsize=7)

        if tokens and len(tokens) == seq_len:
            xlabels = [t.replace("\u0120", " ") for t in tokens]
        else:
            xlabels = [str(i) for i in range(seq_len)]
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")

        ax.set_xlabel("Token position", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)
        ax.set_title("Causal Trace — (Layer, Position)", fontsize=11, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label("Recovery effect", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        fig.tight_layout()
        return _save_and_show(fig, save_path, "position_trace.png")


# ── Logit lens heatmap ───────────────────────────────────────────


def plot_lens(
    predictions: list[dict[str, Any]],
    save_path: str | None = None,
    input_tokens: list[str] | None = None,
) -> str:
    """Logit lens heatmap.

    When predictions contain per-position data (``positions`` key), renders the
    classic 2D heatmap with layers on the y-axis and token positions on the
    x-axis, coloured by top-1 probability.  Falls back to a 1D column when
    only a single position is present per layer.
    """
    if not predictions:
        return ""

    has_multi_pos = (
        "positions" in predictions[0]
        and len(predictions[0]["positions"]) > 1
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "interpkit_lens", ["#1a1a2e", "#0f3460", "#28a745", "#ffdd57"]
    )

    with plt.rc_context(_INTERPKIT_RC):
        layers = [p["layer_name"] for p in predictions]

        if has_multi_pos:
            num_pos = len(predictions[0]["positions"])
            data = np.zeros((len(layers), num_pos))
            annotations = [[" "] * num_pos for _ in layers]

            for i, pred in enumerate(predictions):
                for pp in pred["positions"]:
                    j = pp["pos"]
                    if j < num_pos:
                        data[i, j] = pp["top1_prob"]
                        annotations[i][j] = pp["top1_token"]

            fig_w = max(6, num_pos * 0.9)
            fig_h = max(3, len(layers) * 0.45)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers, fontsize=7)

            if input_tokens is not None and len(input_tokens) == num_pos:
                xlabels = [t.replace("\u0120", " ") for t in input_tokens]
            else:
                xlabels = [str(i) for i in range(num_pos)]
            ax.set_xticks(range(num_pos))
            ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")

            if num_pos <= 20:
                for i in range(len(layers)):
                    for j in range(num_pos):
                        prob = data[i, j]
                        tok = annotations[i][j].strip()
                        if tok:
                            text_color = _PALETTE["bg"] if prob > 0.5 else _PALETTE["text"]
                            ax.text(j, i, tok, ha="center", va="center",
                                    fontsize=6, color=text_color)

            ax.set_title("Logit Lens — Top-1 per (Layer, Position)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Token position", fontsize=9)

        else:
            probs = [p["top1_prob"] for p in predictions]
            token_labels = [p["top1_token"] for p in predictions]

            fig, ax = plt.subplots(figsize=(6, max(3, len(layers) * 0.4)))

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

    with plt.rc_context(_INTERPKIT_RC):
        labels_orig = [t[0].strip() for t in original_tokens[:n]]
        probs_orig = [t[1] for t in original_tokens[:n]]
        labels_steer = [t[0].strip() for t in steered_tokens[:n]]
        probs_steer = [t[1] for t in steered_tokens[:n]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(3, n * 0.4)), sharey=False)

        ax1.barh(range(n), probs_orig, color="#0f3460", height=0.6)
        ax1.set_yticks(range(n))
        ax1.set_yticklabels(labels_orig, fontsize=9)
        ax1.set_xlabel("Probability", fontsize=9)
        ax1.set_title("Original", fontsize=11, fontweight="bold")
        ax1.invert_yaxis()
        ax1.set_xlim(0, max(probs_orig + probs_steer) * 1.2)
        ax1.grid(axis="x", alpha=0.2)

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

    with plt.rc_context(_INTERPKIT_RC):
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
        max_dist = max(distances) if distances else 1
        ax.set_xlim(0, max(max_dist * 1.15, 0.001))
        ax.grid(axis="x", alpha=0.2)

        for bar, val in zip(bars, distances):
            ax.text(bar.get_width() + max(max_dist, 0.001) * 0.02, bar.get_y() + bar.get_height() / 2,
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

    with plt.rc_context(_INTERPKIT_RC):
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


# ── Direct Logit Attribution ─────────────────────────────────────


def plot_dla(
    result: dict[str, Any],
    top_k: int = 10,
    save_path: str | None = None,
) -> str:
    """Horizontal bar chart of component contributions to the target logit."""
    contributions = result["contributions"]
    if not contributions:
        return ""

    shown = contributions[:top_k]

    with plt.rc_context(_INTERPKIT_RC):
        labels = [c["component"] for c in shown]
        values = [c["logit_contribution"] for c in shown]

        fig, ax = plt.subplots(figsize=(8, max(3, len(shown) * 0.35)))

        colors = [_PALETTE["accent"] if v < 0 else "#28a745" for v in values]
        y_pos = range(len(labels))
        ax.barh(y_pos, values, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Logit contribution", fontsize=10)
        ax.axvline(0, color=_PALETTE["muted"], linewidth=0.8)
        target_tok = result.get("target_token", "?")
        ax.set_title(
            f"Direct Logit Attribution → \"{target_tok}\"",
            fontsize=11, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.2)

        # Head breakdown subplot if available
        head_contribs = result.get("head_contributions", [])
        if head_contribs:
            head_shown = head_contribs[:top_k]
            fig2, ax2 = plt.subplots(figsize=(8, max(3, len(head_shown) * 0.3)))
            hlabels = [c["component"] for c in head_shown]
            hvalues = [c["logit_contribution"] for c in head_shown]
            hcolors = [_PALETTE["accent"] if v < 0 else "#28a745" for v in hvalues]
            ax2.barh(range(len(hlabels)), hvalues, color=hcolors, height=0.6)
            ax2.set_yticks(range(len(hlabels)))
            ax2.set_yticklabels(hlabels, fontsize=7)
            ax2.invert_yaxis()
            ax2.set_xlabel("Logit contribution", fontsize=10)
            ax2.axvline(0, color=_PALETTE["muted"], linewidth=0.8)
            ax2.set_title("Per-Head Contributions", fontsize=11, fontweight="bold")
            ax2.grid(axis="x", alpha=0.2)
            fig2.tight_layout()
            if save_path:
                head_path = save_path.rsplit(".", 1)
                head_save = f"{head_path[0]}_heads.{head_path[1]}" if len(head_path) == 2 else f"{save_path}_heads"
                _save_and_show(fig2, head_save, "dla_heads.png")
            else:
                _save_and_show(fig2, None, "dla_heads.png")

        fig.tight_layout()
        return _save_and_show(fig, save_path, "dla.png")
