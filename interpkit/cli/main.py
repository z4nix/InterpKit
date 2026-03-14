"""CLI entry point — Typer app with all interpkit commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="interpkit",
    help="Mech interp for any HuggingFace model.",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _load_model(model_name: str, device: str | None = None):
    from interpkit.core.model import load

    with console.status(f"Loading {model_name}..."):
        return load(model_name, device=device)


# ══════════════════════════════════════════════════════════════════
# help — rich overview panel
# ══════════════════════════════════════════════════════════════════


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Mech interp for any HuggingFace model."""
    if ctx.invoked_subcommand is not None:
        return

    logo = r"""
 ____ ____ ____ ____ ____ ____ ____ ____ ____ _________
||I |||n |||t |||e |||r |||p |||K |||i |||t |||       ||
||__|||__|||__|||__|||__|||__|||__|||__|||__|||_______||
|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/_______\|
"""
    console.print(f"[bold cyan]{logo}[/bold cyan]", highlight=False)

    table = Table(
        show_header=True, header_style="bold", show_lines=False,
        pad_edge=True, expand=True,
    )
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Example", style="dim")

    rows = [
        ("", "[bold]Core Operations[/bold]", ""),
        ("inspect", "Module tree with types, params, roles", "interpkit inspect gpt2"),
        ("patch", "Activation patching at a module", "interpkit patch gpt2 --clean '...' --corrupted '...' --at transformer.h.8.mlp"),
        ("trace", "Causal tracing — rank modules by effect", "interpkit trace gpt2 --clean '...' --corrupted '...'"),
        ("lens", "Logit lens — project layers to vocab", "interpkit lens gpt2 'The capital of France is'"),
        ("attribute", "Gradient saliency over inputs", "interpkit attribute gpt2 'The capital of France is'"),
        ("", "", ""),
        ("", "[bold]Analysis Operations[/bold]", ""),
        ("activations", "Extract raw activation tensors", "interpkit activations gpt2 '...' --at transformer.h.8"),
        ("ablate", "Zero/mean ablate a component", "interpkit ablate gpt2 '...' --at transformer.h.8.mlp"),
        ("attention", "Visualize attention patterns", "interpkit attention gpt2 '...' --layer 8"),
        ("steer", "Apply a steering vector", "interpkit steer gpt2 '...' --positive Love --negative Hate --at transformer.h.8"),
        ("probe", "Linear probe on activations", "interpkit probe gpt2 --at transformer.h.8 --data data.json"),
        ("diff", "Compare two models' activations", "interpkit diff gpt2 my-finetuned-gpt2 '...'"),
        ("", "", ""),
        ("", "[bold]Advanced[/bold]", ""),
        ("features", "SAE feature decomposition", "interpkit features gpt2 '...' --at transformer.h.8 --sae jbloom/..."),
    ]

    for cmd, desc, example in rows:
        table.add_row(cmd, desc, example)

    panel = Panel(
        table,
        title="[bold cyan]Commands[/bold cyan]",
        subtitle="[dim]Mech interp for any HuggingFace model.[/dim]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)

    save_hint = Text.assemble(
        ("  Tip: ", "bold"),
        ("Most commands accept ", ""),
        ("--save path.png", "bold green"),
        (" to export a matplotlib figure and ", ""),
        ("--html path.html", "bold green"),
        (" for interactive visualizations.\n", ""),
    )
    console.print(save_hint)
    console.print("  Run [bold cyan]interpkit <command> --help[/bold cyan] for detailed usage.\n")


# ══════════════════════════════════════════════════════════════════
# inspect
# ══════════════════════════════════════════════════════════════════


@app.command()
def inspect(
    model_name: str = typer.Argument(..., help="HuggingFace model ID (e.g. gpt2, microsoft/resnet-50)"),
    device: Optional[str] = typer.Option(None, help="Device (cpu, cuda, mps). Auto-detected if omitted."),
) -> None:
    """Print the model's module tree with types, param counts, and detected roles."""
    m = _load_model(model_name, device=device)
    m.inspect()


# ══════════════════════════════════════════════════════════════════
# patch
# ══════════════════════════════════════════════════════════════════


@app.command()
def patch(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    clean: str = typer.Option(..., "--clean", help="Clean input (text string or image path)"),
    corrupted: str = typer.Option(..., "--corrupted", help="Corrupted input (text string or image path)"),
    at: str = typer.Option(..., "--at", help="Module name to patch (e.g. transformer.h.8.mlp)"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Activation patching: swap one module's output from clean into corrupted run."""
    m = _load_model(model_name, device=device)
    m.patch(clean, corrupted, at=at)


# ══════════════════════════════════════════════════════════════════
# trace
# ══════════════════════════════════════════════════════════════════


@app.command()
def trace(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    clean: str = typer.Option(..., "--clean", help="Clean input"),
    corrupted: str = typer.Option(..., "--corrupted", help="Corrupted input"),
    top_k: int = typer.Option(20, "--top-k", help="Scan top-K modules by proxy score. 0 = scan all."),
    save: Optional[str] = typer.Option(None, "--save", help="Save bar chart to file (e.g. trace.png)"),
    html_path: Optional[str] = typer.Option(None, "--html", help="Save interactive HTML to file (e.g. trace.html)"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Causal tracing: rank modules by how much patching them restores clean output."""
    effective_top_k: int | None = top_k if top_k > 0 else None
    m = _load_model(model_name, device=device)
    m.trace(clean, corrupted, top_k=effective_top_k, save=save, html=html_path)


# ══════════════════════════════════════════════════════════════════
# lens
# ══════════════════════════════════════════════════════════════════


@app.command()
def lens(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    text: str = typer.Argument(..., help="Input text"),
    save: Optional[str] = typer.Option(None, "--save", help="Save heatmap to file (e.g. lens.png)"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Logit lens: project each layer's hidden state to vocabulary space."""
    m = _load_model(model_name, device=device)
    m.lens(text, save=save)


# ══════════════════════════════════════════════════════════════════
# attribute
# ══════════════════════════════════════════════════════════════════


@app.command()
def attribute(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text or image path"),
    target: Optional[int] = typer.Option(None, "--target", help="Target class/token index for attribution"),
    save: Optional[str] = typer.Option(None, "--save", help="Save figure to file (e.g. attribution.png)"),
    html_path: Optional[str] = typer.Option(None, "--html", help="Save interactive HTML to file (e.g. attribution.html)"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Gradient saliency over input tokens or pixels."""
    m = _load_model(model_name, device=device)
    m.attribute(input_data, target=target, save=save, html=html_path)


# ══════════════════════════════════════════════════════════════════
# activations
# ══════════════════════════════════════════════════════════════════


@app.command()
def activations(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text or image path"),
    at: str = typer.Option(..., "--at", help="Module name(s) to extract, comma-separated"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Extract and display activation statistics at named modules."""
    m = _load_model(model_name, device=device)
    modules = [s.strip() for s in at.split(",")]
    if len(modules) == 1:
        m.activations(input_data, at=modules[0])
    else:
        m.activations(input_data, at=modules)


# ══════════════════════════════════════════════════════════════════
# ablate
# ══════════════════════════════════════════════════════════════════


@app.command()
def ablate(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text or image path"),
    at: str = typer.Option(..., "--at", help="Module name to ablate"),
    method: str = typer.Option("zero", "--method", help="Ablation method: zero or mean"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Zero or mean ablate a module and measure the effect on output."""
    m = _load_model(model_name, device=device)
    m.ablate(input_data, at=at, method=method)


# ══════════════════════════════════════════════════════════════════
# attention
# ══════════════════════════════════════════════════════════════════


@app.command()
def attention(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text"),
    layer: Optional[int] = typer.Option(None, "--layer", help="Specific layer index"),
    head: Optional[int] = typer.Option(None, "--head", help="Specific head index"),
    save: Optional[str] = typer.Option(None, "--save", help="Save heatmap to file (e.g. attention.png)"),
    html_path: Optional[str] = typer.Option(None, "--html", help="Save interactive HTML to file (e.g. attention.html)"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Show attention patterns for transformer models."""
    m = _load_model(model_name, device=device)
    m.attention(input_data, layer=layer, head=head, save=save, html=html_path)


# ══════════════════════════════════════════════════════════════════
# steer
# ══════════════════════════════════════════════════════════════════


@app.command()
def steer(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text to steer"),
    positive: str = typer.Option(..., "--positive", help="Positive direction text"),
    negative: str = typer.Option(..., "--negative", help="Negative direction text"),
    at: str = typer.Option(..., "--at", help="Module name to apply steering at"),
    scale: float = typer.Option(2.0, "--scale", help="Steering vector scale factor"),
    save: Optional[str] = typer.Option(None, "--save", help="Save comparison chart to file"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Extract a steering vector and apply it during inference."""
    m = _load_model(model_name, device=device)
    vector = m.steer_vector(positive, negative, at=at)
    m.steer(input_data, vector=vector, at=at, scale=scale, save=save)


# ══════════════════════════════════════════════════════════════════
# probe
# ══════════════════════════════════════════════════════════════════


@app.command()
def probe(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    at: str = typer.Option(..., "--at", help="Module name to probe"),
    data: str = typer.Option(..., "--data", help="JSON file with {texts: [...], labels: [...]}"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Train a linear probe on activations to test linear separability."""
    import json
    from pathlib import Path

    probe_data = json.loads(Path(data).read_text())
    m = _load_model(model_name, device=device)
    m.probe(texts=probe_data["texts"], labels=probe_data["labels"], at=at)


# ══════════════════════════════════════════════════════════════════
# diff
# ══════════════════════════════════════════════════════════════════


@app.command()
def diff(
    model_a_name: str = typer.Argument(..., help="First model (e.g. gpt2)"),
    model_b_name: str = typer.Argument(..., help="Second model (e.g. my-finetuned-gpt2)"),
    input_data: str = typer.Argument(..., help="Input text to compare on"),
    save: Optional[str] = typer.Option(None, "--save", help="Save bar chart to file"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Compare activations between two models on the same input."""
    import interpkit

    m_a = _load_model(model_a_name, device=device)
    m_b = _load_model(model_b_name, device=device)
    interpkit.diff(m_a, m_b, input_data, save=save)


# ══════════════════════════════════════════════════════════════════
# features (SAE)
# ══════════════════════════════════════════════════════════════════


@app.command()
def features(
    model_name: str = typer.Argument(..., help="HuggingFace model ID (e.g. gpt2)"),
    input_data: str = typer.Argument(..., help="Input text"),
    at: str = typer.Option(..., "--at", help="Module name to decompose (e.g. transformer.h.8)"),
    sae: str = typer.Option(..., "--sae", help="HuggingFace repo ID for the SAE weights"),
    top_k: int = typer.Option(20, "--top-k", help="Number of top features to display"),
    device: Optional[str] = typer.Option(None, help="Device"),
) -> None:
    """Decompose activations through a Sparse Autoencoder into interpretable features."""
    m = _load_model(model_name, device=device)
    m.features(input_data, at=at, sae=sae, top_k=top_k)


if __name__ == "__main__":
    app()
