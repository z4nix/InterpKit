"""CLI entry point — Typer app with all interpkit commands."""

from __future__ import annotations

import json as _json
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

_output_format: str = "rich"


def _json_dump(result: dict) -> None:
    """Pretty-print a result dict as JSON, converting tensors to lists."""
    import torch

    def _default(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if hasattr(obj, "__float__"):
            return float(obj)
        return str(obj)

    print(_json.dumps(result, indent=2, default=_default))


def _load_model(
    model_name: str,
    device: str | None = None,
    dtype: str | None = None,
    device_map: str | None = None,
):
    from interpkit.core.model import load

    with console.status(f"Loading {model_name}..."):
        return load(model_name, device=device, dtype=dtype, device_map=device_map)


# ══════════════════════════════════════════════════════════════════
# help — rich overview panel
# ══════════════════════════════════════════════════════════════════


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    fmt: str = typer.Option("rich", "--format", help="Output format: rich (default) or json"),
) -> None:
    """Mech interp for any HuggingFace model."""
    global _output_format
    _output_format = fmt
    if ctx.invoked_subcommand is not None:
        return

    logo = r"""
IIIII         tt                          KK  KK iii tt
 III  nn nnn  tt      eee  rr rr  pp pp   KK KK      tt
 III  nnn  nn tttt  ee   e rrr  r ppp  pp KKKK   iii tttt
 III  nn   nn tt    eeeee  rr     pppppp  KK KK  iii tt
IIIII nn   nn  tttt  eeeee rr     pp      KK  KK iii  tttt
                                  pp
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
        ("", "[bold]Quick Start[/bold]", ""),
        ("scan", "One-command overview — DLA, lens, attention, attribution", "interpkit scan gpt2 'The capital of France is'"),
        ("", "", ""),
        ("", "[bold]Core Operations[/bold]", ""),
        ("inspect", "Module tree with types, params, roles", "interpkit inspect gpt2"),
        ("dla", "Direct Logit Attribution — decompose logit by component", "interpkit dla gpt2 'The capital of France is'"),
        ("trace", "Causal tracing — module or position-aware (Meng et al.)", "interpkit trace gpt2 --clean '...' --corrupted '...'"),
        ("lens", "Logit lens — project layers to vocab (all positions)", "interpkit lens gpt2 'The capital of France is'"),
        ("attribute", "Gradient saliency over inputs", "interpkit attribute gpt2 'The capital of France is'"),
        ("patch", "Activation patching at module/head/position", "interpkit patch gpt2 --clean '...' --corrupted '...' --at transformer.h.8.mlp"),
        ("", "", ""),
        ("", "[bold]Analysis Operations[/bold]", ""),
        ("activations", "Extract raw activation tensors", "interpkit activations gpt2 '...' --at transformer.h.8"),
        ("ablate", "Zero/mean ablate a component", "interpkit ablate gpt2 '...' --at transformer.h.8.mlp"),
        ("attention", "Visualize attention patterns", "interpkit attention gpt2 '...' --layer 8"),
        ("decompose", "Residual stream decomposition by component", "interpkit decompose gpt2 'The capital of France is'"),
        ("steer", "Apply a steering vector", "interpkit steer gpt2 '...' --positive Love --negative Hate --at transformer.h.8"),
        ("probe", "Linear probe on activations", "interpkit probe gpt2 --at transformer.h.8 --data data.json"),
        ("diff", "Compare two models' activations", "interpkit diff gpt2 my-finetuned-gpt2 '...'"),
        ("", "", ""),
        ("", "[bold]Circuit Analysis[/bold]", ""),
        ("find-circuit", "Automated circuit discovery via iterative ablation", "interpkit find-circuit gpt2 --clean '...' --corrupted '...'"),
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
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Print the model's module tree with types, param counts, and detected roles."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
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
    head: Optional[int] = typer.Option(None, "--head", help="Specific attention head to patch (requires attention module)"),
    positions: Optional[str] = typer.Option(None, "--positions", help="Comma-separated token positions to patch (e.g. '3,4,5')"),
    metric: str = typer.Option("logit_diff", "--metric", help="Effect metric: logit_diff, kl_div, target_prob, l2_prob"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Activation patching: swap one module's output from clean into corrupted run."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    pos_list: list[int] | None = None
    if positions is not None:
        pos_list = [int(p.strip()) for p in positions.split(",")]
    result = m.patch(clean, corrupted, at=at, head=head, positions=pos_list, metric=metric)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# trace
# ══════════════════════════════════════════════════════════════════


@app.command()
def trace(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    clean: str = typer.Option(..., "--clean", help="Clean input"),
    corrupted: str = typer.Option(..., "--corrupted", help="Corrupted input"),
    top_k: int = typer.Option(20, "--top-k", help="Scan top-K modules by proxy score. 0 = scan all."),
    mode: str = typer.Option("module", "--mode", help="Tracing mode: 'module' (default) or 'position' (Meng et al. 2D heatmap)"),
    metric: str = typer.Option("logit_diff", "--metric", help="Effect metric: logit_diff, kl_div, target_prob, l2_prob"),
    save: Optional[str] = typer.Option(None, "--save", help="Save bar chart / heatmap to file (e.g. trace.png)"),
    html_path: Optional[str] = typer.Option(None, "--html", help="Save interactive HTML to file (e.g. trace.html)"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Causal tracing: rank modules by how much patching them restores clean output."""
    effective_top_k: int | None = top_k if top_k > 0 else None
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.trace(clean, corrupted, top_k=effective_top_k, mode=mode, metric=metric, save=save, html=html_path)
    if _output_format == "json":
        _json_dump(result if isinstance(result, dict) else {"results": result})


# ══════════════════════════════════════════════════════════════════
# lens
# ══════════════════════════════════════════════════════════════════


@app.command()
def lens(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    text: str = typer.Argument(..., help="Input text"),
    save: Optional[str] = typer.Option(None, "--save", help="Save heatmap to file (e.g. lens.png)"),
    html_path: Optional[str] = typer.Option(None, "--html", help="Save interactive HTML to file"),
    position: Optional[int] = typer.Option(None, "--position", help="Single token position to analyse (-1 = last). Omit for all positions."),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Logit lens: project each layer's hidden state to vocabulary space."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.lens(text, save=save, html=html_path, position=position)
    if _output_format == "json":
        _json_dump(result if isinstance(result, dict) else {"results": result})


# ══════════════════════════════════════════════════════════════════
# attribute
# ══════════════════════════════════════════════════════════════════


@app.command()
def attribute(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text or image path"),
    target: Optional[int] = typer.Option(None, "--target", help="Target class/token index for attribution"),
    method: str = typer.Option("integrated_gradients", "--method", help="Attribution method: integrated_gradients, gradient, gradient_x_input"),
    save: Optional[str] = typer.Option(None, "--save", help="Save figure to file (e.g. attribution.png)"),
    html_path: Optional[str] = typer.Option(None, "--html", help="Save interactive HTML to file (e.g. attribution.html)"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Gradient-based attribution over input tokens or pixels."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.attribute(input_data, target=target, method=method, save=save, html=html_path)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# activations
# ══════════════════════════════════════════════════════════════════


@app.command()
def activations(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text or image path"),
    at: str = typer.Option(..., "--at", help="Module name(s) to extract, comma-separated"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Extract and display activation statistics at named modules."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    modules = [s.strip() for s in at.split(",")]
    if len(modules) == 1:
        result = m.activations(input_data, at=modules[0])
    else:
        result = m.activations(input_data, at=modules)
    if _output_format == "json":
        _json_dump(result if isinstance(result, dict) else {"activations": result})


# ══════════════════════════════════════════════════════════════════
# ablate
# ══════════════════════════════════════════════════════════════════


@app.command()
def ablate(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text or image path"),
    at: str = typer.Option(..., "--at", help="Module name to ablate"),
    method: str = typer.Option("zero", "--method", help="Ablation method: zero, mean, or resample"),
    reference: Optional[str] = typer.Option(None, "--reference", help="Reference input for resample ablation"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Zero, mean, or resample ablate a module and measure the effect on output."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.ablate(input_data, at=at, method=method, reference=reference)
    if _output_format == "json":
        _json_dump(result)


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
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Show attention patterns for transformer models."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.attention(input_data, layer=layer, head=head, save=save, html=html_path)
    if _output_format == "json" and result is not None:
        _json_dump({"results": result} if isinstance(result, list) else result)


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
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Extract a steering vector and apply it during inference."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    vector = m.steer_vector(positive, negative, at=at)
    result = m.steer(input_data, vector=vector, at=at, scale=scale, save=save)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# probe
# ══════════════════════════════════════════════════════════════════


@app.command()
def probe(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    at: str = typer.Option(..., "--at", help="Module name to probe"),
    data: str = typer.Option(..., "--data", help="JSON file with {texts: [...], labels: [...]}"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Train a linear probe on activations to test linear separability."""
    import json
    from pathlib import Path

    probe_data = json.loads(Path(data).read_text())
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.probe(texts=probe_data["texts"], labels=probe_data["labels"], at=at)
    if _output_format == "json":
        _json_dump(result)


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
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Compare activations between two models on the same input."""
    import interpkit

    m_a = _load_model(model_a_name, device=device, dtype=dtype, device_map=device_map)
    m_b = _load_model(model_b_name, device=device, dtype=dtype, device_map=device_map)
    result = interpkit.diff(m_a, m_b, input_data, save=save)
    if _output_format == "json":
        _json_dump(result)


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
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Decompose activations through a Sparse Autoencoder into interpretable features."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.features(input_data, at=at, sae=sae, top_k=top_k)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# scan
# ══════════════════════════════════════════════════════════════════


@app.command()
def scan(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text"),
    save: Optional[str] = typer.Option(None, "--save", help="Prefix for exported figures (e.g. scan → scan_dla.png, scan_lens.png)"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """One-command model overview: runs DLA, logit lens, attention, and attribution."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.scan(input_data, save=save)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# dla
# ══════════════════════════════════════════════════════════════════


@app.command()
def dla(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text"),
    token: Optional[str] = typer.Option(None, "--token", help="Target token (string or int). Uses top-1 prediction if omitted."),
    position: int = typer.Option(-1, "--position", help="Token position to analyse (-1 = last)"),
    top_k: int = typer.Option(10, "--top-k", help="Number of top/bottom contributors to show"),
    save: Optional[str] = typer.Option(None, "--save", help="Save bar chart to file (e.g. dla.png)"),
    html_path: Optional[str] = typer.Option(None, "--html", help="Save interactive HTML to file"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Direct Logit Attribution: decompose output logits by component."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    parsed_token: int | str | None = None
    if token is not None:
        try:
            parsed_token = int(token)
        except ValueError:
            parsed_token = token
    result = m.dla(input_data, token=parsed_token, position=position, top_k=top_k, save=save, html=html_path)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# decompose
# ══════════════════════════════════════════════════════════════════


@app.command()
def decompose(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text"),
    position: int = typer.Option(-1, "--position", help="Token position to decompose (-1 = last)"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Decompose the residual stream into per-component contributions."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.decompose(input_data, position=position)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# find-circuit
# ══════════════════════════════════════════════════════════════════


@app.command("find-circuit")
def find_circuit(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    clean: str = typer.Option(..., "--clean", help="Clean input text"),
    corrupted: str = typer.Option(..., "--corrupted", help="Corrupted input text"),
    threshold: float = typer.Option(0.01, "--threshold", help="Minimum ablation effect to include in circuit (0-1)"),
    method: str = typer.Option("mean", "--method", help="Ablation method: mean (default), zero, resample"),
    metric: str = typer.Option("logit_diff", "--metric", help="Effect metric: logit_diff, kl_div, target_prob, l2_prob"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Automated circuit discovery: find the minimal circuit for a behaviour."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.find_circuit(clean, corrupted, threshold=threshold, method=method, metric=metric)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# report
# ══════════════════════════════════════════════════════════════════


@app.command()
def report(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text"),
    save: str = typer.Option("report.html", "--save", help="Output HTML report path"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Generate a comprehensive HTML report: prediction, DLA, logit lens, attention, attribution."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.report(input_data, save=save)
    if _output_format == "json":
        _json_dump(result)


if __name__ == "__main__":
    app()
