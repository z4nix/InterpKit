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
        m = load(model_name, device=device, dtype=dtype, device_map=device_map)
    console.print(f"  [dim]Device: {m._device}[/dim]")
    return m


# ══════════════════════════════════════════════════════════════════
# help — rich overview panel
# ══════════════════════════════════════════════════════════════════


def _show_extensive_help() -> None:
    """Render the detailed, beginner-friendly command guide."""
    from rich.rule import Rule

    console.print()
    console.print(Panel(
        "[bold]All commands share this basic shape:[/bold]\n\n"
        "  [bold cyan]interpkit[/bold cyan] [bold]<command>[/bold] [bold yellow]<model>[/bold yellow]"
        " [dim]'your text'[/dim] [dim][options][/dim]\n\n"
        "  [bold yellow]<model>[/bold yellow] is any HuggingFace model ID —"
        " e.g. [dim]gpt2[/dim], [dim]EleutherAI/pythia-70m[/dim], [dim]meta-llama/Llama-3-8B[/dim]\n\n"
        "  Most commands accept [bold green]--save path.png[/bold green] to export a figure"
        " and [bold green]--html path.html[/bold green] for an interactive version.\n"
        "  Use [bold green]--device cpu|cuda|mps[/bold green] and [bold green]--dtype float16|bfloat16|float32|auto"
        "[/bold green] to control how the model loads.",
        title="[bold cyan]InterpKit — Beginner's Command Guide[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # ── Quick Start ───────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Quick Start[/bold]", style="cyan"))
    console.print()

    console.print(Panel(
        "[bold cyan]scan[/bold cyan]  [dim]interpkit scan gpt2 'The capital of France is'[/dim]\n\n"
        "The best place to start. Runs four analyses in a single pass — DLA, logit lens, attention,"
        " and gradient attribution — and prints a combined overview. Think of it as a model health"
        " check that gives you a broad picture before you zoom in on anything specific.\n\n"
        "  [bold green]--save prefix[/bold green]  writes each sub-figure to [dim]prefix_dla.png[/dim],"
        " [dim]prefix_lens.png[/dim], etc.",
        title="scan",
        border_style="dim cyan",
        padding=(0, 2),
    ))

    console.print()
    console.print(Panel(
        "[bold cyan]report[/bold cyan]  [dim]interpkit report gpt2 'The capital of France is'[/dim]\n\n"
        "Like [bold cyan]scan[/bold cyan], but bundles everything into a self-contained, interactive"
        " HTML file instead of printing to the terminal. Hand it to a colleague or open it in a"
        " browser for a polished, shareable analysis.\n\n"
        "  [bold green]--save report.html[/bold green]  output path (default: [dim]report.html[/dim])",
        title="report",
        border_style="dim cyan",
        padding=(0, 2),
    ))

    # ── Core Operations ───────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Core Operations[/bold]", style="cyan"))
    console.print()

    entries = [
        (
            "inspect",
            "interpkit inspect gpt2",
            "Prints the model's internal module tree — every layer, its type, parameter count, and"
            " the role InterpKit inferred for it (e.g. attention, MLP, embedding). Run this first"
            " whenever you're working with an unfamiliar architecture; the module names printed here"
            " are what you pass to [bold green]--at[/bold green] in other commands.",
            [],
        ),
        (
            "dla",
            "interpkit dla gpt2 'The capital of France is'",
            "Direct Logit Attribution. Decomposes the model's final logit for the predicted token"
            " into per-component contributions. A [bold green]positive score[/bold green] means that"
            " layer/head pushed the model toward that token; a [bold red]negative score[/bold red]"
            " means it pushed against it. Great for a quick answer to 'which parts of the model are"
            " responsible for this prediction?'",
            [
                ("--token", "Target token to explain (string or integer ID). Defaults to the top-1 prediction."),
                ("--position", "Which token position to attribute (-1 = the last one)."),
                ("--top-k", "How many top/bottom contributors to display (default 10)."),
            ],
        ),
        (
            "trace",
            "interpkit trace gpt2 --clean 'The Eiffel Tower is in Paris' --corrupted 'The Eiffel Tower is in Rome'",
            "Causal tracing (Meng et al. 2022). You give it a clean input and a corrupted one."
            " It runs both, then systematically patches each module's activation from the clean run"
            " into the corrupted run and measures how much the output recovers. The modules with the"
            " highest recovery score are causally responsible for the behavior — they carry the"
            " 'right' information.",
            [
                ("--mode module", "Rank modules by causal impact (default)."),
                ("--mode position", "2-D heatmap over (layer × token position), like the original paper."),
                ("--top-k", "How many modules to scan (0 = all, which is slower)."),
                ("--metric", "logit_diff · kl_div · target_prob · l2_prob"),
            ],
        ),
        (
            "lens",
            "interpkit lens gpt2 'The capital of France is'",
            "Logit lens. After every transformer layer, the hidden state is projected directly into"
            " vocabulary space so you can see what the model 'thinks' it's predicting at each depth."
            " Lets you watch a vague representation sharpen into the final answer layer by layer.",
            [
                ("--position N", "Analyse a single token position instead of all positions."),
            ],
        ),
        (
            "attribute",
            "interpkit attribute gpt2 'The capital of France is'",
            "Gradient-based input attribution. Computes how much each input token influenced the"
            " output by following gradients back through the network. Useful when you want token-level"
            " importance — 'which words in my prompt drove this prediction?'",
            [
                ("--method", "integrated_gradients (default, most faithful) · gradient · gradient_x_input"),
                ("--target", "Target class or token index for attribution."),
            ],
        ),
        (
            "patch",
            "interpkit patch gpt2 --clean '...' --corrupted '...' --at transformer.h.8.mlp",
            "Activation patching. The experiment works like this: you run the model on two inputs —"
            " a [bold green]--clean[/bold green] one that produces the right answer, and a"
            " [bold green]--corrupted[/bold green] one that doesn't. Both runs complete normally."
            " Then, for the single module you specify with [bold green]--at[/bold green], you take"
            " its output from the clean run and silently swap it in during the corrupted run —"
            " everything else stays from the corrupted run. You then check whether the output"
            " recovers toward the correct answer.\n\n"
            "  [bold]If it recovers:[/bold] that module was the one carrying the critical"
            " information — the corrupted run had the right answer sitting there, it just wasn't"
            " being used.\n"
            "  [bold]If it doesn't:[/bold] the information isn't stored there; look elsewhere.\n\n"
            "  Think of it as a targeted transplant: you're isolating one component and asking"
            " 'is the fix inside here?' Use [bold cyan]trace[/bold cyan] first to rank candidates,"
            " then [bold cyan]patch[/bold cyan] to confirm.",
            [
                ("--at", "Module to patch — get exact names from [bold cyan]inspect[/bold cyan]."),
                ("--head", "Patch only a specific attention head within the module."),
                ("--positions", "Restrict the patch to certain token positions (e.g. 3,4,5)."),
                ("--metric", "How to measure recovery: logit_diff · kl_div · target_prob · l2_prob"),
            ],
        ),
    ]

    for name, example, description, opts in entries:
        opt_lines = ""
        if opts:
            opt_lines = "\n\n  [bold]Key options:[/bold]\n" + "\n".join(
                f"    [bold green]{k}[/bold green]  {v}" for k, v in opts
            )
        console.print(Panel(
            f"[bold cyan]{name}[/bold cyan]  [dim]{example}[/dim]\n\n{description}{opt_lines}",
            title=name,
            border_style="dim cyan",
            padding=(0, 2),
        ))
        console.print()

    # ── Analysis Operations ───────────────────────────────────────
    console.print(Rule("[bold]Analysis Operations[/bold]", style="cyan"))
    console.print()

    analysis_entries = [
        (
            "activations",
            "interpkit activations gpt2 'Hello world' --at transformer.h.8",
            "Extracts the raw activation tensor at one or more named modules and prints summary"
            " statistics (shape, mean, std, min/max). Use this when you want to inspect or export"
            " internal representations directly — for instance, to feed them into your own analysis.",
            [("--at", "Module name(s), comma-separated. Find names with [bold cyan]inspect[/bold cyan].")],
        ),
        (
            "ablate",
            "interpkit ablate gpt2 'Hello world' --at transformer.h.8.mlp",
            "Ablation study. Replaces a module's output with zeros, its mean activation, or a"
            " resampled value from another input, then reports how much the prediction changed."
            " If ablating a module wrecks the output, that module matters. If nothing changes,"
            " the module is likely redundant for this behavior.",
            [
                ("--method", "zero (default) · mean · resample"),
                ("--reference", "Reference text for resample ablation."),
            ],
        ),
        (
            "attention",
            "interpkit attention gpt2 'The capital of France is' --layer 8",
            "Visualizes attention weight heatmaps for transformer models, showing which tokens"
            " attend to which other tokens at each layer and head. Use [bold green]--layer[/bold green]"
            " and [bold green]--head[/bold green] to zoom into a specific one.",
            [
                ("--layer N", "Only show this layer (omit for all layers)."),
                ("--head N", "Only show this head within the layer."),
            ],
        ),
        (
            "decompose",
            "interpkit decompose gpt2 'The capital of France is'",
            "Breaks down the residual stream at a given token position into contributions from each"
            " individual component — embeddings, each attention layer, each MLP. Similar to DLA but"
            " at the residual stream level rather than the final logit.",
            [("--position", "Token position to decompose (-1 = last).")],
        ),
        (
            "steer",
            "interpkit steer gpt2 'The sky is' --positive Love --negative Hate --at transformer.h.8",
            "Activation steering. Computes a 'steering vector' as the mean-difference between"
            " activations for contrasting concepts ([bold green]--positive[/bold green] vs"
            " [bold green]--negative[/bold green]), then adds a scaled copy of it to the activations"
            " of the specified module during inference. Shows how the model's output shifts when you"
            " nudge it in that direction.\n\n"
            "  For more robust vectors, pass text files with many examples instead of a single pair."
            " The activations are averaged across all examples before computing the difference"
            " (Contrastive Activation Addition).\n"
            "  [dim]interpkit steer gpt2 'The sky is' --positive-file pos.txt --negative-file neg.txt --at transformer.h.8[/dim]",
            [
                ("--positive / --negative", "Single contrasting concept texts that define the direction."),
                ("--positive-file / --negative-file", "Text files with one example per line for multi-example steering."),
                ("--at", "Which module to apply the vector at."),
                ("--scale", "How strongly to apply it (default 2.0; higher = more extreme)."),
            ],
        ),
        (
            "probe",
            "interpkit probe gpt2 --at transformer.h.8 --data data.json",
            "Trains a lightweight linear classifier on top of a module's activations using labeled"
            " examples you provide, then reports accuracy. If the probe does well, the concept you're"
            " testing is [italic]linearly[/italic] encoded at that location in the network — a strong"
            " sign it's represented in a human-interpretable direction.\n\n"
            "  [dim]data.json should contain: {\"texts\": [...], \"labels\": [...]}[/dim]",
            [
                ("--at", "Module to probe."),
                ("--data", "Path to a JSON file with texts and integer labels."),
            ],
        ),
        (
            "diff",
            "interpkit diff gpt2 my-finetuned-gpt2 'The capital of France is'",
            "Runs two models on the same input and compares their activations layer by layer,"
            " highlighting where they diverge most. Useful for understanding what fine-tuning changed"
            " internally — not just in outputs.",
            [],
        ),
    ]

    for name, example, description, opts in analysis_entries:
        opt_lines = ""
        if opts:
            opt_lines = "\n\n  [bold]Key options:[/bold]\n" + "\n".join(
                f"    [bold green]{k}[/bold green]  {v}" for k, v in opts
            )
        console.print(Panel(
            f"[bold cyan]{name}[/bold cyan]  [dim]{example}[/dim]\n\n{description}{opt_lines}",
            title=name,
            border_style="dim cyan",
            padding=(0, 2),
        ))
        console.print()

    # ── Circuit Analysis ──────────────────────────────────────────
    console.print(Rule("[bold]Circuit Analysis[/bold]", style="cyan"))
    console.print()

    console.print(Panel(
        "[bold cyan]find-circuit[/bold cyan]  "
        "[dim]interpkit find-circuit gpt2 --clean '...' --corrupted '...'[/dim]\n\n"
        "Automated circuit discovery. Iteratively ablates every module and keeps only those whose"
        " removal meaningfully changes the output (above [bold green]--threshold[/bold green])."
        " What remains is the minimal set of components responsible for the behavior — the"
        " 'circuit' in the mechanistic interpretability sense. Can be slow on large models since"
        " it runs many forward passes.\n\n"
        "  For more robust circuits, pass text files with multiple clean/corrupted pairs."
        " Ablation effects are averaged across all pairs, keeping only components that are"
        " consistently important.\n"
        "  [dim]interpkit find-circuit gpt2 --clean-file cleans.txt --corrupted-file corrupteds.txt[/dim]\n\n"
        "  [bold]Key options:[/bold]\n"
        "    [bold green]--clean / --corrupted[/bold green]  Single clean and corrupted input texts.\n"
        "    [bold green]--clean-file / --corrupted-file[/bold green]  Text files with one example per line (paired by line number).\n"
        "    [bold green]--threshold[/bold green]  Minimum ablation effect to include (default 0.01).\n"
        "    [bold green]--method[/bold green]  Ablation method: mean (default), zero, resample.\n"
        "    [bold green]--metric[/bold green]  logit_diff · kl_div · target_prob · l2_prob",
        title="find-circuit",
        border_style="dim cyan",
        padding=(0, 2),
    ))
    console.print()

    console.print(Panel(
        "[bold cyan]features[/bold cyan]  "
        "[dim]interpkit features gpt2 '...' --at transformer.h.8 --sae jbloom/GPT2-Small-SAEs[/dim]\n\n"
        "Sparse Autoencoder (SAE) feature decomposition. Takes a module's activation and projects"
        " it through a separately trained SAE to recover a sparse set of interpretable features."
        " Each feature typically corresponds to a human-readable concept. Requires a compatible"
        " SAE checkpoint available on HuggingFace.\n\n"
        "  [bold]Contrastive mode:[/bold] pass [bold green]--positive-file[/bold green] and"
        " [bold green]--negative-file[/bold green] (omit the input text argument) to find features that"
        " differentially activate between two groups of inputs.\n"
        "  [dim]interpkit features gpt2 --at transformer.h.8 --sae jbloom/... --positive-file pos.txt --negative-file neg.txt[/dim]\n\n"
        "  [bold green]--at[/bold green]   Which module's activations to decompose.\n"
        "  [bold green]--sae[/bold green]  HuggingFace repo ID of the SAE weights.\n"
        "  [bold green]--top-k[/bold green]  How many top features to display (default 20).\n"
        "  [bold green]--positive-file / --negative-file[/bold green]  Text files for contrastive feature analysis.",
        title="features",
        border_style="dim cyan",
        padding=(0, 2),
    ))

    console.print()
    console.print(
        "  Run [bold cyan]interpkit <command> --help[/bold cyan] for the full option list of any command.\n"
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    fmt: str = typer.Option("rich", "--format", help="Output format: rich (default) or json"),
    extensive: bool = typer.Option(
        False,
        "--extensive",
        help=(
            "Show a detailed, beginner-friendly explanation of every command. "
            "Useful if you're new to mech interp or want to understand what each command actually does."
        ),
    ),
) -> None:
    """Mech interp for any HuggingFace model."""
    global _output_format
    _output_format = fmt
    if ctx.invoked_subcommand is not None:
        return
    if extensive:
        _show_extensive_help()
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
        ("steer", "Steering vector (inline or file-based)", "interpkit steer gpt2 '...' --positive-file pos.txt --negative-file neg.txt --at ..."),
        ("probe", "Linear probe on activations", "interpkit probe gpt2 --at transformer.h.8 --data data.json"),
        ("diff", "Compare two models' activations", "interpkit diff gpt2 my-finetuned-gpt2 '...'"),
        ("", "", ""),
        ("", "[bold]Circuit Analysis[/bold]", ""),
        ("find-circuit", "Circuit discovery (single or multi-pair)", "interpkit find-circuit gpt2 --clean-file c.txt --corrupted-file r.txt"),
        ("features", "SAE features (single or contrastive)", "interpkit features gpt2 '...' --at ... --sae jbloom/..."),
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
    console.print(
        "  Run [bold cyan]interpkit <command> --help[/bold cyan] for detailed usage.\n"
        "  New here? Run [bold cyan]interpkit --extensive[/bold cyan] for a plain-English"
        " walkthrough of every command.\n"
    )


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
    with console.status("Inspecting model..."):
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
    with console.status("Running activation patching..."):
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
    with console.status("Running logit lens..."):
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
    with console.status("Extracting activations..."):
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
    with console.status("Running ablation..."):
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
    with console.status("Computing attention patterns..."):
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
    positive: Optional[str] = typer.Option(None, "--positive", help="Positive direction text (single example)"),
    negative: Optional[str] = typer.Option(None, "--negative", help="Negative direction text (single example)"),
    positive_file: Optional[str] = typer.Option(None, "--positive-file", help="Text file with positive examples, one per line"),
    negative_file: Optional[str] = typer.Option(None, "--negative-file", help="Text file with negative examples, one per line"),
    at: str = typer.Option(..., "--at", help="Module name to apply steering at"),
    scale: float = typer.Option(2.0, "--scale", help="Steering vector scale factor"),
    save: Optional[str] = typer.Option(None, "--save", help="Save comparison chart to file"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Extract a steering vector and apply it during inference."""
    from interpkit.core.inputs import read_examples_file

    pos_inputs: str | list[str]
    neg_inputs: str | list[str]

    if positive_file:
        pos_inputs = read_examples_file(positive_file)
    elif positive:
        pos_inputs = positive
    else:
        raise typer.BadParameter("Provide --positive or --positive-file")

    if negative_file:
        neg_inputs = read_examples_file(negative_file)
    elif negative:
        neg_inputs = negative
    else:
        raise typer.BadParameter("Provide --negative or --negative-file")

    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    vector = m.steer_vector(pos_inputs, neg_inputs, at=at)
    with console.status("Running steered inference..."):
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
    with console.status("Comparing models..."):
        result = interpkit.diff(m_a, m_b, input_data, save=save)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# features (SAE)
# ══════════════════════════════════════════════════════════════════


@app.command()
def features(
    model_name: str = typer.Argument(..., help="HuggingFace model ID (e.g. gpt2)"),
    input_data: Optional[str] = typer.Argument(None, help="Input text (omit when using --positive-file / --negative-file)"),
    at: str = typer.Option(..., "--at", help="Module name to decompose (e.g. transformer.h.8)"),
    sae: str = typer.Option(..., "--sae", help="HuggingFace repo ID for the SAE weights"),
    top_k: int = typer.Option(20, "--top-k", help="Number of top features to display"),
    positive_file: Optional[str] = typer.Option(None, "--positive-file", help="Text file with positive examples for contrastive analysis, one per line"),
    negative_file: Optional[str] = typer.Option(None, "--negative-file", help="Text file with negative examples for contrastive analysis, one per line"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Decompose activations through a Sparse Autoencoder into interpretable features."""
    contrastive = positive_file is not None or negative_file is not None
    if contrastive:
        if not positive_file or not negative_file:
            raise typer.BadParameter("Both --positive-file and --negative-file are required for contrastive mode")

        from interpkit.core.inputs import read_examples_file

        pos_inputs = read_examples_file(positive_file)
        neg_inputs = read_examples_file(negative_file)
        m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
        result = m.contrastive_features(pos_inputs, neg_inputs, at=at, sae=sae, top_k=top_k)
    else:
        if input_data is None:
            raise typer.BadParameter("Provide input text or use --positive-file / --negative-file for contrastive mode")
        m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
        with console.status("Decomposing features..."):
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
    with console.status("Running DLA..."):
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
    with console.status("Decomposing residual stream..."):
        result = m.decompose(input_data, position=position)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# find-circuit
# ══════════════════════════════════════════════════════════════════


@app.command("find-circuit")
def find_circuit(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    clean: Optional[str] = typer.Option(None, "--clean", help="Clean input text (single example)"),
    corrupted: Optional[str] = typer.Option(None, "--corrupted", help="Corrupted input text (single example)"),
    clean_file: Optional[str] = typer.Option(None, "--clean-file", help="Text file with clean examples, one per line"),
    corrupted_file: Optional[str] = typer.Option(None, "--corrupted-file", help="Text file with corrupted examples, one per line (must match --clean-file line count)"),
    threshold: float = typer.Option(0.01, "--threshold", help="Minimum ablation effect to include in circuit (0-1)"),
    method: str = typer.Option("mean", "--method", help="Ablation method: mean (default), zero, resample"),
    metric: str = typer.Option("logit_diff", "--metric", help="Effect metric: logit_diff, kl_div, target_prob, l2_prob"),
    device: Optional[str] = typer.Option(None, help="Device"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Automated circuit discovery: find the minimal circuit for a behaviour."""
    from interpkit.core.inputs import read_examples_file

    clean_inputs: str | list[str]
    corrupted_inputs: str | list[str]

    if clean_file:
        clean_inputs = read_examples_file(clean_file)
    elif clean:
        clean_inputs = clean
    else:
        raise typer.BadParameter("Provide --clean or --clean-file")

    if corrupted_file:
        corrupted_inputs = read_examples_file(corrupted_file)
    elif corrupted:
        corrupted_inputs = corrupted
    else:
        raise typer.BadParameter("Provide --corrupted or --corrupted-file")

    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.find_circuit(clean_inputs, corrupted_inputs, threshold=threshold, method=method, metric=metric)
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
