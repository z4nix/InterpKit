"""CLI entry point — Typer app with all interpkit commands.

When ``--format json`` is set, all status / progress output (rich panels,
load progress bars, tqdm) is silenced or routed to stderr (F-023). The
stdout stream stays clean JSON for programmatic consumers — pre-1.0
``--format json`` interleaved rich panels and tqdm bars with the JSON
block, breaking ``json.loads(p.stdout)`` for every CLI invocation.
"""

from __future__ import annotations

import json as _json
import os as _os
import sys as _sys
from importlib.metadata import version as _pkg_version

import typer
import typer.rich_utils as _ru
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich_gradient import Rule as GradientRule
from rich_gradient import Text as GradientText

from interpkit.core.theme import ACCENT, ACCENT_DIM, BRAND_COLORS

_ru.STYLE_OPTION = f"bold {ACCENT}"
_ru.STYLE_SWITCH = f"bold {ACCENT}"
_ru.STYLE_METAVAR = f"bold {ACCENT}"
_ru.STYLE_USAGE = ACCENT
_ru.STYLE_USAGE_COMMAND = "bold"
_ru.STYLE_COMMANDS_TABLE_FIRST_COLUMN = f"bold {ACCENT}"
_ru.STYLE_OPTIONS_PANEL_BORDER = ACCENT_DIM
_ru.STYLE_COMMANDS_PANEL_BORDER = ACCENT_DIM
_ru.STYLE_REQUIRED_SHORT = ACCENT
_ru.STYLE_REQUIRED_LONG = ACCENT_DIM
_ru.STYLE_NEGATIVE_OPTION = f"bold {ACCENT}"

app = typer.Typer(
    name="interpkit",
    help="Mech interp for any HuggingFace model.",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
    # interpkit's own errors (OperationNotSupportedForArchitecture,
    # WrongInputType, LensPipelineMismatch, …) are deliberate, well-messaged,
    # user-facing failures — not bugs. Disable Typer's rich-traceback so they
    # don't reach the user as a scary stack trace; ``run()`` renders them as a
    # clean one-line error instead.
    pretty_exceptions_enable=False,
)
# F-023: console object — production code should call _make_console() so
# JSON-mode stderr routing happens uniformly. The module-level singleton
# is reassigned by main() once --format is parsed.
console = Console()

_output_format: str = "rich"


def _make_console() -> Console:
    """Construct a Console that respects the active output format.

    In ``json`` mode, status / progress output goes to stderr so stdout
    remains clean JSON. In ``rich`` mode, behaves identically to the
    pre-1.0 module-level singleton.
    """
    if _output_format == "json":
        return Console(file=_sys.stderr)
    return Console()


def _silence_third_party_loaders() -> None:
    """Mute transformers / tqdm / huggingface chatter in JSON mode.

    Pre-1.0 ``--format json`` had model-loading tqdm bars and the
    "Loaded ... on cpu" rich line interleaved with the actual JSON
    payload (F-023). Programmatic consumers couldn't json.loads(stdout).

    Also re-binds every op-module console to write to stderr so rich
    op-level rendering doesn't pollute the JSON stream.
    """
    if _output_format != "json":
        return
    # Silence HF transformers progress / warnings to stderr-only.
    try:
        from transformers import logging as _hf_logging
        _hf_logging.set_verbosity_error()
        _hf_logging.disable_progress_bar()
    except (ImportError, AttributeError):
        pass
    # Silence raw tqdm.
    _os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    _os.environ["TQDM_DISABLE"] = "1"
    _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Re-bind op-module consoles to stderr so renders don't pollute stdout.
    import importlib

    _stderr_console = Console(file=_sys.stderr)
    for mod_name in (
        "interpkit.core.render",
        "interpkit.core.plot",
        "interpkit.ops.attention",
        "interpkit.ops.attribute",
        "interpkit.ops.batch",
        "interpkit.ops.circuits",
        "interpkit.ops.diff",
        "interpkit.ops.find_circuit",
        "interpkit.ops.lens",
        "interpkit.ops.probe",
        "interpkit.ops.report",
        "interpkit.ops.sae",
        "interpkit.ops.scan",
        "interpkit.ops.steer",
        "interpkit.ops.trace",
    ):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "console"):
                mod.console = _stderr_console  # type: ignore[attr-defined]
        except ImportError:
            continue

_VERSION = _pkg_version("interpkit")


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

    # F-007 fix: don't forward dtype=None — load() now requires explicit
    # dtype. Defer to its built-in default (fp32) when the CLI user didn't
    # specify --dtype.
    kwargs: dict = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map

    with console.status(f"  Loading [bold]{model_name}[/bold]..."):
        m = load(model_name, **kwargs)
    console.print(f"  [bold green]Loaded[/bold green] [{ACCENT}]{model_name}[/{ACCENT}] on [bold]{m._device}[/bold]")
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
        f"  [bold {ACCENT}]interpkit[/bold {ACCENT}] [bold]<command>[/bold] [bold yellow]<model>[/bold yellow]"
        " [dim]'your text'[/dim] [dim][options][/dim]\n\n"
        "  [bold yellow]<model>[/bold yellow] is any HuggingFace model ID —"
        " e.g. [dim]gpt2[/dim], [dim]EleutherAI/pythia-70m[/dim], [dim]meta-llama/Llama-3-8B[/dim]\n\n"
        "  Most commands accept [bold green]--save path.png[/bold green] to export a figure"
        " and [bold green]--html path.html[/bold green] for an interactive version.\n"
        "  Use [bold green]--device cpu|cuda|mps[/bold green] and [bold green]--dtype float16|bfloat16|float32|auto"
        "[/bold green] to control how the model loads.",
        title=f"[bold {ACCENT}]InterpKit — Beginner's Command Guide[/bold {ACCENT}]",
        border_style=ACCENT,
        padding=(1, 2),
    ))

    # ── Quick Start ───────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Quick Start[/bold]", style=ACCENT))
    console.print()

    console.print(Panel(
        f"[bold {ACCENT}]gui[/bold {ACCENT}]  [dim]interpkit gui[/dim]\n\n"
        "Prefer clicking to typing? Launch the local web GUI — a point-and-click interface that runs"
        " every command in your browser, with module/layer pickers populated from the model's detected"
        " architecture and results drawn natively (heatmaps, bar charts, attention explorer, chat).\n\n"
        "  Requires the optional extra: [bold green]pip install \"interpkit\\[gui]\"[/bold green]\n"
        "  [bold green]--host / --port[/bold green]  Bind address (default 127.0.0.1:7860).\n"
        "  [bold green]--no-browser[/bold green]  Don't open the browser automatically.",
        title="gui",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))

    console.print()
    console.print(Panel(
        f"[bold {ACCENT}]scan[/bold {ACCENT}]  [dim]interpkit scan gpt2 'The capital of France is'[/dim]\n\n"
        "The best place to start. Runs four analyses in a single pass — DLA, logit lens, attention,"
        " and gradient attribution — and prints a combined overview. Think of it as a model health"
        " check that gives you a broad picture before you zoom in on anything specific.\n\n"
        "  [bold green]--save prefix[/bold green]  writes each sub-figure to [dim]prefix_dla.png[/dim],"
        " [dim]prefix_lens.png[/dim], etc.",
        title="scan",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))

    console.print()
    console.print(Panel(
        f"[bold {ACCENT}]report[/bold {ACCENT}]  [dim]interpkit report gpt2 'The capital of France is'[/dim]\n\n"
        f"Like [bold {ACCENT}]scan[/bold {ACCENT}], but bundles everything into a self-contained, interactive"
        " HTML file instead of printing to the terminal. Hand it to a colleague or open it in a"
        " browser for a polished, shareable analysis.\n\n"
        "  [bold green]--save report.html[/bold green]  output path (default: [dim]report.html[/dim])",
        title="report",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))

    console.print()
    console.print(Panel(
        f"[bold {ACCENT}]chat[/bold {ACCENT}]  "
        "[dim]interpkit chat HuggingFaceTB/SmolLM2-360M-Instruct 'Write a haiku.'[/dim]\n\n"
        "Send a message to an instruction-tuned chat model and print its reply. The message is"
        " routed through the tokenizer's chat template (e.g. ChatML, Llama-2 Inst, Qwen, Gemma)"
        " with [dim]add_generation_prompt=True[/dim] before generation, so any HF chat model that"
        " ships a template just works.\n\n"
        "  Errors clearly when the model has no chat template (i.e. a base/non-instruct model) —"
        " in that case load an instruct variant or call any other command with a plain string.\n\n"
        "  [bold]Key options:[/bold]\n"
        "    [bold green]--system 'be brief'[/bold green]  Optional system prompt prepended to the conversation.\n"
        "    [bold green]--max-new-tokens N[/bold green]  Generation budget (default 128).\n"
        "    [bold green]--sample / --no-sample[/bold green]  Sampling vs greedy decoding (default greedy).\n"
        "    [bold green]--temperature / --top-p[/bold green]  Standard sampling controls (used when --sample).\n"
        "    [bold green]--show-prompt[/bold green]  Print the chat-templated prompt before generating.",
        title="chat",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))

    console.print()
    console.print(Panel(
        f"[bold {ACCENT}]generate[/bold {ACCENT}]  "
        "[dim]interpkit generate gpt2 'I feel' --positive ' joy' --negative ' fear' --at transformer.h.6 --scale 8[/dim]\n\n"
        "Generate text with interventions active across [italic]every[/italic] decode step —"
        " the generation-time counterpart of [bold]steer[/bold] / [bold]ablate[/bold], which"
        " analyse a single forward pass. A steering vector or ablation stays hooked for the"
        " prefill and all KV-cached decode steps, so you can watch a nudged model write.\n\n"
        "  Add [bold green]--capture lens[/bold green] to record each generated token's"
        " logit-lens trajectory: which layer first predicted the token the model ended up"
        " emitting.\n\n"
        "  [bold]Key options:[/bold]\n"
        "    [bold green]--positive / --negative + --at[/bold green]  Build a steering vector and apply it while generating.\n"
        "    [bold green]--sae + --feature + --at[/bold green]  Clamp an SAE feature while generating (Golden Gate style; --feature-mode add|clamp, --strength).\n"
        "    [bold green]--ablate-at / --ablate-method[/bold green]  Knock out a module for the whole generation.\n"
        "    [bold green]--capture lens|logits[/bold green]  Per-token lens trajectory or raw step logits.\n"
        "    [bold green]--max-new-tokens N[/bold green]  Generation budget (default 64).\n"
        "    [bold green]--sample / --temperature / --top-p[/bold green]  Standard sampling controls.",
        title="generate",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))

    # ── Core Operations ───────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Core Operations[/bold]", style=ACCENT))
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
            " Lets you watch a vague representation sharpen into the final answer layer by layer.\n\n"
            "  The raw projection is biased for early layers (their basis isn't aligned with the"
            " unembedding). Train per-layer translators once with"
            f" [bold {ACCENT}]train-tuned-lens[/bold {ACCENT}] and pass"
            " [bold green]--tuned-lens <path>[/bold green] for the unbiased tuned-lens readout"
            " (Belrose et al. 2023).\n"
            "  [dim]interpkit train-tuned-lens gpt2 --corpus-file texts.txt --save lens_dir/[/dim]\n"
            "  [dim]interpkit lens gpt2 'The capital of France is' --tuned-lens lens_dir/[/dim]",
            [
                ("--position N", "Analyse a single token position instead of all positions."),
                ("--tuned-lens PATH", "Apply saved tuned-lens translators instead of the raw projection."),
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
            f" 'is the fix inside here?' Use [bold {ACCENT}]trace[/bold {ACCENT}] first to rank candidates,"
            f" then [bold {ACCENT}]patch[/bold {ACCENT}] to confirm.",
            [
                ("--at", f"Module to patch — get exact names from [bold {ACCENT}]inspect[/bold {ACCENT}]."),
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
            f"[bold {ACCENT}]{name}[/bold {ACCENT}]  [dim]{example}[/dim]\n\n{description}{opt_lines}",
            title=name,
            border_style=ACCENT_DIM,
            padding=(0, 2),
        ))
        console.print()

    # ── Analysis Operations ───────────────────────────────────────
    console.print(Rule("[bold]Analysis Operations[/bold]", style=ACCENT))
    console.print()

    analysis_entries = [
        (
            "activations",
            "interpkit activations gpt2 'Hello world' --at transformer.h.8",
            "Extracts the raw activation tensor at one or more named modules and prints summary"
            " statistics (shape, mean, std, min/max). Use this when you want to inspect or export"
            " internal representations directly — for instance, to feed them into your own analysis.",
            [("--at", f"Module name(s), comma-separated. Find names with [bold {ACCENT}]inspect[/bold {ACCENT}].")],
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
            "interpkit steer gpt2 'The sky is' --positive ' love' --negative ' hate' --at transformer.h.8",
            "Activation steering. Computes a 'steering vector' as the mean-difference between"
            " activations for contrasting concepts ([bold green]--positive[/bold green] vs"
            " [bold green]--negative[/bold green]), then adds a scaled copy of it to the activations"
            " of the specified module during inference. Shows how the model's output shifts when you"
            " nudge it in that direction.\n\n"
            "  For more robust vectors, pass text files with many examples instead of a single pair."
            " The activations are averaged across all examples before computing the difference"
            " (Contrastive Activation Addition).\n"
            "  [dim]interpkit steer gpt2 'The sky is' --positive-file pos.txt --negative-file neg.txt --at transformer.h.8[/dim]\n\n"
            "  Alternatively, steer along a single [bold]SAE feature[/bold]'s decoder direction"
            " ([bold green]--sae[/bold green] + [bold green]--feature[/bold green]) — the Golden Gate"
            " Claude manipulation. [bold green]--feature-mode clamp[/bold green] (default) pins the"
            " feature's activation to [bold green]--strength[/bold green]; [bold green]add[/bold green]"
            " injects the direction unconditionally. Find feature indices with"
            " [bold]features[/bold] or [bold]maxact[/bold].\n"
            "  [dim]interpkit steer gpt2 'My favorite place in the world is' --sae jbloom/GPT2-Small-SAEs-Reformatted/blocks.8.hook_resid_pre --feature 9752 --at transformer.h.7 --strength 50[/dim]",
            [
                ("--positive / --negative", "Single contrasting concept texts that define the direction."),
                ("--positive-file / --negative-file", "Text files with one example per line for multi-example steering."),
                ("--at", "Which module to apply the steering at."),
                ("--scale", "How strongly to apply a contrastive vector (default 2.0; higher = more extreme)."),
                ("--sae / --feature", "Steer along an SAE feature's decoder direction instead (mutually exclusive with --positive/--negative)."),
                ("--feature-mode", "'clamp' (pin the feature's activation; default) or 'add' (inject the direction)."),
                ("--strength", "Feature target activation (clamp) or added activation (add); try a few times the feature's max activation."),
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
            f"[bold {ACCENT}]{name}[/bold {ACCENT}]  [dim]{example}[/dim]\n\n{description}{opt_lines}",
            title=name,
            border_style=ACCENT_DIM,
            padding=(0, 2),
        ))
        console.print()

    # ── Circuit Analysis ──────────────────────────────────────────
    console.print(Rule("[bold]Circuit Analysis[/bold]", style=ACCENT))
    console.print()

    console.print(Panel(
        f"[bold {ACCENT}]find-circuit[/bold {ACCENT}]  "
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
        "    [bold green]--method[/bold green]  mean (default) · zero · resample (ablation), or eap · eap-ig"
        " (gradient-based selection in a handful of passes — much faster; the circuit is still"
        " verified causally).\n"
        "    [bold green]--metric[/bold green]  logit_diff · kl_div · target_prob · l2_prob",
        title="find-circuit",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))
    console.print()

    console.print(Panel(
        f"[bold {ACCENT}]atp[/bold {ACCENT}]  "
        "[dim]interpkit atp gpt2 --clean 'The capital of France is' --corrupted 'The capital of Germany is'[/dim]\n\n"
        "Attribution Patching (Syed et al. 2023). A first-order gradient approximation of"
        " activation patching: one clean forward, one corrupted forward, and one backward pass"
        " score [italic]every[/italic] module simultaneously — versus one forward per module for"
        " exhaustive tracing. Correlation with true patch effects is typically 0.85–0.95."
        " Use it as the fast first look, then confirm top candidates with"
        f" [bold {ACCENT}]trace[/bold {ACCENT}] or [bold {ACCENT}]patch[/bold {ACCENT}].\n\n"
        "  [bold]Key options:[/bold]\n"
        "    [bold green]--clean / --corrupted[/bold green]  The contrast pair to attribute.\n"
        "    [bold green]--top-k[/bold green]  Top modules to report by absolute score (0 = all).",
        title="atp",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))
    console.print()

    console.print(Panel(
        f"[bold {ACCENT}]eap[/bold {ACCENT}]  "
        "[dim]interpkit eap gpt2 --clean 'The capital of France is' --corrupted 'The capital of Germany is'[/dim]\n\n"
        "Edge Attribution Patching. Where [bold]atp[/bold] scores modules, eap scores"
        " [italic]edges[/italic]: how much each component's clean-vs-corrupted delta matters as it"
        " flows into each downstream residual-stream layer. The edge at a component's own layer is"
        " its total effect; deeper edges show how the effect persists down the stream. Inputs must"
        " tokenize to the same length.\n\n"
        "  [bold green]--ig-steps 5[/bold green] switches to EAP-IG: gradients averaged over"
        " embeddings interpolated from corrupted toward clean — more faithful scores when the"
        " corrupted point sits in a saturated region.\n\n"
        "  [bold]Key options:[/bold]\n"
        "    [bold green]--clean / --corrupted[/bold green]  Token-aligned contrast pair.\n"
        "    [bold green]--ig-steps[/bold green]  EAP-IG interpolation steps (0 = plain EAP).\n"
        "    [bold green]--top-k-edges[/bold green]  Top edges to report by absolute score (0 = all).",
        title="eap",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))
    console.print()

    console.print(Panel(
        f"[bold {ACCENT}]maxact[/bold {ACCENT}]  "
        "[dim]interpkit maxact gpt2 --at transformer.h.6.mlp --neuron 42 --texts-file corpus.txt[/dim]\n\n"
        "Max-activating examples — the feature-browsing workflow: scan a corpus and show the"
        " contexts where one unit fires hardest, with the peak token highlighted. Works for raw"
        " neurons ([bold green]--neuron[/bold green]), SAE features ([bold green]--feature[/bold green]"
        " + [bold green]--sae[/bold green]), and attention heads ([bold green]--head[/bold green])."
        " Streams batched forwards and keeps only the top-k scored contexts, so memory stays flat"
        " however large the corpus.\n\n"
        "  HF datasets work too (requires [dim]pip install 'interpkit[data]'[/dim]):\n"
        "  [dim]interpkit maxact gpt2 --at transformer.h.6.mlp --neuron 42 --dataset hf:imdb --max-examples 256[/dim]\n\n"
        "  [bold]Key options:[/bold]\n"
        "    [bold green]--texts-file / --dataset[/bold green]  Corpus: one-per-line file, or hf:name[:split[:column]].\n"
        "    [bold green]--neuron / --feature / --head[/bold green]  Which unit to scan (exactly one).\n"
        "    [bold green]--sae[/bold green]  SAE repo ID or path (with --feature).\n"
        "    [bold green]--top-k / --max-examples[/bold green]  How many results / how much corpus.",
        title="maxact",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))
    console.print()

    console.print(Panel(
        f"[bold {ACCENT}]features[/bold {ACCENT}]  "
        "[dim]interpkit features gpt2 '...' --at transformer.h.8 --sae jbloom/GPT2-Small-SAEs[/dim]\n\n"
        "Sparse Autoencoder (SAE) feature decomposition. Takes a module's activation and projects"
        " it through a separately trained SAE to recover a sparse set of interpretable features."
        " Each feature typically corresponds to a human-readable concept. Requires a compatible"
        " SAE checkpoint (HuggingFace repo or local .safetensors / .pt file).\n\n"
        "  [bold]Contrastive mode:[/bold] pass [bold green]--positive-file[/bold green] and"
        " [bold green]--negative-file[/bold green] (omit the input text argument) to find features that"
        " differentially activate between two groups of inputs.\n"
        "  [dim]interpkit features gpt2 --at transformer.h.8 --sae jbloom/... --positive-file pos.txt --negative-file neg.txt[/dim]\n\n"
        "  [bold green]--at[/bold green]   Which module's activations to decompose.\n"
        "  [bold green]--sae[/bold green]  HuggingFace repo ID or local file path of the SAE weights.\n"
        "  [bold green]--top-k[/bold green]  How many top features to display (default 20).\n"
        "  [bold green]--positive-file / --negative-file[/bold green]  Text files for contrastive feature analysis.",
        title="features",
        border_style=ACCENT_DIM,
        padding=(0, 2),
    ))

    console.print()
    console.print(
        f"  Run [bold {ACCENT}]interpkit <command> --help[/bold {ACCENT}] for the full option list of any command.\n"
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
    global _output_format, console
    _output_format = fmt
    # F-023: re-bind module-level console so it routes to stderr in JSON mode.
    console = _make_console()
    _silence_third_party_loaders()
    if ctx.invoked_subcommand is not None:
        return
    if extensive:
        _show_extensive_help()
        return

    _LOGO_STR = (
        "\u2588\u2588\u2557\u2588\u2588\u2588\u2557   \u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557  \u2588\u2588\u2557\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\n"
        "\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2551\u255a\u2550\u2550\u2588\u2588\u2554\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551 \u2588\u2588\u2554\u255d\u2588\u2588\u2551\u255a\u2550\u2550\u2588\u2588\u2554\u2550\u2550\u255d\n"
        "\u2588\u2588\u2551\u2588\u2588\u2554\u2588\u2588\u2557 \u2588\u2588\u2551   \u2588\u2588\u2551   \u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2588\u2588\u2588\u2554\u255d \u2588\u2588\u2551   \u2588\u2588\u2551\n"
        "\u2588\u2588\u2551\u2588\u2588\u2551\u255a\u2588\u2588\u2557\u2588\u2588\u2551   \u2588\u2588\u2551   \u2588\u2588\u2554\u2550\u2550\u255d  \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2550\u255d \u2588\u2588\u2554\u2550\u2588\u2588\u2557 \u2588\u2588\u2551   \u2588\u2588\u2551\n"
        "\u2588\u2588\u2551\u2588\u2588\u2551 \u255a\u2588\u2588\u2588\u2588\u2551   \u2588\u2588\u2551   \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2551     \u2588\u2588\u2551  \u2588\u2588\u2557\u2588\u2588\u2551   \u2588\u2588\u2551\n"
        "\u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u2550\u2550\u255d   \u255a\u2550\u255d   \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d     \u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d   \u255a\u2550\u255d"
    )
    console.print()
    console.print(GradientText(_LOGO_STR, colors=BRAND_COLORS, style="bold"), highlight=False)
    console.print(
        GradientText(
            f"  Mech interp for any HuggingFace model  v{_VERSION}",
            colors=BRAND_COLORS,
        )
    )

    def _cmd_table(commands: list[tuple[str, str]]) -> Table:
        table = Table(
            show_header=False, box=None, pad_edge=False,
            padding=(0, 2), expand=True,
        )
        table.add_column("Command", style=f"bold {ACCENT}", no_wrap=True, min_width=16)
        table.add_column("Description")
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        return table

    quick_start = _cmd_table([
        ("scan", "One-command overview \u2014 DLA, lens, attention, attribution"),
        ("report", "Generate an interactive HTML report"),
        ("chat", "Send a message to a chat / instruct model"),
        ("generate", "Generate with steering/ablation active + per-token lens"),
    ])

    core_ops = _cmd_table([
        ("inspect", "Module tree with types, params, roles"),
        ("dla", "Direct Logit Attribution \u2014 decompose logit by component"),
        ("trace", "Causal tracing \u2014 module or position-aware"),
        ("lens", "Logit lens \u2014 project layers to vocab (--tuned-lens for tuned)"),
        ("train-tuned-lens", "Train per-layer tuned-lens translators"),
        ("attribute", "Gradient saliency over inputs"),
        ("patch", "Activation patching at module/head/position"),
    ])

    analysis_ops = _cmd_table([
        ("activations", "Extract raw activation tensors"),
        ("ablate", "Zero/mean/resample ablate a component"),
        ("attention", "Visualize attention patterns"),
        ("decompose", "Residual stream decomposition by component"),
        ("steer", "Activation steering (contrastive vector or SAE feature)"),
        ("probe", "Linear probe on activations"),
        ("diff", "Compare two models' activations"),
    ])

    circuit_ops = _cmd_table([
        ("find-circuit", "Automated circuit discovery (ablation or EAP)"),
        ("atp", "Attribution Patching — score all modules in 3 passes"),
        ("eap", "Edge Attribution Patching — gradient-based edge scores"),
        ("features", "SAE feature decomposition (single or contrastive)"),
        ("maxact", "Max-activating examples for a neuron / SAE feature / head"),
    ])

    layout = Table(show_header=False, box=None, pad_edge=False, padding=0, expand=True)
    layout.add_column(ratio=1)

    layout.add_row(GradientRule("Quick Start", colors=BRAND_COLORS, align="left"))
    layout.add_row(quick_start)
    layout.add_row("")
    layout.add_row(GradientRule("Core Operations", colors=BRAND_COLORS, align="left"))
    layout.add_row(core_ops)
    layout.add_row("")
    layout.add_row(GradientRule("Analysis", colors=BRAND_COLORS, align="left"))
    layout.add_row(analysis_ops)
    layout.add_row("")
    layout.add_row(GradientRule("Circuit Analysis", colors=BRAND_COLORS, align="left"))
    layout.add_row(circuit_ops)

    console.print()
    console.print(layout)

    console.print()
    console.print("  [dim]\u25b8[/dim] Most commands accept [bold green]--save[/bold green] and [bold green]--html[/bold green] for exports.")
    console.print(f"  [dim]\u25b8[/dim] Run [bold {ACCENT}]interpkit <command> --help[/bold {ACCENT}] for detailed usage.")
    console.print(
        f"  [dim]\u25b8[/dim] No console script on PATH? [bold {ACCENT}]python -m interpkit[/bold {ACCENT}]"
        " works the same everywhere."
    )
    console.print(f"  [dim]\u25b8[/dim] New here? Try [bold {ACCENT}]interpkit --extensive[/bold {ACCENT}] for a plain-English walkthrough.")
    console.print()


# ══════════════════════════════════════════════════════════════════
# inspect
# ══════════════════════════════════════════════════════════════════


@app.command()
def inspect(
    model_name: str = typer.Argument(..., help="HuggingFace model ID (e.g. gpt2, microsoft/resnet-50)"),
    device: str | None = typer.Option(None, help="Device (cpu, cuda, mps). Auto-detected if omitted."),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Print the model's module tree with types, param counts, and detected roles."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    if _output_format == "json":
        # F-023: inspect previously ignored --format json. Now emits a
        # structured JSON description of the architecture.
        arch = m.arch_info
        result = {
            "model": model_name,
            "family": arch.family.value if hasattr(arch.family, "value") else str(arch.family),
            "arch_family": arch.arch_family,
            "device": m.device,
            "dtype": str(m.dtype),
            "num_layers": arch.num_layers,
            "hidden_size": arch.hidden_size,
            "num_attention_heads": arch.num_attention_heads,
            "vocab_size": arch.vocab_size,
            "is_encoder_decoder": arch.is_encoder_decoder,
            "spatial": arch.spatial,
            "head_path": arch.head_path,
            "embed_path": arch.embed_path,
            "pre_head_path": arch.pre_head_path,
            "project_out_path": arch.project_out_path,
            "blocks": [
                {"path": b.path, "stage": b.stage,
                 "has_attention": b.has_attention, "has_residual": b.has_residual}
                for b in arch.blocks
            ],
            "modules": [
                {"name": m.name, "type": m.type_name, "param_count": m.param_count, "role": m.role}
                for m in arch.modules
            ],
        }
        _json_dump(result)
        return
    with console.status("  Inspecting model..."):
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
    head: int | None = typer.Option(None, "--head", help="Specific attention head to patch (requires attention module)"),
    positions: str | None = typer.Option(None, "--positions", help="Comma-separated token positions to patch (e.g. '3,4,5')"),
    metric: str = typer.Option("logit_diff", "--metric", help="Effect metric: logit_diff, kl_div, target_prob, l2_prob"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Activation patching: swap one module's output from clean into corrupted run."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    pos_list: list[int] | None = None
    if positions is not None:
        pos_list = [int(p.strip()) for p in positions.split(",")]
    with console.status("  Patching activations..."):
        result = m.patch(clean, corrupted, at=at, head=head, positions=pos_list, metric=metric)
    console.print(f"  [bold green]Patched[/bold green] [{ACCENT}]{at}[/{ACCENT}]")
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
    save: str | None = typer.Option(None, "--save", help="Save bar chart / heatmap to file (e.g. trace.png)"),
    html_path: str | None = typer.Option(None, "--html", help="Save interactive HTML to file (e.g. trace.html)"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
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
    save: str | None = typer.Option(None, "--save", help="Save heatmap to file (e.g. lens.png)"),
    html_path: str | None = typer.Option(None, "--html", help="Save interactive HTML to file"),
    position: int | None = typer.Option(None, "--position", help="Single token position to analyse (-1 = last). Omit for all positions."),
    tuned_lens_path: str | None = typer.Option(None, "--tuned-lens", help="Path to a saved tuned lens (switches to kind='tuned'). Train with `interpkit train-tuned-lens`."),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Logit lens: project each layer's hidden state to vocabulary space.

    Pass --tuned-lens <path> to apply trained per-layer translators
    (Belrose et al. 2023) for an unbiased early-layer readout.
    """
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    kind = "tuned" if tuned_lens_path is not None else "logit"
    with console.status("  Running logit lens..."):
        result = m.lens(
            text, save=save, html=html_path, position=position,
            kind=kind, tuned_lens=tuned_lens_path,
        )
    if _output_format == "json":
        _json_dump(result if isinstance(result, dict) else {"results": result})


# ══════════════════════════════════════════════════════════════════
# train-tuned-lens
# ══════════════════════════════════════════════════════════════════


@app.command("train-tuned-lens")
def train_tuned_lens_cmd(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    corpus_file: str = typer.Option(..., "--corpus-file", help="Text file with training sentences, one per line"),
    steps: int = typer.Option(200, "--steps", help="Training steps"),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Adam learning rate"),
    max_length: int = typer.Option(64, "--max-length", help="Token truncation length"),
    seed: int = typer.Option(0, "--seed", help="Random seed (deterministic on CPU)"),
    save: str | None = typer.Option(None, "--save", help="Output directory or .safetensors path (default: ~/.cache/interpkit/tuned_lens/<model>/)"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Train tuned-lens translators (Belrose et al. 2023) for a model.

    The model stays frozen; only per-layer affine translators train.
    Use the result with `interpkit lens ... --tuned-lens <path>`.
    """
    from interpkit.core.inputs import read_examples_file
    from interpkit.ops.tuned_lens import default_tuned_lens_dir

    corpus = read_examples_file(corpus_file)
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    out = save if save is not None else str(default_tuned_lens_dir(model_name))
    lens_obj = m.train_tuned_lens(
        corpus, steps=steps, batch_size=batch_size, lr=lr,
        max_length=max_length, seed=seed, save=out,
    )
    if _output_format == "json":
        _json_dump({"saved_to": out, "meta": lens_obj.meta})


# ══════════════════════════════════════════════════════════════════
# attribute
# ══════════════════════════════════════════════════════════════════


@app.command()
def attribute(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text or image path"),
    target: int | None = typer.Option(None, "--target", help="Target class/token index for attribution"),
    method: str = typer.Option("integrated_gradients", "--method", help="Attribution method: integrated_gradients, gradient, gradient_x_input"),
    save: str | None = typer.Option(None, "--save", help="Save figure to file (e.g. attribution.png)"),
    html_path: str | None = typer.Option(None, "--html", help="Save interactive HTML to file (e.g. attribution.html)"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
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
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Extract and display activation statistics at named modules."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    modules = [s.strip() for s in at.split(",")]
    with console.status("  Extracting activations..."):
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
    reference: str | None = typer.Option(None, "--reference", help="Reference input for resample ablation"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Zero, mean, or resample ablate a module and measure the effect on output."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    with console.status("  Running ablation..."):
        result = m.ablate(input_data, at=at, method=method, reference=reference)
    console.print(f"  [bold green]Ablated[/bold green] [{ACCENT}]{at}[/{ACCENT}] ({method})")
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# attention
# ══════════════════════════════════════════════════════════════════


@app.command()
def attention(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text"),
    layer: int | None = typer.Option(None, "--layer", help="Specific layer index"),
    head: int | None = typer.Option(None, "--head", help="Specific head index"),
    save: str | None = typer.Option(None, "--save", help="Save heatmap to file (e.g. attention.png)"),
    html_path: str | None = typer.Option(None, "--html", help="Save interactive HTML to file (e.g. attention.html)"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Show attention patterns for transformer models."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    with console.status("  Computing attention patterns..."):
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
    positive: str | None = typer.Option(None, "--positive", help="Positive direction text (single example)"),
    negative: str | None = typer.Option(None, "--negative", help="Negative direction text (single example)"),
    positive_file: str | None = typer.Option(None, "--positive-file", help="Text file with positive examples, one per line"),
    negative_file: str | None = typer.Option(None, "--negative-file", help="Text file with negative examples, one per line"),
    at: str = typer.Option(..., "--at", help="Module name to apply steering at"),
    scale: float = typer.Option(2.0, "--scale", help="Steering vector scale factor (contrastive mode)"),
    sae: str | None = typer.Option(None, "--sae", help="SAE source: HuggingFace repo ID, local file path (.safetensors / .pt), or 'org/repo/subfolder' shorthand (with --feature)"),
    sae_subfolder: str | None = typer.Option(None, "--sae-subfolder", help="Subfolder inside the SAE repo (e.g. 'blocks.8.hook_resid_pre'). Equivalent to appending it to --sae."),
    feature: int | None = typer.Option(None, "--feature", help="SAE feature index to steer along (requires --sae)"),
    feature_mode: str = typer.Option("clamp", "--feature-mode", help="SAE feature steering mode: 'clamp' (pin the feature's activation — Golden Gate style) or 'add' (inject the decoder direction)"),
    strength: float = typer.Option(10.0, "--strength", help="Feature target activation (clamp) or added activation (add)"),
    save: str | None = typer.Option(None, "--save", help="Save comparison chart to file"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Apply steering during inference: a contrastive vector or an SAE feature."""
    from interpkit.core.inputs import read_examples_file

    wants_contrastive = any([positive, negative, positive_file, negative_file])
    wants_feature = feature is not None or sae is not None
    if wants_contrastive and wants_feature:
        raise typer.BadParameter(
            "--positive/--negative (contrastive steering) and --sae/--feature "
            "(SAE feature steering) are mutually exclusive."
        )
    if wants_feature and (feature is None or sae is None):
        raise typer.BadParameter("SAE feature steering requires both --sae and --feature.")

    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)

    if wants_feature:
        with console.status("  Running steered inference..."):
            result = m.steer(
                input_data, at=at, sae=sae, feature=feature,
                mode=feature_mode, strength=strength,
                sae_subfolder=sae_subfolder, save=save,
            )
        if _output_format == "json":
            _json_dump(result)
        return

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

    vector = m.steer_vector(pos_inputs, neg_inputs, at=at)
    with console.status("  Running steered inference..."):
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
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
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
    save: str | None = typer.Option(None, "--save", help="Save bar chart to file"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Compare activations between two models on the same input."""
    import interpkit

    m_a = _load_model(model_a_name, device=device, dtype=dtype, device_map=device_map)
    m_b = _load_model(model_b_name, device=device, dtype=dtype, device_map=device_map)
    with console.status("  Comparing models..."):
        result = interpkit.diff(m_a, m_b, input_data, save=save)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# features (SAE)
# ══════════════════════════════════════════════════════════════════


@app.command()
def features(
    model_name: str = typer.Argument(..., help="HuggingFace model ID (e.g. gpt2)"),
    input_data: str | None = typer.Argument(None, help="Input text (omit when using --positive-file / --negative-file)"),
    at: str = typer.Option(..., "--at", help="Module name to decompose (e.g. transformer.h.8)"),
    sae: str = typer.Option(..., "--sae", help="SAE source: HuggingFace repo ID, local file path (.safetensors / .pt), or 'org/repo/subfolder' shorthand"),
    sae_subfolder: str | None = typer.Option(None, "--sae-subfolder", help="Subfolder inside the SAE repo (e.g. 'blocks.8.hook_resid_pre'). Equivalent to appending it to --sae."),
    top_k: int = typer.Option(20, "--top-k", help="Number of top features to display"),
    positive_file: str | None = typer.Option(None, "--positive-file", help="Text file with positive examples for contrastive analysis, one per line"),
    negative_file: str | None = typer.Option(None, "--negative-file", help="Text file with negative examples for contrastive analysis, one per line"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
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
        result = m.contrastive_features(
            pos_inputs, neg_inputs, at=at, sae=sae, top_k=top_k,
            sae_subfolder=sae_subfolder,
        )
    else:
        if input_data is None:
            raise typer.BadParameter("Provide input text or use --positive-file / --negative-file for contrastive mode")
        m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
        with console.status("  Decomposing features..."):
            result = m.features(
                input_data, at=at, sae=sae, top_k=top_k,
                sae_subfolder=sae_subfolder,
            )

    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# scan
# ══════════════════════════════════════════════════════════════════


@app.command()
def scan(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    input_data: str = typer.Argument(..., help="Input text"),
    save: str | None = typer.Option(None, "--save", help="Prefix for exported figures (e.g. scan → scan_dla.png, scan_lens.png)"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
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
    token: str | None = typer.Option(None, "--token", help="Target token (string or int). Uses top-1 prediction if omitted."),
    position: int = typer.Option(-1, "--position", help="Token position to analyse (-1 = last)"),
    top_k: int = typer.Option(10, "--top-k", help="Number of top/bottom contributors to show"),
    save: str | None = typer.Option(None, "--save", help="Save bar chart to file (e.g. dla.png)"),
    html_path: str | None = typer.Option(None, "--html", help="Save interactive HTML to file"),
    sae: str | None = typer.Option(None, "--sae", help="SAE source: HuggingFace repo ID, local file path (.safetensors / .pt), or 'org/repo/subfolder' shorthand"),
    sae_at: str | None = typer.Option(None, "--sae-at", help="Module to decompose through the SAE (e.g. transformer.h.11.attn)"),
    sae_subfolder: str | None = typer.Option(None, "--sae-subfolder", help="Subfolder inside the SAE repo (e.g. 'blocks.8.hook_resid_pre'). Equivalent to appending it to --sae."),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Direct Logit Attribution: decompose output logits by component."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    parsed_token: int | str | None = None
    if token is not None:
        try:
            parsed_token = int(token)
        except ValueError:
            parsed_token = token
    with console.status("  Running DLA..."):
        result = m.dla(
            input_data, token=parsed_token, position=position,
            top_k=top_k, save=save, html=html_path,
            sae=sae, sae_at=sae_at, sae_subfolder=sae_subfolder,
        )
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
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Decompose the residual stream into per-component contributions."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    with console.status("  Decomposing residual stream..."):
        result = m.decompose(input_data, position=position)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# find-circuit
# ══════════════════════════════════════════════════════════════════


@app.command("find-circuit")
def find_circuit(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    clean: str | None = typer.Option(None, "--clean", help="Clean input text (single example)"),
    corrupted: str | None = typer.Option(None, "--corrupted", help="Corrupted input text (single example)"),
    clean_file: str | None = typer.Option(None, "--clean-file", help="Text file with clean examples, one per line"),
    corrupted_file: str | None = typer.Option(None, "--corrupted-file", help="Text file with corrupted examples, one per line (must match --clean-file line count)"),
    threshold: float = typer.Option(0.01, "--threshold", help="Minimum ablation effect to include in circuit (0-1)"),
    method: str = typer.Option("mean", "--method", help="Selection method: mean (default), zero, resample (ablation), or eap / eap-ig (gradient-based, much faster)"),
    metric: str = typer.Option("logit_diff", "--metric", help="Effect metric: logit_diff, kl_div, target_prob, l2_prob"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
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
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Generate a comprehensive HTML report: prediction, DLA, logit lens, attention, attribution."""
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.report(input_data, save=save)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# chat
# ══════════════════════════════════════════════════════════════════


@app.command()
def chat(
    model_name: str = typer.Argument(..., help="HuggingFace chat/instruct model ID (e.g. HuggingFaceTB/SmolLM2-360M-Instruct)"),
    message: str = typer.Argument(..., help="User message to send"),
    system: str | None = typer.Option(None, "--system", help="Optional system prompt"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens", help="Max generation length"),
    sample: bool = typer.Option(False, "--sample/--no-sample", help="Sample (True) or use greedy decoding (False, default)"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature (used when --sample)"),
    top_p: float = typer.Option(1.0, "--top-p", help="Nucleus sampling cutoff (used when --sample)"),
    show_prompt: bool = typer.Option(False, "--show-prompt", help="Print the chat-templated prompt before generating"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Send a chat message and print the model's response.

    Routes the message through the tokenizer's chat template
    (``apply_chat_template`` with ``add_generation_prompt=True``) and
    calls ``model.generate``.  Errors clearly when the loaded model has
    no chat template (i.e. is a base/non-instruct model).
    """
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    with console.status("  Generating response..."):
        result = m.chat(
            message,
            system=system,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature,
            top_p=top_p,
        )

    if show_prompt:
        console.print(Panel(
            result["prompt"],
            title="[bold]Prompt[/bold]",
            border_style=ACCENT_DIM,
            padding=(0, 1),
        ))

    console.print()
    console.print(Panel(
        result["response"],
        title=f"[bold]{model_name}[/bold]",
        border_style=ACCENT,
        padding=(0, 2),
    ))

    if _output_format == "json":
        _json_dump({k: v for k, v in result.items() if k not in {"input_ids", "output_ids"}})


# ══════════════════════════════════════════════════════════════════
# atp / eap
# ══════════════════════════════════════════════════════════════════


@app.command()
def atp(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    clean: str = typer.Option(..., "--clean", help="Clean input"),
    corrupted: str = typer.Option(..., "--corrupted", help="Corrupted input"),
    top_k: int = typer.Option(20, "--top-k", help="Top modules to report by absolute score. 0 = all."),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Attribution Patching: first-order patch-effect scores for every module.

    Three model passes score all modules at once — the fast first look
    before committing to `trace`'s per-module full patching.
    """
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    effective_top_k: int | None = top_k if top_k > 0 else None
    with console.status("  Computing attribution patching scores..."):
        result = m.atp(clean, corrupted, top_k=effective_top_k)
    if _output_format == "json":
        _json_dump(result)


@app.command()
def eap(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    clean: str = typer.Option(..., "--clean", help="Clean input (must tokenize to same length as --corrupted)"),
    corrupted: str = typer.Option(..., "--corrupted", help="Corrupted input"),
    ig_steps: int = typer.Option(0, "--ig-steps", help="EAP-IG interpolation steps (0 = plain EAP; try 5)"),
    top_k_edges: int = typer.Option(30, "--top-k-edges", help="Top edges to report by absolute score. 0 = all."),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Edge Attribution Patching: gradient-based edge scores for circuit discovery.

    Scores every (component → residual stream) edge from a handful of
    passes. Pair with `find-circuit --method eap` for a causally
    verified circuit.
    """
    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    effective_top_k: int | None = top_k_edges if top_k_edges > 0 else None
    with console.status("  Computing edge attribution scores..."):
        result = m.eap(clean, corrupted, ig_steps=ig_steps, top_k_edges=effective_top_k)
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# maxact
# ══════════════════════════════════════════════════════════════════


@app.command()
def maxact(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    at: str = typer.Option(..., "--at", help="Module whose activations to scan (e.g. transformer.h.6.mlp)"),
    texts_file: str | None = typer.Option(None, "--texts-file", help="Text file with one example per line"),
    dataset: str | None = typer.Option(None, "--dataset", help="HF dataset spec: hf:name[:split[:column]] (needs interpkit[data] + --max-examples)"),
    neuron: int | None = typer.Option(None, "--neuron", help="Neuron index at the module (raw activation score)"),
    feature: int | None = typer.Option(None, "--feature", help="SAE feature index (requires --sae)"),
    head: int | None = typer.Option(None, "--head", help="Attention head index (pre-projection output norm)"),
    sae: str | None = typer.Option(None, "--sae", help="SAE repo ID or local path (with --feature)"),
    top_k: int = typer.Option(20, "--top-k", help="Top examples to keep"),
    batch_size: int = typer.Option(8, "--batch-size", help="Forward batch size"),
    max_examples: int | None = typer.Option(None, "--max-examples", help="Cap on dataset examples scanned"),
    max_length: int = typer.Option(128, "--max-length", help="Token truncation length"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Find the dataset examples that most activate a neuron / SAE feature / head.

    The feature-browsing workflow: "what does this unit fire on?".
    Streams batched forwards and keeps only the top-k scored contexts.
    """
    from interpkit.core.inputs import read_examples_file

    if (texts_file is None) == (dataset is None):
        raise typer.BadParameter("Provide exactly one of --texts-file or --dataset.")
    data: list[str] | str = (
        read_examples_file(texts_file) if texts_file is not None else dataset  # type: ignore[assignment]
    )

    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)
    result = m.max_activating(
        data, at=at,
        neuron=neuron, feature=feature, head=head, sae=sae,
        top_k=top_k, batch_size=batch_size, max_examples=max_examples,
        max_length=max_length,
    )
    if _output_format == "json":
        _json_dump(result)


# ══════════════════════════════════════════════════════════════════
# generate
# ══════════════════════════════════════════════════════════════════


@app.command()
def generate(
    model_name: str = typer.Argument(..., help="HuggingFace model ID"),
    prompt: str = typer.Argument(..., help="Prompt text to generate from"),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", help="Max generation length"),
    positive: str | None = typer.Option(None, "--positive", help="Positive steering text (single example)"),
    negative: str | None = typer.Option(None, "--negative", help="Negative steering text (single example)"),
    positive_file: str | None = typer.Option(None, "--positive-file", help="Text file with positive examples, one per line"),
    negative_file: str | None = typer.Option(None, "--negative-file", help="Text file with negative examples, one per line"),
    at: str | None = typer.Option(None, "--at", help="Module to apply steering at (required with --positive/--negative or --sae/--feature)"),
    scale: float = typer.Option(2.0, "--scale", help="Steering vector scale factor"),
    sae: str | None = typer.Option(None, "--sae", help="SAE source: HuggingFace repo ID, local file path (.safetensors / .pt), or 'org/repo/subfolder' shorthand (with --feature)"),
    sae_subfolder: str | None = typer.Option(None, "--sae-subfolder", help="Subfolder inside the SAE repo (e.g. 'blocks.8.hook_resid_pre'). Equivalent to appending it to --sae."),
    feature: int | None = typer.Option(None, "--feature", help="SAE feature index to steer along during generation (requires --sae)"),
    feature_mode: str = typer.Option("clamp", "--feature-mode", help="SAE feature steering mode: 'clamp' (pin the feature's activation — Golden Gate style) or 'add' (inject the decoder direction)"),
    strength: float = typer.Option(10.0, "--strength", help="Feature target activation (clamp) or added activation (add)"),
    ablate_at: str | None = typer.Option(None, "--ablate-at", help="Module to ablate during generation"),
    ablate_method: str = typer.Option("zero", "--ablate-method", help="Ablation method: zero, mean"),
    capture: str | None = typer.Option(None, "--capture", help="Per-token capture: 'lens' (logit-lens trajectory) or 'logits'"),
    sample: bool = typer.Option(False, "--sample/--no-sample", help="Sample (True) or use greedy decoding (False, default)"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature (used when --sample)"),
    top_p: float = typer.Option(1.0, "--top-p", help="Nucleus sampling cutoff (used when --sample)"),
    device: str | None = typer.Option(None, help="Device"),
    dtype: str | None = typer.Option(None, "--dtype", help="Model dtype: float16, bfloat16, float32, auto"),
    device_map: str | None = typer.Option(None, "--device-map", help="HF device_map (e.g. 'auto')"),
) -> None:
    """Generate text with interventions active across every decode step.

    Steering — contrastive (``--positive`` / ``--negative`` + ``--at``) or
    SAE-feature (``--sae`` + ``--feature`` + ``--at``, the Golden Gate
    manipulation) — and ablation (``--ablate-at``) stay hooked for the
    prefill and all KV-cached decode steps — the generation-time
    counterpart of ``steer`` / ``ablate``. ``--capture lens`` additionally
    records each generated token's logit-lens trajectory through every
    block.
    """
    from interpkit.core.inputs import read_examples_file
    from interpkit.core.interventions import (
        AblateIntervention,
        SAEFeatureIntervention,
        SteerIntervention,
    )

    wants_steering = any([positive, negative, positive_file, negative_file])
    wants_feature = feature is not None or sae is not None
    if wants_steering and wants_feature:
        raise typer.BadParameter(
            "--positive/--negative (contrastive steering) and --sae/--feature "
            "(SAE feature steering) are mutually exclusive."
        )
    if wants_feature and (feature is None or sae is None):
        raise typer.BadParameter("SAE feature steering requires both --sae and --feature.")
    if (wants_steering or wants_feature) and at is None:
        raise typer.BadParameter("Steering requires --at (module to apply it at).")

    m = _load_model(model_name, device=device, dtype=dtype, device_map=device_map)

    interventions: list = []
    if wants_feature:
        from interpkit.ops.sae import _ensure_sae_on_device, load_sae

        assert at is not None and sae is not None and feature is not None
        loaded_sae = _ensure_sae_on_device(
            load_sae(sae, device=m._device, subfolder=sae_subfolder), m._device,
        )
        interventions.append(SAEFeatureIntervention(
            at, sae=loaded_sae, feature=feature, strength=strength, mode=feature_mode,
        ))
    if wants_steering:
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

        assert at is not None
        vector = m.steer_vector(pos_inputs, neg_inputs, at=at)
        interventions.append(SteerIntervention(at, vector=vector, scale=scale))

    if ablate_at is not None:
        interventions.append(AblateIntervention(ablate_at, method=ablate_method))

    with console.status("  Generating..."):
        result = m.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            interventions=interventions or None,
            capture=capture,
            do_sample=sample,
            temperature=temperature,
            top_p=top_p,
        )

    if _output_format == "json":
        # Trim tensors: ids ride along in the Python API but bloat JSON, and
        # per-step logits are (1, vocab) each.
        out = {k: v for k, v in result.items() if k not in {"input_ids", "output_ids"}}
        if "steps" in out:
            out["steps"] = [
                {k: v for k, v in step.items() if k != "logits"}
                for step in out["steps"]
            ]
        _json_dump(out)


# ══════════════════════════════════════════════════════════════════
# gui
# ══════════════════════════════════════════════════════════════════


@app.command()
def gui(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind the server to"),
    port: int = typer.Option(7860, "--port", help="Port to serve the GUI on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open the browser automatically"),
) -> None:
    """Launch the local web GUI: load models and run every op from the browser.

    Requires the optional [gui] extra: pip install "interpkit[gui]".
    """
    try:
        from interpkit.gui import serve
    except ImportError:
        console.print(
            "\n  [bold red]The GUI requires the optional \\[gui] extra.[/bold red]\n"
            f'  Install it with: [bold {ACCENT}]pip install "interpkit\\[gui]"[/bold {ACCENT}]\n'
        )
        raise typer.Exit(1) from None
    serve(host=host, port=port, open_browser=not no_browser)


def run() -> None:
    """CLI entry point that renders interpkit's intentional errors cleanly.

    The ``InterpkitError`` family (e.g. ``OperationNotSupportedForArchitecture``,
    ``WrongInputType``, ``LensPipelineMismatch``) is the project's fail-loud
    contract — these are clear, actionable, user-facing messages, not crashes.
    Presenting them as a Python traceback undermines that, so we catch them at
    the boundary and print a single clean line (JSON object in ``--format json``)
    + exit non-zero. Unexpected exceptions still propagate as a normal traceback.
    """
    from interpkit.core.exceptions import InterpkitError

    try:
        app()
    except (InterpkitError, ValueError, KeyError, IndexError) as exc:
        # interpkit's user-facing validation failures: unsupported op / wrong
        # input type (InterpkitError family), empty input (ValueError), unknown
        # module path (KeyError with a "did you mean" hint), out-of-range
        # position (ValueError / IndexError). These are clear, actionable
        # messages — render one line, not a traceback. Genuine internal bugs
        # raise other types (RuntimeError, TypeError, …) and still surface a
        # full traceback. ``KeyError.__str__`` wraps the message in quotes, so
        # pull ``args[0]`` for it.
        msg = exc.args[0] if (isinstance(exc, KeyError) and exc.args) else str(exc)
        if _output_format == "json":
            import json as _json

            print(_json.dumps({"error": type(exc).__name__, "message": str(msg)}))
        else:
            Console(file=_sys.stderr).print(f"[bold red]Error:[/bold red] {msg}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    run()
