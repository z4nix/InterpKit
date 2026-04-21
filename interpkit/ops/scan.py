"""scan — automated multi-analysis that surfaces the most interesting findings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich_gradient import Rule as GradientRule

from interpkit.core.theme import ACCENT, BRAND_COLORS, SECTION_STYLE

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_scan(
    model: Model,
    input_data: Any,
    *,
    save: str | None = None,
) -> dict[str, Any]:
    """Automated model analysis: runs DLA, logit lens, attention, and attribution.

    Collects results from each analysis, then prints a synthesised summary
    highlighting the most noteworthy findings (e.g. dominant heads,
    layer convergence in logit lens, concentrated attention patterns).

    Parameters
    ----------
    save:
        If provided, used as a prefix for exported figures
        (e.g. ``"scan"`` → ``scan_dla.png``, ``scan_lens.png``, etc.).
    """
    from interpkit.core.inputs import _is_message_list

    arch = model.arch_info
    is_lm = arch.is_language_model and arch.unembedding_name is not None
    has_tokenizer = model._tokenizer is not None
    is_str_input = isinstance(input_data, str)
    is_msgs_input = _is_message_list(input_data)
    is_text = is_str_input or is_msgs_input

    results: dict[str, Any]
    if is_str_input:
        results = {"input": input_data}
    elif is_msgs_input:
        results = {"input": input_data, "input_type": "messages"}
    else:
        results = {"input": "<tensor>"}
    findings: list[dict[str, Any]] = []

    # Determine save paths
    def _save(suffix: str) -> str | None:
        if save is None:
            return None
        return f"{save}_{suffix}.png"

    # ------------------------------------------------------------------
    # 1. Model prediction
    # ------------------------------------------------------------------
    if is_lm and has_tokenizer and is_text:
        assert model._tokenizer is not None
        model_input = model._prepare(input_data)
        with torch.no_grad():
            logits = model._forward(model_input)
        if logits.dim() == 3:
            last_logits = logits[0, -1, :]
        else:
            last_logits = logits[0]
        probs = torch.softmax(last_logits.float(), dim=-1)
        top5_probs, top5_ids = probs.topk(5)
        top5_tokens = [model._tokenizer.decode([tid]) for tid in top5_ids.tolist()]
        top5_probs_list = top5_probs.tolist()

        results["prediction"] = {
            "top5_tokens": top5_tokens,
            "top5_probs": top5_probs_list,
        }

        findings.append({
            "section": "prediction",
            "text": f"Top prediction: \"{top5_tokens[0]}\" ({top5_probs_list[0]:.1%})",
            "importance": top5_probs_list[0],
        })

    # ------------------------------------------------------------------
    # 2. Direct Logit Attribution
    # ------------------------------------------------------------------
    console.print(f"  [{ACCENT}]\u25b8[/{ACCENT}] Running DLA...")
    if is_lm and has_tokenizer and arch.num_attention_heads:
        try:
            from interpkit.ops.dla import run_dla

            dla_result = run_dla(
                model, input_data, top_k=5, save=_save("dla"),
            )
            results["dla"] = dla_result

            contribs = dla_result.get("contributions", [])
            head_contribs = dla_result.get("head_contributions", [])

            if contribs:
                top_comp = contribs[0]
                findings.append({
                    "section": "dla",
                    "text": (
                        f"Top contributor to \"{dla_result['target_token']}\": "
                        f"{top_comp['component']} ({top_comp['logit_contribution']:+.3f})"
                    ),
                    "importance": abs(top_comp["logit_contribution"]),
                })

            if head_contribs:
                top_head = head_contribs[0]
                findings.append({
                    "section": "dla",
                    "text": (
                        f"Top attention head: {top_head['component']} "
                        f"({top_head['logit_contribution']:+.3f})"
                    ),
                    "importance": abs(top_head["logit_contribution"]),
                })

        except Exception as exc:
            console.print(f"    [dim yellow]\u2022 DLA skipped ({type(exc).__name__}: {exc})[/dim yellow]")

    # ------------------------------------------------------------------
    # 3. Logit lens
    # ------------------------------------------------------------------
    console.print(f"  [{ACCENT}]\u25b8[/{ACCENT}] Running logit lens...")
    if is_lm and has_tokenizer:
        try:
            from interpkit.ops.lens import run_lens

            lens_result = run_lens(model, input_data, save=_save("lens"))
            results["lens"] = lens_result

            if lens_result:
                first_correct_layer = None
                final_token = lens_result[-1]["top1_token"] if lens_result else None
                for i, pred in enumerate(lens_result):
                    if pred["top1_token"] == final_token and pred["top1_prob"] > 0.1:
                        first_correct_layer = i
                        break

                if first_correct_layer is not None:
                    total_layers = len(lens_result)
                    findings.append({
                        "section": "lens",
                        "text": (
                            f"Answer \"{final_token}\" first appears at layer "
                            f"{first_correct_layer}/{total_layers} "
                            f"({lens_result[first_correct_layer]['layer_name']})"
                        ),
                        "importance": 1.0 - (first_correct_layer / total_layers),
                    })
        except Exception as exc:
            console.print(f"    [dim yellow]\u2022 Logit lens skipped ({type(exc).__name__}: {exc})[/dim yellow]")

    # ------------------------------------------------------------------
    # 4. Attention patterns
    # ------------------------------------------------------------------
    console.print(f"  [{ACCENT}]\u25b8[/{ACCENT}] Running attention analysis...")
    attn_modules = [m for m in arch.modules if m.role == "attention"]
    if attn_modules and is_text:
        try:
            from interpkit.ops.attention import run_attention

            attn_result = run_attention(model, input_data, save=_save("attn"))
            results["attention"] = attn_result

            if attn_result:
                # Find heads with lowest entropy (most focused attention)
                focused = sorted(attn_result, key=lambda r: r.get("entropy", float("inf")))
                if focused:
                    best = focused[0]
                    findings.append({
                        "section": "attention",
                        "text": (
                            f"Most focused attention: L{best['layer']}.H{best['head']} "
                            f"(entropy {best['entropy']:.2f})"
                        ),
                        "importance": 1.0 / (1.0 + best["entropy"]),
                    })

                    top_pairs = best.get("top_pairs", [])
                    if top_pairs and model._tokenizer and is_str_input:
                        enc = model._tokenizer(input_data, return_tensors="pt")
                        toks = model._tokenizer.convert_ids_to_tokens(
                            enc["input_ids"][0].tolist()
                        )
                        src, tgt, weight = top_pairs[0]
                        if src < len(toks) and tgt < len(toks):
                            findings.append({
                                "section": "attention",
                                "text": (
                                    f"  → attends \"{toks[src]}\" → \"{toks[tgt]}\" "
                                    f"(weight {weight:.2f})"
                                ),
                                "importance": weight,
                            })
        except Exception as exc:
            console.print(f"    [dim yellow]\u2022 Attention skipped ({type(exc).__name__}: {exc})[/dim yellow]")

    # ------------------------------------------------------------------
    # 5. Attribution
    # ------------------------------------------------------------------
    console.print(f"  [{ACCENT}]\u25b8[/{ACCENT}] Running attribution...")
    if is_text and has_tokenizer:
        try:
            from interpkit.ops.attribute import run_attribute

            attr_result = run_attribute(
                model, input_data, method="gradient", save=_save("attr"),
            )
            results["attribution"] = attr_result

            if isinstance(attr_result, dict) and "tokens" in attr_result:
                scores = attr_result["scores"]
                tokens = attr_result["tokens"]
                if scores:
                    max_idx = max(range(len(scores)), key=lambda i: abs(scores[i]))
                    findings.append({
                        "section": "attribution",
                        "text": (
                            f"Most salient input token: \"{tokens[max_idx]}\" "
                            f"(score {scores[max_idx]:.3f})"
                        ),
                        "importance": abs(scores[max_idx]) / (max(abs(s) for s in scores) or 1.0),
                    })
        except Exception as exc:
            console.print(f"    [dim yellow]\u2022 Attribution skipped ({type(exc).__name__}: {exc})[/dim yellow]")

    # ------------------------------------------------------------------
    # Synthesised summary
    # ------------------------------------------------------------------
    results["findings"] = findings
    _render_scan(results)

    return results


def _render_scan(results: dict[str, Any]) -> None:
    """Print the scan summary with key findings."""
    findings = results.get("findings", [])
    findings.sort(key=lambda f: f.get("importance", 0), reverse=True)

    input_value = results.get("input", "")
    if isinstance(input_value, str):
        input_str = input_value
    elif isinstance(input_value, list):
        parts = []
        for msg in input_value:
            if isinstance(msg, dict):
                role = msg.get("role", "?")
                content = str(msg.get("content", ""))
                parts.append(f"[{role}] {content}")
        input_str = " | ".join(parts)
    else:
        input_str = str(input_value)
    if len(input_str) > 60:
        input_str = input_str[:57] + "..."

    console.print()
    console.print(GradientRule("Scan Summary", colors=BRAND_COLORS, align="left"))

    if "prediction" in results:
        pred = results["prediction"]
        pred_parts = []
        for tok, prob in zip(pred["top5_tokens"][:3], pred["top5_probs"][:3]):
            pred_parts.append(f'[bold]"{tok}"[/bold] [dim]({prob:.1%})[/dim]')
        preds_str = ", ".join(pred_parts)
        console.print(f"\n  Predictions: {preds_str}")

    if findings:
        console.print()
        table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
        table.add_column("N", justify="right", style="dim", width=4)
        table.add_column("Tag", width=14)
        table.add_column("Finding")

        for i, f in enumerate(findings[:8], 1):
            section = f["section"]
            style = SECTION_STYLE.get(section, "dim")
            tag = f"[{style}] {section} [/{style}]"
            table.add_row(str(i), tag, f["text"])

        console.print(table)

        top_finding = findings[0]
        console.print()
        console.print(Panel(
            f"{top_finding['text']}",
            title="[bold]Top finding[/bold]",
            border_style=ACCENT,
            padding=(0, 2),
            expand=False,
        ))
    else:
        console.print("\n  [dim]No significant findings detected.[/dim]")

    analyses_run = [k for k in ("dla", "lens", "attention", "attribution") if k in results]
    analyses_str = " \u00b7 ".join(analyses_run)
    console.print(f'\n  [dim]Analyses: {analyses_str}  |  Input: "{input_str}"[/dim]')
    console.print()
