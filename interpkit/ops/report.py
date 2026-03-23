"""report — generate a comprehensive HTML report for a model on a given input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_report(
    model: "Model",
    input_data: Any,
    *,
    save: str = "report.html",
) -> dict[str, Any]:
    """Run prediction, DLA, logit lens, attention, and attribution, then
    generate a single self-contained HTML document combining all results.

    Returns a dict with the individual section results and the HTML path.
    """
    import html as _html

    from interpkit.core.html import (
        _ACCENT,
        _DARK_BG,
        _DIM_TEXT,
        _GREEN,
        _HIGHLIGHT,
        _PANEL_BG,
        _TEXT,
        html_attribution,
        html_dla,
        html_lens,
        save_html,
    )

    sections: dict[str, Any] = {}

    # 1. Prediction
    console.print("  [dim]Running prediction...[/dim]")
    import torch

    predictions = []
    try:
        with torch.no_grad():
            model_input = model._prepare(input_data)
            logits = model._forward(model_input)

        if logits.dim() == 3:
            last_logits = logits[0, -1, :]
        elif logits.dim() == 2:
            last_logits = logits[0]
        else:
            last_logits = logits.view(-1)

        probs = torch.softmax(last_logits.float(), dim=-1)
        top5_vals, top5_ids = probs.topk(5)
        for tid, p in zip(top5_ids.tolist(), top5_vals.tolist()):
            tok_str = model._tokenizer.decode([tid]) if model._tokenizer else str(tid)
            predictions.append({"token": tok_str, "token_id": tid, "probability": round(p, 4)})
    except Exception:
        pass
    sections["prediction"] = predictions

    # 2. DLA
    console.print("  [dim]Running DLA...[/dim]")
    try:
        dla_result = model.dla(input_data)
        sections["dla"] = dla_result
    except Exception:
        sections["dla"] = None

    # 3. Logit Lens
    console.print("  [dim]Running logit lens...[/dim]")
    try:
        lens_result = model.lens(input_data)
        sections["lens"] = lens_result
    except Exception:
        sections["lens"] = None

    # 4. Attention
    console.print("  [dim]Running attention analysis...[/dim]")
    try:
        attn_result = model.attention(input_data)
        sections["attention"] = attn_result
    except Exception:
        sections["attention"] = None

    # 5. Attribution
    console.print("  [dim]Running attribution...[/dim]")
    try:
        attr_result = model.attribute(input_data, method="integrated_gradients")
        sections["attribution"] = attr_result
    except Exception:
        sections["attribution"] = None

    # Build combined HTML
    model_name = type(model._model).__name__
    input_preview = str(input_data)[:200]

    nav_items = [
        ("prediction", "Prediction"),
        ("dla", "DLA"),
        ("lens", "Logit Lens"),
        ("attention", "Attention"),
        ("attribution", "Attribution"),
    ]

    nav_html = "\n".join(
        f'<a href="#{sid}" style="display:block;padding:6px 12px;color:{_DIM_TEXT};'
        f'text-decoration:none;border-left:3px solid transparent;margin:2px 0"'
        f' onmouseover="this.style.borderLeftColor=\'{_HIGHLIGHT}\'"'
        f' onmouseout="this.style.borderLeftColor=\'transparent\'">{label}</a>'
        for sid, label in nav_items
    )

    # Prediction section
    pred_rows = "\n".join(
        f'<tr><td style="font-weight:600">{_html.escape(p["token"])}</td>'
        f'<td><div style="width:{p["probability"]*100:.1f}%;height:18px;background:{_GREEN};'
        f'border-radius:3px;min-width:2px"></div></td>'
        f'<td style="text-align:right">{p["probability"]:.4f}</td></tr>'
        for p in predictions
    )
    pred_section = f"""
<div id="prediction" class="panel">
    <h2>Top-5 Predictions</h2>
    <table><tr><th>Token</th><th style="width:50%">Probability</th><th>Value</th></tr>
    {pred_rows}
    </table>
</div>
"""

    # DLA section
    if sections["dla"] is not None:
        dla_inner = html_dla(sections["dla"])
        # Extract just the body content between <body> and </body>
        dla_body = _extract_body(dla_inner)
        dla_section = f'<div id="dla" class="panel">{dla_body}</div>'
    else:
        dla_section = '<div id="dla" class="panel"><h2>DLA</h2><p>Not available for this model.</p></div>'

    # Lens section
    if sections["lens"]:
        input_tokens = None
        if model._tokenizer and isinstance(input_data, str):
            enc = model._tokenizer(input_data, return_tensors="pt")
            input_tokens = model._tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

        flat_preds = []
        for pred in sections["lens"]:
            import re as _re_report
            _lm = _re_report.search(r"\.(\d+)", pred.get("layer_name", ""))
            li = int(_lm.group(1)) if _lm else 0
            for pos_data in pred.get("positions", []):
                flat_preds.append({
                    "layer": li,
                    "position": pos_data.get("pos", 0),
                    "prediction": pos_data.get("top1_token", "?"),
                    "prob": pos_data.get("top1_prob", 0.0),
                })
        lens_inner = html_lens(flat_preds, input_tokens)
        lens_body = _extract_body(lens_inner)
        lens_section = f'<div id="lens" class="panel">{lens_body}</div>'
    else:
        lens_section = '<div id="lens" class="panel"><h2>Logit Lens</h2><p>Not available.</p></div>'

    # Attention section
    if sections["attention"] is not None:
        attn_summary = f'<p>{len(sections["attention"])} head(s) captured.</p>'
        attn_section = f'<div id="attention" class="panel"><h2>Attention</h2>{attn_summary}</div>'
    else:
        attn_section = '<div id="attention" class="panel"><h2>Attention</h2><p>Not available.</p></div>'

    # Attribution section
    if sections["attribution"] is not None:
        attr_inner = html_attribution(
            sections["attribution"].get("tokens", []),
            sections["attribution"].get("scores", []),
        )
        attr_body = _extract_body(attr_inner)
        attr_section = f'<div id="attribution" class="panel">{attr_body}</div>'
    else:
        attr_section = '<div id="attribution" class="panel"><h2>Attribution</h2><p>Not available.</p></div>'

    full_body = f"""
<div style="display:flex;gap:24px">
    <nav style="min-width:160px;position:sticky;top:24px;align-self:flex-start;
                background:{_PANEL_BG};border-radius:8px;padding:12px;border:1px solid {_ACCENT}">
        <h2 style="margin:0 0 8px;font-size:1em">Sections</h2>
        {nav_html}
    </nav>
    <main style="flex:1;max-width:calc(100% - 200px)">
        <h1>InterpKit Report</h1>
        <p class="subtitle">Model: <strong>{_html.escape(model_name)}</strong> |
            Input: <em>{_html.escape(input_preview)}</em></p>
        {pred_section}
        {dla_section}
        {lens_section}
        {attn_section}
        {attr_section}
    </main>
</div>
"""

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InterpKit Report — {_html.escape(model_name)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: {_DARK_BG};
    color: {_TEXT};
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    padding: 24px;
    line-height: 1.5;
}}
h1 {{ color: {_HIGHLIGHT}; margin-bottom: 8px; font-size: 1.6em; }}
h2 {{ color: {_GREEN}; margin: 16px 0 8px; font-size: 1.2em; }}
.subtitle {{ color: {_DIM_TEXT}; margin-bottom: 20px; }}
.panel {{
    background: {_PANEL_BG};
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid {_ACCENT};
}}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid {_ACCENT}; }}
th {{ color: {_GREEN}; font-weight: 600; }}
</style>
</head>
<body>
{full_body}
</body>
</html>"""

    save_html(page, save)

    sections["html_path"] = save
    return sections


def _extract_body(full_html: str) -> str:
    """Extract content between <body> and </body> tags."""
    start = full_html.find("<body>")
    end = full_html.find("</body>")
    if start != -1 and end != -1:
        return full_html[start + 6 : end]
    return full_html
