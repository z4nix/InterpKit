"""Interactive HTML visualization generators — self-contained files with inline JS/CSS."""

from __future__ import annotations

import html
import json
from typing import Any


_DARK_BG = "#1a1a2e"
_PANEL_BG = "#16213e"
_ACCENT = "#0f3460"
_HIGHLIGHT = "#e94560"
_TEXT = "#eee"
_DIM_TEXT = "#aaa"
_GREEN = "#53d769"


def _wrap_page(title: str, body: str, *, extra_css: str = "", extra_js: str = "") -> str:
    """Wrap body HTML in a full self-contained page with dark theme."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
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
.heatmap-cell {{
    width: 28px; height: 28px;
    display: inline-block;
    text-align: center;
    font-size: 10px;
    line-height: 28px;
    cursor: pointer;
    border-radius: 2px;
    position: relative;
}}
.tooltip {{
    position: fixed;
    background: #222;
    color: {_TEXT};
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    z-index: 1000;
    display: none;
    box-shadow: 0 2px 8px rgba(0,0,0,.4);
}}
.bar {{
    height: 22px;
    border-radius: 3px;
    display: inline-block;
    vertical-align: middle;
    min-width: 2px;
    transition: width 0.3s;
}}
.controls {{ margin-bottom: 16px; }}
select, input[type=range] {{ margin: 0 8px; }}
select {{
    background: {_ACCENT};
    color: {_TEXT};
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
}}
label {{ color: {_DIM_TEXT}; font-size: 0.9em; }}
.token {{
    display: inline-block;
    padding: 2px 4px;
    margin: 1px;
    border-radius: 3px;
    cursor: pointer;
    transition: background 0.2s;
}}
.filter-btn {{
    background: {_ACCENT};
    color: {_TEXT};
    border: none;
    padding: 4px 12px;
    border-radius: 4px;
    cursor: pointer;
    margin: 2px;
    font-size: 0.85em;
}}
.filter-btn.active {{ background: {_HIGHLIGHT}; }}
{extra_css}
</style>
</head>
<body>
{body}
<div class="tooltip" id="tooltip"></div>
<script>
const tooltip = document.getElementById('tooltip');
function showTip(e, text) {{
    tooltip.textContent = text;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY - 30) + 'px';
}}
function hideTip() {{ tooltip.style.display = 'none'; }}
{extra_js}
</script>
</body>
</html>"""


def html_attention(
    attention_data: list[dict[str, Any]],
    tokens: list[str] | None,
) -> str:
    """Generate an interactive attention heatmap HTML page.

    Each head is rendered as a grid. Click a head to expand it.
    Hover cells for exact attention scores. Dropdown to select layer.
    """
    if not attention_data:
        return _wrap_page("Attention", "<h1>Attention</h1><p>No attention data.</p>")

    layers: dict[int, list[dict]] = {}
    for entry in attention_data:
        layers.setdefault(entry["layer"], []).append(entry)

    tok_labels = tokens or [str(i) for i in range(max(len(entry.get("weights", [])) for entry in attention_data))]
    tok_json = json.dumps([html.escape(t) for t in tok_labels])
    data_json = json.dumps({
        str(layer): [
            {
                "head": e["head"],
                "weights": [[round(float(w), 4) for w in row] for row in e.get("weights", [])],
                "entropy": round(e.get("entropy", 0.0), 3),
            }
            for e in heads
        ]
        for layer, heads in sorted(layers.items())
    })
    layer_options = "".join(
        f'<option value="{layer}">Layer {layer}</option>' for layer in sorted(layers.keys())
    )

    body = f"""
<h1>Attention Patterns</h1>
<div class="controls panel">
    <label>Layer: <select id="layerSelect" onchange="renderLayer(this.value)">
        {layer_options}
    </select></label>
</div>
<div id="headsContainer"></div>
"""

    js = f"""
const DATA = {data_json};
const TOKENS = {tok_json};

function colorForWeight(w) {{
    const r = Math.round(233 * w + 22 * (1-w));
    const g = Math.round(69 * w + 33 * (1-w));
    const b = Math.round(96 * w + 62 * (1-w));
    return `rgb(${{r}},${{g}},${{b}})`;
}}

function renderLayer(layer) {{
    const container = document.getElementById('headsContainer');
    const heads = DATA[layer] || [];
    let html = '';
    for (const h of heads) {{
        const n = h.weights.length;
        let grid = '<div style="display:inline-grid;grid-template-columns:auto ' +
            'repeat(' + n + ', 28px);gap:1px;margin:8px 0">';
        grid += '<div></div>';
        for (let j = 0; j < n; j++) {{
            grid += '<div style="font-size:9px;color:{_DIM_TEXT};text-align:center;overflow:hidden;max-width:28px">' + TOKENS[j] + '</div>';
        }}
        for (let i = 0; i < n; i++) {{
            grid += '<div style="font-size:9px;color:{_DIM_TEXT};text-align:right;padding-right:4px">' + TOKENS[i] + '</div>';
            for (let j = 0; j < n; j++) {{
                const w = h.weights[i][j];
                grid += '<div class="heatmap-cell" style="background:' + colorForWeight(w) + '"' +
                    ' onmousemove="showTip(event, TOKENS['+i+']+\\'→\\'+TOKENS['+j+']+\\': \\'+' + w.toFixed(4) + ')"' +
                    ' onmouseleave="hideTip()"></div>';
            }}
        }}
        grid += '</div>';
        html += '<div class="panel"><h2>Head ' + h.head + ' <span style="color:{_DIM_TEXT};font-size:0.8em">(entropy: ' + h.entropy + ')</span></h2>' + grid + '</div>';
    }}
    container.innerHTML = html;
}}

renderLayer(Object.keys(DATA)[0] || '0');
"""

    return _wrap_page("Attention Patterns", body, extra_js=js)


def html_trace(results: list[dict[str, Any]]) -> str:
    """Generate an interactive causal trace HTML page.

    Sortable horizontal bar chart. Hover for exact effect values.
    Click filter buttons to filter by role.
    """
    if not results:
        return _wrap_page("Causal Trace", "<h1>Causal Trace</h1><p>No results.</p>")

    data_json = json.dumps([
        {
            "module": r["module"],
            "effect": round(r["effect"], 4),
            "role": r.get("role", ""),
        }
        for r in results
    ])

    roles = sorted(set(r.get("role", "") for r in results if r.get("role")))
    role_buttons = "".join(
        f'<button class="filter-btn" onclick="toggleFilter(\'{role}\')">{role}</button>'
        for role in roles
    )

    body = f"""
<h1>Causal Trace</h1>
<div class="panel controls">
    <label>Filter by role:</label>
    <button class="filter-btn active" onclick="toggleFilter('all')">All</button>
    {role_buttons}
    <span style="margin-left:16px;color:{_DIM_TEXT}">Click bars for details</span>
</div>
<div id="chartContainer" class="panel"></div>
"""

    js = f"""
const TRACE_DATA = {data_json};
let activeFilter = 'all';

function toggleFilter(role) {{
    activeFilter = role;
    document.querySelectorAll('.filter-btn').forEach(b => {{
        b.classList.toggle('active', b.textContent === role || (role === 'all' && b.textContent === 'All'));
    }});
    renderChart();
}}

function renderChart() {{
    const filtered = activeFilter === 'all' ? TRACE_DATA : TRACE_DATA.filter(d => d.role === activeFilter);
    const maxEffect = Math.max(...filtered.map(d => d.effect), 0.001);
    const container = document.getElementById('chartContainer');
    let html = '<table><tr><th>Module</th><th>Role</th><th style="width:50%">Effect</th><th>Value</th></tr>';
    for (const d of filtered) {{
        const pct = (d.effect / maxEffect * 100).toFixed(1);
        html += '<tr onmousemove="showTip(event, \\'' + d.module + ': ' + d.effect.toFixed(4) + '\\')" onmouseleave="hideTip()">' +
            '<td style="color:{_TEXT};font-family:monospace;font-size:0.85em">' + d.module + '</td>' +
            '<td style="color:{_DIM_TEXT}">' + (d.role || '-') + '</td>' +
            '<td><div class="bar" style="width:' + pct + '%;background:linear-gradient(90deg,{_GREEN},{_HIGHLIGHT})"></div></td>' +
            '<td style="text-align:right;font-weight:600">' + d.effect.toFixed(4) + '</td></tr>';
    }}
    html += '</table>';
    container.innerHTML = html;
}}

renderChart();
"""

    return _wrap_page("Causal Trace", body, extra_js=js)


def html_attribution(
    tokens: list[str],
    scores: list[float],
) -> str:
    """Generate an interactive token attribution HTML page.

    Slider to adjust display threshold. Hover tokens for exact scores.
    """
    if not tokens or not scores:
        return _wrap_page("Attribution", "<h1>Attribution</h1><p>No data.</p>")

    data_json = json.dumps([
        {"token": html.escape(t), "score": round(s, 6)}
        for t, s in zip(tokens, scores)
    ])

    max_score = max(abs(s) for s in scores) if scores else 1.0

    body = f"""
<h1>Attribution (Gradient Saliency)</h1>
<div class="panel controls">
    <label>Threshold: <input type="range" id="threshold" min="0" max="100" value="0"
        oninput="renderAttribution(this.value / 100)">
    <span id="threshVal">0%</span></label>
</div>
<div class="panel">
    <h2>Token Coloring</h2>
    <div id="tokenContainer" style="margin:12px 0;line-height:2.2"></div>
</div>
<div class="panel">
    <h2>Ranked Tokens</h2>
    <div id="rankedContainer"></div>
</div>
"""

    js = f"""
const ATTR_DATA = {data_json};
const MAX_SCORE = {max_score};

function scoreColor(intensity) {{
    if (intensity > 0.7) return 'rgba(233,69,96,0.85)';
    if (intensity > 0.4) return 'rgba(255,193,7,0.65)';
    if (intensity > 0.15) return 'rgba(255,255,255,0.15)';
    return 'transparent';
}}

function renderAttribution(threshold) {{
    document.getElementById('threshVal').textContent = (threshold * 100).toFixed(0) + '%';
    const tc = document.getElementById('tokenContainer');
    const rc = document.getElementById('rankedContainer');
    let tokHtml = '';
    for (const d of ATTR_DATA) {{
        const intensity = Math.abs(d.score) / MAX_SCORE;
        if (intensity < threshold) {{
            tokHtml += '<span class="token" style="opacity:0.3" onmousemove="showTip(event, \\'' + d.token + ': ' + d.score.toFixed(4) + '\\')" onmouseleave="hideTip()">' + d.token + '</span>';
        }} else {{
            tokHtml += '<span class="token" style="background:' + scoreColor(intensity) + '" onmousemove="showTip(event, \\'' + d.token + ': ' + d.score.toFixed(4) + '\\')" onmouseleave="hideTip()">' + d.token + '</span>';
        }}
    }}
    tc.innerHTML = tokHtml;

    const sorted = [...ATTR_DATA].sort((a,b) => Math.abs(b.score) - Math.abs(a.score));
    let rankHtml = '<table><tr><th>Token</th><th style="width:60%">Score</th><th>Value</th></tr>';
    for (const d of sorted.slice(0, 20)) {{
        const intensity = Math.abs(d.score) / MAX_SCORE;
        if (intensity < threshold) continue;
        const pct = (intensity * 100).toFixed(1);
        rankHtml += '<tr><td style="font-weight:600">' + d.token + '</td>' +
            '<td><div class="bar" style="width:' + pct + '%;background:{_HIGHLIGHT}"></div></td>' +
            '<td style="text-align:right">' + d.score.toFixed(4) + '</td></tr>';
    }}
    rankHtml += '</table>';
    rc.innerHTML = rankHtml;
}}

renderAttribution(0);
"""

    return _wrap_page("Attribution", body, extra_js=js)


def save_html(content: str, path: str) -> None:
    """Write HTML content to a file and print confirmation."""
    from pathlib import Path

    from rich.console import Console

    Path(path).write_text(content, encoding="utf-8")
    Console().print(f"  Interactive HTML saved to [bold]{path}[/bold]")
