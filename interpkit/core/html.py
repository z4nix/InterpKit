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

    tok_labels = tokens or [str(i) for i in range(max((len(entry.get("weights", [])) for entry in attention_data), default=0))]
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
        f'<button class="filter-btn" onclick="toggleFilter(\'{html.escape(role, quote=True)}\')">'
        f'{html.escape(role)}</button>'
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

    body = """
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


def html_lens(predictions: list[dict[str, Any]], input_tokens: list[str] | None = None) -> str:
    """Interactive logit lens heatmap — layer x position with hover predictions."""
    if not predictions:
        return _wrap_page("Logit Lens", "<h1>Logit Lens</h1><p>No data.</p>")

    data_json = json.dumps(predictions, default=str)
    tok_json = json.dumps([html.escape(t) for t in input_tokens] if input_tokens else [])

    body = """
<h1>Logit Lens</h1>
<p class="subtitle">Each cell shows the model's top-1 prediction at that (layer, position).</p>
<div class="controls panel">
    <label>Color by: <select id="colorBy" onchange="renderLens()">
        <option value="prob">Probability</option>
    </select></label>
</div>
<div id="lensContainer" class="panel" style="overflow-x:auto"></div>
"""

    js = f"""
const LENS_DATA = {data_json};
const LENS_TOKENS = {tok_json};

function renderLens() {{
    const container = document.getElementById('lensContainer');
    if (!LENS_DATA.length) {{ container.innerHTML = '<p>No data</p>'; return; }}

    const layers = [...new Set(LENS_DATA.map(d => d.layer || 0))].sort((a,b) => a-b);
    const positions = [...new Set(LENS_DATA.map(d => d.position || 0))].sort((a,b) => a-b);

    const lookup = {{}};
    for (const d of LENS_DATA) {{
        lookup[(d.layer||0) + ',' + (d.position||0)] = d;
    }}

    let h = '<table><tr><th>Layer \\\\ Pos</th>';
    for (const p of positions) {{
        const tok = LENS_TOKENS[p] || p;
        h += '<th style="font-size:10px;max-width:50px;overflow:hidden">' + tok + '</th>';
    }}
    h += '</tr>';

    for (const l of layers) {{
        h += '<tr><td style="font-weight:600">L' + l + '</td>';
        for (const p of positions) {{
            const d = lookup[l + ',' + p];
            if (!d) {{ h += '<td>-</td>'; continue; }}
            const pred = d.top_prediction || d.prediction || '-';
            const prob = d.probability || d.prob || 0;
            const intensity = Math.min(prob * 2, 1);
            const r = Math.round(233 * intensity + 26 * (1-intensity));
            const g = Math.round(69 * intensity + 33 * (1-intensity));
            const b = Math.round(96 * intensity + 62 * (1-intensity));
            const bg = 'rgb('+r+','+g+','+b+')';
            const tip = 'L'+l+' pos'+p+': '+pred+' ('+prob.toFixed(3)+')';
            h += '<td class="heatmap-cell" style="background:'+bg+';font-size:9px;color:white"' +
                ' onmousemove="showTip(event,\\''+tip.replace(/'/g,"\\\\'")+'\\')" onmouseleave="hideTip()">' +
                (pred.length > 6 ? pred.slice(0,5)+'…' : pred) + '</td>';
        }}
        h += '</tr>';
    }}
    h += '</table>';
    container.innerHTML = h;
}}
renderLens();
"""

    return _wrap_page("Logit Lens", body, extra_js=js)


def html_dla(result: dict[str, Any]) -> str:
    """Interactive DLA bar chart — sortable, with head breakdown tabs."""
    contributions = result.get("contributions", [])
    if not contributions:
        return _wrap_page("Direct Logit Attribution", "<h1>DLA</h1><p>No data.</p>")

    target_token = result.get("target_token", "?")
    total_logit = result.get("total_logit", 0)
    note = result.get("approximation_note", "")

    data_json = json.dumps([
        {
            "component": c.get("component", ""),
            "layer": c.get("layer", 0),
            "type": c.get("type", ""),
            "logit": round(c.get("logit_contribution", 0), 4),
        }
        for c in contributions
    ])

    body = f"""
<h1>Direct Logit Attribution</h1>
<p class="subtitle">Target: <strong>{html.escape(str(target_token))}</strong> | Total logit (approx): {total_logit:.3f}</p>
{"<p class='subtitle' style='font-size:0.8em'>" + html.escape(note) + "</p>" if note else ""}
<div class="controls panel">
    <label>Sort by:
        <select id="sortBy" onchange="renderDLA()">
            <option value="abs">|Contribution|</option>
            <option value="pos">Positive first</option>
            <option value="neg">Negative first</option>
        </select>
    </label>
    <label style="margin-left:16px">Show:
        <select id="filterType" onchange="renderDLA()">
            <option value="all">All</option>
            <option value="attn">Attention only</option>
            <option value="mlp">MLP only</option>
        </select>
    </label>
</div>
<div id="dlaContainer" class="panel"></div>
"""

    js = f"""
const DLA_DATA = {data_json};

function renderDLA() {{
    const sortBy = document.getElementById('sortBy').value;
    const filterType = document.getElementById('filterType').value;
    let data = [...DLA_DATA];
    if (filterType !== 'all') data = data.filter(d => d.type === filterType);
    if (sortBy === 'abs') data.sort((a,b) => Math.abs(b.logit) - Math.abs(a.logit));
    else if (sortBy === 'pos') data.sort((a,b) => b.logit - a.logit);
    else data.sort((a,b) => a.logit - b.logit);

    const maxAbs = Math.max(...data.map(d => Math.abs(d.logit)), 0.001);
    let h = '<table><tr><th>Component</th><th>Type</th><th style="width:50%">Contribution</th><th>Logit</th></tr>';
    for (const d of data.slice(0, 50)) {{
        const pct = (Math.abs(d.logit) / maxAbs * 100).toFixed(1);
        const color = d.logit >= 0 ? '{_GREEN}' : '{_HIGHLIGHT}';
        h += '<tr onmousemove="showTip(event, \\'' + d.component + ': ' + d.logit.toFixed(4) + '\\')" onmouseleave="hideTip()">' +
            '<td style="font-family:monospace;font-size:0.85em">' + d.component + '</td>' +
            '<td style="color:{_DIM_TEXT}">' + d.type + '</td>' +
            '<td><div class="bar" style="width:' + pct + '%;background:' + color + '"></div></td>' +
            '<td style="text-align:right;font-weight:600;color:' + color + '">' + d.logit.toFixed(4) + '</td></tr>';
    }}
    h += '</table>';
    document.getElementById('dlaContainer').innerHTML = h;
}}
renderDLA();
"""

    return _wrap_page("Direct Logit Attribution", body, extra_js=js)


def html_position_trace(result: dict[str, Any]) -> str:
    """Interactive position-level causal trace heatmap (Meng et al. style)."""
    effects = result.get("effects")
    layer_names = result.get("layer_names", [])
    tokens = result.get("tokens", [])

    if effects is None:
        return _wrap_page("Position Trace", "<h1>Position Trace</h1><p>No data.</p>")

    import torch
    if isinstance(effects, torch.Tensor):
        effects_list = effects.detach().cpu().tolist()
    else:
        effects_list = effects

    data_json = json.dumps(effects_list)
    layers_json = json.dumps(layer_names)
    tokens_json = json.dumps([html.escape(str(t)) for t in tokens] if tokens else [])

    body = """
<h1>Position-Level Causal Trace</h1>
<p class="subtitle">Rows = layers, columns = token positions. Brighter = higher recovery effect.</p>
<div id="ptContainer" class="panel" style="overflow-x:auto"></div>
"""

    js = f"""
const PT_EFFECTS = {data_json};
const PT_LAYERS = {layers_json};
const PT_TOKENS = {tokens_json};

(function() {{
    const container = document.getElementById('ptContainer');
    const nLayers = PT_EFFECTS.length;
    if (!nLayers) {{ container.innerHTML = '<p>No data</p>'; return; }}
    const nPos = Math.max(...PT_EFFECTS.map(r => r.length), 0);

    let maxVal = 0;
    for (let i = 0; i < nLayers; i++)
        for (let j = 0; j < (PT_EFFECTS[i] || []).length; j++)
            if (PT_EFFECTS[i][j] > maxVal) maxVal = PT_EFFECTS[i][j];
    if (maxVal < 0.001) maxVal = 1;

    let h = '<table><tr><th></th>';
    for (let j = 0; j < nPos; j++) {{
        h += '<th style="font-size:9px;max-width:40px;overflow:hidden">' + (PT_TOKENS[j] || j) + '</th>';
    }}
    h += '</tr>';

    for (let i = 0; i < nLayers; i++) {{
        const lbl = PT_LAYERS[i] || ('L' + i);
        h += '<tr><td style="font-size:10px;font-weight:600;white-space:nowrap">' + lbl + '</td>';
        for (let j = 0; j < nPos; j++) {{
            const v = (PT_EFFECTS[i] && PT_EFFECTS[i][j] != null) ? PT_EFFECTS[i][j] : 0;
            const intensity = v / maxVal;
            const r = Math.round(233 * intensity + 26 * (1-intensity));
            const g = Math.round(69 * intensity + 33 * (1-intensity));
            const b = Math.round(96 * intensity + 62 * (1-intensity));
            const tip = lbl + ', pos ' + j + ': ' + v.toFixed(4);
            h += '<td class="heatmap-cell" style="background:rgb('+r+','+g+','+b+')"' +
                ' onmousemove="showTip(event,\\''+tip+'\\')" onmouseleave="hideTip()"></td>';
        }}
        h += '</tr>';
    }}
    h += '</table>';
    container.innerHTML = h;
}})();
"""

    return _wrap_page("Position Trace", body, extra_js=js)


def html_steer(
    original: dict[str, Any],
    steered: dict[str, Any],
    module: str,
    scale: float,
) -> str:
    """Side-by-side comparison of original vs steered top predictions."""
    orig_preds = original.get("top_predictions", original.get("predictions", []))
    steer_preds = steered.get("top_predictions", steered.get("predictions", []))

    orig_json = json.dumps(orig_preds[:20], default=str)
    steer_json = json.dumps(steer_preds[:20], default=str)

    body = f"""
<h1>Steering Comparison</h1>
<p class="subtitle">Module: <strong>{html.escape(module)}</strong> | Scale: {scale}</p>
<div style="display:flex;gap:16px;flex-wrap:wrap">
    <div class="panel" style="flex:1;min-width:300px">
        <h2>Original</h2>
        <div id="origContainer"></div>
    </div>
    <div class="panel" style="flex:1;min-width:300px">
        <h2>Steered</h2>
        <div id="steerContainer"></div>
    </div>
</div>
"""

    js = f"""
const ORIG = {orig_json};
const STEERED = {steer_json};

function renderPreds(data, containerId) {{
    const c = document.getElementById(containerId);
    if (!data.length) {{ c.innerHTML = '<p>No predictions</p>'; return; }}
    const maxP = Math.max(...data.map(d => typeof d === 'object' ? (d.prob || d.probability || 0) : 0), 0.001);
    let h = '<table><tr><th>Token</th><th style="width:60%">Probability</th><th>Value</th></tr>';
    for (const d of data) {{
        const token = typeof d === 'object' ? (d.token || d.word || '?') : String(d);
        const prob = typeof d === 'object' ? (d.prob || d.probability || 0) : 0;
        const pct = (prob / maxP * 100).toFixed(1);
        h += '<tr><td style="font-weight:600">' + token + '</td>' +
            '<td><div class="bar" style="width:'+pct+'%;background:{_GREEN}"></div></td>' +
            '<td style="text-align:right">' + prob.toFixed(4) + '</td></tr>';
    }}
    h += '</table>';
    c.innerHTML = h;
}}
renderPreds(ORIG, 'origContainer');
renderPreds(STEERED, 'steerContainer');
"""

    return _wrap_page("Steering", body, extra_js=js)


def html_diff(results: dict[str, Any], model_a: str, model_b: str) -> str:
    """Per-module distance bars comparing two models' activations."""
    distances = results.get("distances", results.get("module_distances", results.get("results", [])))
    if not distances:
        return _wrap_page("Model Diff", "<h1>Diff</h1><p>No data.</p>")

    data_json = json.dumps(distances, default=str)

    body = f"""
<h1>Model Diff</h1>
<p class="subtitle">{html.escape(model_a)} vs {html.escape(model_b)}</p>
<div class="controls panel">
    <label>Sort:
        <select id="diffSort" onchange="renderDiff()">
            <option value="desc">Largest first</option>
            <option value="asc">Smallest first</option>
            <option value="name">By name</option>
        </select>
    </label>
</div>
<div id="diffContainer" class="panel"></div>
"""

    js = f"""
const DIFF_DATA = {data_json};

function renderDiff() {{
    const sort = document.getElementById('diffSort').value;
    let data = [...DIFF_DATA];
    if (sort === 'desc') data.sort((a,b) => (b.distance||0) - (a.distance||0));
    else if (sort === 'asc') data.sort((a,b) => (a.distance||0) - (b.distance||0));
    else data.sort((a,b) => (a.module||'').localeCompare(b.module||''));

    const maxD = Math.max(...data.map(d => d.distance || 0), 0.001);
    let h = '<table><tr><th>Module</th><th style="width:50%">Distance</th><th>Value</th></tr>';
    for (const d of data.slice(0, 50)) {{
        const dist = d.distance || 0;
        const pct = (dist / maxD * 100).toFixed(1);
        h += '<tr><td style="font-family:monospace;font-size:0.85em">' + (d.module || '') + '</td>' +
            '<td><div class="bar" style="width:' + pct + '%;background:linear-gradient(90deg,{_GREEN},{_HIGHLIGHT})"></div></td>' +
            '<td style="text-align:right;font-weight:600">' + dist.toFixed(4) + '</td></tr>';
    }}
    h += '</table>';
    document.getElementById('diffContainer').innerHTML = h;
}}
renderDiff();
"""

    return _wrap_page("Model Diff", body, extra_js=js)


def save_html(content: str, path: str) -> None:
    """Write HTML content to a file and print confirmation."""
    from pathlib import Path

    from rich.console import Console

    Path(path).write_text(content, encoding="utf-8")
    Console().print(f"  Interactive HTML saved to [bold]{path}[/bold]")
