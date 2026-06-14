// Tiny DOM + formatting helpers shared across the app.

/** Create an element: el('div', {class: 'x', onclick: fn}, child1, child2). */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null) continue;
    if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2), value);
    } else if (key === 'dataset') {
      Object.assign(node.dataset, value);
    } else if (key in node && key !== 'list' && key !== 'type') {
      node[key] = value;
    } else {
      node.setAttribute(key, value);
    }
  }
  for (const child of children.flat()) {
    if (child == null) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/** Compact number formatting for result values. */
export function fmtNum(v, digits = 4) {
  if (v == null || Number.isNaN(v)) return '—';
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e7) return String(v);
  if (v !== 0 && Math.abs(v) < 1e-4) return v.toExponential(2);
  return v.toFixed(digits);
}

export function fmtPct(v) {
  return v == null ? '—' : (v * 100).toFixed(1) + '%';
}

export function fmtParams(n) {
  if (n == null) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

// Shared hover tooltip (singleton element in index.html).
const tip = () => document.getElementById('tooltip');

export function showTip(event, text) {
  const t = tip();
  t.textContent = text;
  t.style.display = 'block';
  const pad = 14;
  const x = Math.min(event.clientX + pad, window.innerWidth - t.offsetWidth - pad);
  const y = Math.min(event.clientY + pad, window.innerHeight - t.offsetHeight - pad);
  t.style.left = x + 'px';
  t.style.top = y + 'px';
}

export function hideTip() {
  tip().style.display = 'none';
}

/** Attach tooltip behaviour to a node. */
export function withTip(node, textFn) {
  node.addEventListener('mousemove', (e) => showTip(e, typeof textFn === 'function' ? textFn() : textFn));
  node.addEventListener('mouseleave', hideTip);
  return node;
}

/** Trigger a client-side JSON download. */
export function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Shared visualization ramps, tuned to the indigo GUI palette.
const HEAT_LO = [33, 40, 58];     // muted surface (dark indigo-slate)
const HEAT_HI = [108, 123, 255];  // --highlight indigo
const DIV_POS = [242, 85, 122];   // --rose: pushes toward the target
const DIV_NEG = [88, 150, 246];   // cool blue: pushes against

function mix(lo, hi, t) {
  const r = Math.round(hi[0] * t + lo[0] * (1 - t));
  const g = Math.round(hi[1] * t + lo[1] * (1 - t));
  const b = Math.round(hi[2] * t + lo[2] * (1 - t));
  return `rgb(${r},${g},${b})`;
}

/** Single-hue magnitude ramp (surface → indigo) used for probabilities/weights. */
export function heatColor(intensity) {
  return mix(HEAT_LO, HEAT_HI, Math.max(0, Math.min(1, intensity)));
}

/** Diverging colour: negative → cool blue, positive → warm rose. */
export function divergeColor(value, maxAbs) {
  const t = maxAbs > 0 ? Math.max(-1, Math.min(1, value / maxAbs)) : 0;
  return t >= 0 ? mix(HEAT_LO, DIV_POS, t) : mix(HEAT_LO, DIV_NEG, -t);
}

export function truncate(s, n = 80) {
  s = String(s);
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}
