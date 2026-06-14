// Collapsible raw-JSON panel — the universal fallback renderer and the
// "see exactly what the API returned" affordance on every result.

import { downloadJSON, el } from '../util.js';

export function jsonView(data, { open = false, label = 'Raw JSON' } = {}) {
  const pre = el('pre', {}, safeStringify(data));
  return el(
    'details',
    { class: 'json-view', open },
    el('summary', { style: 'cursor:pointer;color:var(--dim)' }, label),
    pre,
  );
}

export function downloadButton(data, filename) {
  return el(
    'button',
    { class: 'btn secondary small', onclick: () => downloadJSON(data, filename) },
    'Download JSON',
  );
}

function safeStringify(data) {
  try {
    const s = JSON.stringify(data, null, 2);
    return s.length > 400000 ? s.slice(0, 400000) + '\n… (truncated for display — use Download JSON)' : s;
  } catch {
    return String(data);
  }
}
