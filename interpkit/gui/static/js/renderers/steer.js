// Steer renderer: original vs steered top-token distributions side by side.

import { barChart } from '../components/charts.js';
import { el, fmtPct } from '../util.js';

function topItems(entries) {
  // _top_tokens entries arrive as [token, prob] pairs or {token, prob} dicts.
  return (entries || []).map((e) => {
    if (Array.isArray(e)) return { label: JSON.stringify(e[0]), value: e[1] };
    return { label: JSON.stringify(e.token ?? e[0] ?? '?'), value: e.prob ?? e.probability ?? e[1] ?? 0 };
  });
}

export function render(result, container) {
  if (result.feature != null) {
    container.append(
      el('p', { class: 'subtitle' }, `SAE feature ${result.feature} · mode ${result.mode} · strength ${result.strength}`),
    );
  }
  const grid = el('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:16px' });
  grid.append(
    el('div', { class: 'panel' }, el('h2', { style: 'margin-top:0' }, 'Original'), barChart(topItems(result.original_top), { fmt: fmtPct })),
    el('div', { class: 'panel' }, el('h2', { style: 'margin-top:0;color:var(--highlight)' }, 'Steered'), barChart(topItems(result.steered_top), { fmt: fmtPct })),
  );
  container.append(grid);
}
