// Causal-trace renderer: module ranking bars or position-mode heatmap.

import { barChart, heatmap, kvGrid } from '../components/charts.js';
import { el, fmtNum } from '../util.js';

export function render(result, container) {
  // Position mode: {effects: 2D, layer_names, tokens, ...}
  if (result.effects) {
    container.append(
      el('p', { class: 'subtitle' }, 'Rows = layers, columns = token positions. Brighter = higher recovery effect.'),
      heatmap({
        values: result.effects,
        rowLabels: result.layer_names || [],
        colLabels: (result.tokens || []).map(String),
        tip: (i, j, v) => `${result.layer_names?.[i] ?? `L${i}`} · pos ${j} ${JSON.stringify(result.tokens?.[j] ?? '')}\neffect: ${fmtNum(v)}`,
      }),
      kvGrid(result, { skip: ['effects', 'layer_names', 'tokens'] }),
    );
    return;
  }

  // Module mode: {results: [{module|name, effect|score, ...}]}
  const rows = result.results || [];
  if (!rows.length) {
    container.append(el('p', { class: 'empty-hint' }, 'No modules traced.'));
    return;
  }
  const scoreKey = ['effect', 'score', 'causal_effect', 'recovery'].find((k) => rows[0][k] !== undefined);
  container.append(
    el('p', { class: 'subtitle' }, 'Modules ranked by how much patching clean activations into the corrupted run restores the clean output.'),
    barChart(
      rows.map((r) => ({
        label: r.module ?? r.name ?? r.module_name ?? '?',
        value: scoreKey ? r[scoreKey] : 0,
        tip: Object.entries(r)
          .filter(([, v]) => typeof v !== 'object')
          .map(([k, v]) => `${k}: ${typeof v === 'number' ? fmtNum(v) : v}`)
          .join('\n'),
      })),
    ),
  );
}
