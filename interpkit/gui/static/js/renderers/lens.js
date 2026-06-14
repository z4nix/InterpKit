// Logit-lens renderer: layers x positions heatmap of top-1 predictions.

import { heatmap } from '../components/charts.js';
import { el, fmtPct } from '../util.js';

export function render(result, container) {
  const entries = result.results || [];
  const tokens = result.tokens || [];
  if (!entries.length) {
    container.append(el('p', { class: 'empty-hint' }, 'No lens projections.'));
    return;
  }

  const perPosition = entries[0].positions && entries[0].positions.length;
  const rowLabels = entries.map((e, i) => e.layer_name || `L${i}`);

  let values, cellData;
  if (perPosition) {
    const nPos = Math.max(...entries.map((e) => e.positions.length));
    values = entries.map((e) => {
      const row = new Array(nPos).fill(null);
      for (const p of e.positions) row[p.pos ?? 0] = p.top1_prob ?? 0;
      return row;
    });
    cellData = (i, j) => entries[i].positions.find((p) => (p.pos ?? 0) === j) || null;
  } else {
    values = entries.map((e) => [e.top1_prob ?? 0]);
    cellData = (i) => entries[i];
  }

  const colLabels = perPosition
    ? values[0].map((_, j) => tokens[j] ?? `pos ${j}`)
    : ['prediction'];

  container.append(
    el('p', { class: 'subtitle' }, 'Each cell is the top-1 prediction read off at that layer. Hover for the top-5.'),
    heatmap({
      values,
      rowLabels,
      colLabels,
      cellText: (i, j) => {
        const d = cellData(i, j);
        return d ? d.top1_token ?? '' : '';
      },
      tip: (i, j) => {
        const d = cellData(i, j);
        if (!d) return 'no data';
        const top5 = (d.top5_tokens || []).map((t, k) => `${JSON.stringify(t)} ${fmtPct((d.top5_probs || [])[k])}`);
        return `${rowLabels[i]}${perPosition ? ` · ${JSON.stringify(colLabels[j])}` : ''}\n${top5.join('\n') || fmtPct(d.top1_prob)}`;
      },
    }),
  );
}
