// Attention renderer: layer/head selectors + per-head weight heatmap.

import { dataTable, heatmap } from '../components/charts.js';
import { el, fmtNum } from '../util.js';

export function render(result, container) {
  const heads = result.heads || [];
  const tokens = (result.tokens || []).map(String);
  if (!heads.length) {
    container.append(el('p', { class: 'empty-hint' }, 'No attention heads matched.'));
    return;
  }

  const layers = [...new Set(heads.map((h) => h.layer))].sort((a, b) => a - b);
  const layerSel = el('select', { style: 'width:auto' });
  for (const l of layers) layerSel.append(el('option', { value: String(l) }, `layer ${l}`));
  const headSel = el('select', { style: 'width:auto' });

  const display = el('div');

  function headsForLayer(layer) {
    return heads.filter((h) => h.layer === layer).sort((a, b) => a.head - b.head);
  }

  function refreshHeads() {
    const layer = parseInt(layerSel.value, 10);
    headSel.replaceChildren();
    for (const h of headsForLayer(layer)) headSel.append(el('option', { value: String(h.head) }, `head ${h.head}`));
    draw();
  }

  function draw() {
    const layer = parseInt(layerSel.value, 10);
    const head = parseInt(headSel.value, 10);
    const entry = heads.find((h) => h.layer === layer && h.head === head);
    display.replaceChildren();
    if (!entry) return;

    display.append(
      el('p', { class: 'subtitle' }, `Rows attend to columns (query → key). Entropy: ${fmtNum(entry.entropy)} · kind: ${entry.attention_kind ?? 'self'}`),
      heatmap({
        values: entry.weights,
        rowLabels: tokens.length ? tokens : entry.weights.map((_, i) => `${i}`),
        colLabels: tokens.length ? tokens : entry.weights.map((_, i) => `${i}`),
        tip: (i, j, v) => `${JSON.stringify(tokens[i] ?? i)} → ${JSON.stringify(tokens[j] ?? j)}\nweight: ${fmtNum(v)}`,
      }),
    );

    if (entry.top_pairs && entry.top_pairs.length) {
      const pairs = entry.top_pairs.map((p) => {
        if (Array.isArray(p)) {
          const [from, to, w] = p;
          return { from: tokens[from] ?? from, to: tokens[to] ?? to, weight: w };
        }
        return { from: p.from ?? p.src, to: p.to ?? p.dst, weight: p.weight ?? p.value };
      });
      display.append(
        el('h2', {}, 'Strongest pairs'),
        dataTable(
          [
            { key: 'from', label: 'From', mono: true },
            { key: 'to', label: 'To', mono: true },
            { key: 'weight', label: 'Weight', numeric: true },
          ],
          pairs,
        ),
      );
    }
  }

  layerSel.addEventListener('change', refreshHeads);
  headSel.addEventListener('change', draw);

  container.append(
    el('div', { class: 'panel', style: 'display:flex;gap:12px;align-items:center' }, el('label', {}, 'Layer'), layerSel, el('label', {}, 'Head'), headSel),
    display,
  );
  refreshHeads();
}
