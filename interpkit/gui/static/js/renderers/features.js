// SAE features renderer: top-feature activations + reconstruction stats.

import { barChart, statCards } from '../components/charts.js';
import { el, fmtNum, fmtPct } from '../util.js';

export function render(result, container) {
  const top = result.top_features || result.features || [];
  const items = top.map((f) => {
    if (Array.isArray(f)) return { label: `feature ${f[0]}`, value: f[1] };
    return { label: `feature ${f.id ?? f.feature ?? '?'}`, value: f.activation ?? f.value ?? f.score ?? 0 };
  });

  const stats = [];
  if (result.reconstruction_error != null) stats.push({ label: 'Reconstruction error', value: fmtNum(result.reconstruction_error) });
  if (result.active_fraction != null) stats.push({ label: 'Active features', value: fmtPct(result.active_fraction), kind: 'green' });
  if (result.loss_ratio != null) stats.push({ label: 'Loss ratio', value: fmtNum(result.loss_ratio) });
  if (stats.length) container.append(statCards(stats));

  if (items.length) {
    container.append(el('h2', {}, `Top features at ${result.module ?? '?'}`), barChart(items));
  } else {
    container.append(el('p', { class: 'empty-hint' }, 'No active features.'));
  }
}
