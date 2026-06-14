// Renderers for ops whose results are scalar dicts / small tables:
// activations, decompose, patch, ablate, probe, diff, report,
// train-tuned-lens, chat (history fallback).

import { barChart, dataTable, kvGrid, statCards } from '../components/charts.js';
import { el, fmtNum, fmtPct } from '../util.js';

export function renderActivations(result, container) {
  const stats = result.stats || [];
  container.append(
    dataTable(
      [
        { key: 'module', label: 'Module', mono: true },
        { key: 'shape', label: 'Shape', fmt: (v) => JSON.stringify(v) },
        { key: 'mean', label: 'Mean', numeric: true },
        { key: 'std', label: 'Std', numeric: true },
        { key: 'min', label: 'Min', numeric: true },
        { key: 'max', label: 'Max', numeric: true },
        { key: 'l2_norm', label: 'L2 norm', numeric: true },
      ],
      stats,
    ),
  );
}

export function renderDecompose(result, container) {
  const components = result.components || [];
  if (components.length) {
    const valueKey = ['contribution', 'norm'].find((k) => components[0][k] !== undefined) || 'norm';
    container.append(
      el('p', { class: 'subtitle' }, `Residual-stream components by ${valueKey}.`),
      barChart(
        components.map((c) => ({
          label: c.name ?? '?',
          value: c[valueKey] ?? 0,
          tip: Object.entries(c)
            .filter(([, v]) => typeof v !== 'object')
            .map(([k, v]) => `${k}: ${typeof v === 'number' ? fmtNum(v) : v}`)
            .join('\n'),
        })),
      ),
    );
  }
  container.append(kvGrid(result, { skip: ['components'] }));
}

export function renderPatch(result, container) {
  const cards = [];
  if (result.effect != null) cards.push({ label: 'Effect', value: fmtNum(result.effect), kind: 'hl' });
  if (result.recovery != null) cards.push({ label: 'Recovery', value: fmtPct(result.recovery), kind: 'green' });
  if (cards.length) container.append(statCards(cards));
  appendTopComparison(result, container);
  container.append(kvGrid(result, { skip: topKeys(result) }));
}

/** Side-by-side top-token panels for any `*_top` summaries in the result. */
function topKeys(result) {
  return Object.keys(result).filter((k) => k.endsWith('_top'));
}

function appendTopComparison(result, container) {
  const keys = topKeys(result);
  if (!keys.length) return;
  const grid = el('div', { style: `display:grid;grid-template-columns:repeat(${keys.length},1fr);gap:16px` });
  for (const key of keys) {
    const items = (result[key] || []).map((e) => ({ label: JSON.stringify(e[0]), value: e[1] }));
    grid.append(
      el('div', { class: 'panel' },
        el('h2', { style: 'margin-top:0' }, key.replace('_top', '')),
        barChart(items, { fmt: fmtPct }),
      ),
    );
  }
  container.append(grid);
}

export function renderProbe(result, container) {
  if (result.accuracy != null) {
    container.append(statCards([{ label: 'Probe accuracy', value: fmtPct(result.accuracy), kind: 'green' }]));
  }
  container.append(kvGrid(result, { skip: ['weights', 'bias'] }));
}

export function renderKv(result, container) {
  container.append(el('div', { class: 'panel' }, kvGrid(result)));
}

export function renderReport(result, container) {
  if (result.html_path) {
    container.append(el('p', { class: 'subtitle' }, `Report saved to ${result.html_path}`));
  }
  if (result.report_html) {
    const blob = new Blob([result.report_html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    container.append(
      el('div', { style: 'margin-bottom:10px' },
        el('a', { href: url, target: '_blank', class: 'btn secondary small', style: 'text-decoration:none' }, 'Open report in new tab'),
      ),
      el('iframe', { class: 'report-frame', src: url }),
    );
  }
}

export function renderTunedLens(result, container) {
  container.append(
    el('div', { class: 'panel' },
      el('h2', { style: 'margin-top:0' }, 'Tuned lens trained'),
      el('p', {}, 'Saved to: ', el('code', { style: 'font-family:var(--mono)' }, result.saved_to ?? '?')),
      el('p', { class: 'subtitle', style: 'margin:8px 0 0' }, 'Use this path in the Logit lens panel ("tuned lens path") for unbiased early-layer readouts.'),
    ),
    result.meta ? kvGrid(result.meta) : null,
  );
}

export function renderChatResult(result, container) {
  container.append(
    el('div', { class: 'panel', style: 'white-space:pre-wrap' }, result.response ?? ''),
    result.prompt
      ? el('details', {}, el('summary', { style: 'cursor:pointer;color:var(--dim)' }, 'Templated prompt'),
          el('pre', { style: 'font-family:var(--mono);font-size:0.85em;white-space:pre-wrap;margin-top:8px' }, result.prompt))
      : null,
  );
}
