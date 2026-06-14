// Shared result-rendering primitives: barchart, heatmap, token strip,
// table, key-value grid, stat cards. All renderers compose these.

import { divergeColor, el, fmtNum, heatColor, truncate, withTip } from '../util.js';

/**
 * Horizontal signed bar chart.
 * items: [{label, value, tip?}] — values may be negative (diverging colors).
 */
export function barChart(items, { fmt = fmtNum } = {}) {
  const maxAbs = Math.max(...items.map((d) => Math.abs(d.value ?? 0)), 1e-12);
  const hasNeg = items.some((d) => (d.value ?? 0) < 0);
  const root = el('div', { class: 'barchart' });

  for (const item of items) {
    const v = item.value ?? 0;
    const frac = Math.abs(v) / maxAbs;
    const track = el('div', { class: 'track' });
    if (hasNeg) {
      track.append(el('div', { class: 'axis', style: 'left:50%' }));
      const w = (frac * 50).toFixed(2);
      const bar = el('div', {
        class: 'bar',
        style: `width:${w}%;${v >= 0 ? 'left:50%' : `right:50%`};background:${divergeColor(v, maxAbs)}`,
      });
      track.append(bar);
    } else {
      track.append(el('div', { class: 'bar', style: `width:${(frac * 100).toFixed(2)}%;left:0;background:${heatColor(0.35 + 0.65 * frac)}` }));
    }
    const row = el(
      'div',
      { class: 'row' },
      el('span', { class: 'lbl', title: item.label }, item.label),
      track,
      el('span', { class: 'val' }, fmt(v)),
    );
    if (item.tip) withTip(row, item.tip);
    root.append(row);
  }
  return root;
}

/**
 * Grid heatmap.
 * opts: {values: number[][], rowLabels, colLabels, cellText?: (i,j)=>string,
 *        tip?: (i,j,v)=>string, signed?: bool, normalize?: 'global'}
 */
export function heatmap({ values, rowLabels = [], colLabels = [], cellText = null, tip = null, signed = false }) {
  const flat = values.flat().filter((v) => v != null && Number.isFinite(v));
  const maxAbs = Math.max(...flat.map(Math.abs), 1e-12);
  const min = Math.min(...flat, 0);
  const max = Math.max(...flat, 1e-12);

  const table = el('table');
  if (colLabels.length) {
    const head = el('tr', {}, el('th'));
    for (const c of colLabels) head.append(el('th', { title: c }, truncate(c, 9)));
    table.append(head);
  }
  values.forEach((row, i) => {
    const tr = el('tr', {}, el('td', { class: 'rowlbl' }, rowLabels[i] ?? `${i}`));
    row.forEach((v, j) => {
      let bg;
      if (v == null || !Number.isFinite(v)) bg = 'var(--accent)';
      else if (signed) bg = divergeColor(v, maxAbs);
      else bg = heatColor((v - min) / (max - min || 1));
      const td = el('td', { class: 'cell', style: `background:${bg}` }, cellText ? cellText(i, j) : '');
      if (tip) withTip(td, () => tip(i, j, v));
      tr.append(td);
    });
    table.append(tr);
  });
  return el('div', { class: 'heatmap' }, table);
}

/**
 * Token strip: tokens coloured by score (diverging when signed).
 * tokens: string[], scores: number[]
 */
export function tokenStrip(tokens, scores, { tip = null } = {}) {
  const maxAbs = Math.max(...scores.map((s) => Math.abs(s ?? 0)), 1e-12);
  const root = el('div', { class: 'token-strip' });
  tokens.forEach((token, i) => {
    const s = scores[i] ?? 0;
    const node = el('span', { class: 'token', style: `background:${divergeColor(s, maxAbs)}` }, token);
    withTip(node, tip ? () => tip(i, s) : `${JSON.stringify(token)}  ${fmtNum(s)}`);
    root.append(node);
  });
  return root;
}

/** Generic table. columns: [{key, label, numeric?, fmt?}], rows: object[] */
export function dataTable(columns, rows) {
  const table = el('table', { class: 'tbl' });
  table.append(el('tr', {}, columns.map((c) => el('th', {}, c.label))));
  for (const row of rows) {
    table.append(
      el(
        'tr',
        {},
        columns.map((c) => {
          let v = row[c.key];
          if (c.fmt) v = c.fmt(v, row);
          else if (typeof v === 'number') v = fmtNum(v);
          else if (v == null) v = '—';
          else if (typeof v === 'object') v = JSON.stringify(v);
          return el('td', { class: c.numeric ? 'num' : c.mono ? 'mono' : '' }, String(v));
        }),
      ),
    );
  }
  return table;
}

/** Key/value grid for scalar result fields. */
export function kvGrid(obj, { skip = [] } = {}) {
  const root = el('div', { class: 'kv' });
  for (const [k, v] of Object.entries(obj)) {
    if (skip.includes(k) || v == null) continue;
    if (typeof v === 'object' && !Array.isArray(v)) continue;
    const shown = Array.isArray(v) ? truncate(JSON.stringify(v), 120) : typeof v === 'number' ? fmtNum(v) : String(v);
    root.append(el('span', { class: 'k' }, k), el('span', { class: 'v' }, shown));
  }
  return root;
}

/** Headline stat cards. stats: [{label, value, kind?: 'hl'|'green'}] */
export function statCards(stats) {
  return el(
    'div',
    { class: 'stat-cards' },
    stats.map((s) =>
      el(
        'div',
        { class: 'stat-card' },
        el('div', { class: 'label' }, s.label),
        el('div', { class: `value ${s.kind || ''}` }, String(s.value)),
      ),
    ),
  );
}
