// Circuit-discovery renderers: find-circuit, atp, eap, maxact.

import { barChart, dataTable, kvGrid, statCards, tokenStrip } from '../components/charts.js';
import { el, fmtNum } from '../util.js';

export function renderFindCircuit(result, container) {
  container.append(
    statCards([
      { label: 'Circuit size', value: (result.circuit || []).length, kind: 'hl' },
      { label: 'Excluded', value: (result.excluded || []).length },
      { label: 'Verification', value: fmtNum(result.verification), kind: 'green' },
    ]),
  );

  const asRows = (items) =>
    (items || []).map((c) => (typeof c === 'string' ? { module: c } : c));

  const circuit = asRows(result.circuit);
  if (circuit.length) {
    const keys = Object.keys(circuit[0]);
    container.append(
      el('h2', {}, 'Circuit'),
      dataTable(keys.map((k) => ({ key: k, label: k, mono: k === 'module' })), circuit),
    );
  }
  if (result.edges && result.edges.length) {
    container.append(el('h2', {}, 'Top edges (EAP selection)'), edgesTable(result.edges.slice(0, 30)));
  }
  container.append(kvGrid(result, { skip: ['circuit', 'excluded', 'edges', 'meta'] }));
}

export function renderAtp(result, container) {
  const rows = result.results || (Array.isArray(result) ? result : []);
  if (!rows.length) {
    container.append(el('p', { class: 'empty-hint' }, 'No module scores.'));
    return;
  }
  container.append(
    el('p', { class: 'subtitle' }, 'First-order patch-effect score per module (signed): the fast preview of a full causal trace.'),
    barChart(
      rows.map((r) => ({
        label: r.module ?? '?',
        value: r.score ?? 0,
        tip: `${r.module}\nrole: ${r.role ?? '—'}\nscore: ${fmtNum(r.score)}  rank: ${r.rank ?? '—'}`,
      })),
    ),
  );
}

export function renderEap(result, container) {
  const edges = result.edges || [];
  if (edges.length) {
    container.append(el('h2', {}, `Top edges (${result.meta?.n_edges ?? edges.length} scored)`), edgesTable(edges));
  }
  const items = nodeItems(result.nodes);
  if (items.length) container.append(el('h2', {}, 'Node totals'), barChart(items));
  if (result.meta?.caveat) container.append(el('div', { class: 'panel notice' }, result.meta.caveat));
}

/** Normalize EAP nodes (array of dicts, or a {node: score} map) into bar items. */
function nodeItems(nodes) {
  if (!nodes) return [];
  let items;
  if (Array.isArray(nodes)) {
    items = nodes.map((n) => ({ label: n.node ?? n.name ?? n.module ?? '?', value: n.score ?? n.value ?? 0 }));
  } else if (typeof nodes === 'object') {
    items = Object.entries(nodes).map(([node, score]) => ({
      label: node,
      value: typeof score === 'number' ? score : (score?.score ?? 0),
    }));
  } else {
    return [];
  }
  return items.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 25);
}

function edgesTable(edges) {
  const rows = edges.map((e) => {
    if (Array.isArray(e)) return { src: e[0], dst: e[1], score: e[2] };
    return {
      src: e.src ?? e.from ?? e.source ?? e.upstream ?? '?',
      dst: e.dst ?? e.to ?? e.target ?? e.downstream ?? '?',
      score: e.score ?? e.value ?? 0,
    };
  });
  return dataTable(
    [
      { key: 'src', label: 'From', mono: true },
      { key: 'dst', label: 'To', mono: true },
      { key: 'score', label: 'Score', numeric: true },
    ],
    rows,
  );
}

export function renderMaxact(result, container) {
  const unit = result.unit || {};
  container.append(
    el('p', { class: 'subtitle' },
      `${unit.kind ?? 'unit'} ${unit.index ?? ''} at ${unit.at ?? '?'} — scanned ${result.n_examples_scanned ?? '?'} examples (${result.n_positions_scanned ?? '?'} positions).`),
  );
  const examples = result.examples || [];
  if (!examples.length) {
    container.append(el('p', { class: 'empty-hint' }, 'No activating examples found.'));
    return;
  }
  for (const ex of examples) {
    const panel = el('div', { class: 'panel' });
    panel.append(el('div', { style: 'color:var(--green);font-family:var(--mono);font-size:0.85em;margin-bottom:6px' }, `score ${fmtNum(ex.score ?? ex.max_score)}`));
    if (ex.context_tokens && ex.context_scores) {
      panel.append(tokenStrip(ex.context_tokens, ex.context_scores));
    } else {
      panel.append(el('div', { style: 'font-family:var(--mono);font-size:0.9em;white-space:pre-wrap' }, ex.text ?? ex.context ?? JSON.stringify(ex)));
    }
    container.append(panel);
  }
}
