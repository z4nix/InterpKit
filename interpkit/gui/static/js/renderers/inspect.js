// Inspect renderer: detected architecture summary + module tree.

import { dataTable, statCards } from '../components/charts.js';
import { el, fmtParams } from '../util.js';

export function render(result, container) {
  container.append(
    statCards([
      { label: 'Family', value: result.family ?? '?', kind: 'hl' },
      { label: 'Layers', value: result.num_layers ?? '—' },
      { label: 'Hidden size', value: result.hidden_size ?? '—' },
      { label: 'Heads', value: result.num_attention_heads ?? '—' },
      { label: 'Parameters', value: fmtParams(result.total_params), kind: 'green' },
    ]),
  );

  if (result.summary) {
    container.append(el('div', { class: 'panel', style: 'font-family:var(--mono);font-size:0.86em;white-space:pre-wrap' }, result.summary));
  }

  const blocks = result.blocks || [];
  if (blocks.length) {
    container.append(
      el('h2', {}, `Blocks (${blocks.length})`),
      dataTable(
        [
          { key: 'path', label: 'Path', mono: true },
          { key: 'mechanism', label: 'Mechanism' },
          { key: 'has_attention', label: 'Attention' },
          { key: 'has_residual', label: 'Residual' },
        ],
        blocks,
      ),
    );
  }

  const modules = result.modules || [];
  if (modules.length) {
    container.append(
      el(
        'details',
        {},
        el('summary', { style: 'cursor:pointer;color:var(--dim);margin:12px 0' }, `Module tree (${modules.length} modules)`),
        el('div', { style: 'max-height:420px;overflow:auto' },
          dataTable(
            [
              { key: 'name', label: 'Module', mono: true },
              { key: 'type', label: 'Type' },
              { key: 'param_count', label: 'Params', numeric: true, fmt: fmtParams },
              { key: 'role', label: 'Role' },
            ],
            modules,
          ),
        ),
      ),
    );
  }
}
