// Generate renderer: generated text + optional per-step capture table.

import { dataTable, kvGrid } from '../components/charts.js';
import { el } from '../util.js';

export function render(result, container, ctx) {
  const prompt = result.prompt ?? ctx?.params?.prompt ?? '';
  const text = result.response ?? result.generated ?? result.text ?? '';
  container.append(
    el('div', { class: 'panel', style: 'white-space:pre-wrap;font-family:var(--mono);font-size:0.95em' },
      el('span', { style: 'color:var(--dim)' }, prompt),
      el('span', {}, text.startsWith(prompt) ? text.slice(prompt.length) : text),
    ),
  );

  const interventions = result.interventions;
  if (result.interventions_active || (Array.isArray(interventions) && interventions.length)) {
    const desc = Array.isArray(interventions) ? `: ${interventions.map((i) => (typeof i === 'string' ? i : JSON.stringify(i))).join(', ')}` : '.';
    container.append(el('p', { class: 'subtitle' }, `Interventions were active at every decode step${desc}`));
  }

  const steps = result.steps || [];
  if (steps.length && typeof steps[0] === 'object') {
    const keys = Object.keys(steps[0]).filter((k) => typeof steps[0][k] !== 'object' || Array.isArray(steps[0][k]));
    container.append(
      el('h2', {}, `Per-token capture (${steps.length} steps)`),
      el('div', { class: 'panel', style: 'max-height:380px;overflow:auto' },
        dataTable(keys.map((k) => ({ key: k, label: k })), steps.slice(0, 200)),
      ),
    );
  }

  container.append(kvGrid(result, { skip: ['generated', 'response', 'text', 'output', 'steps', 'prompt', 'interventions'] }));
}
