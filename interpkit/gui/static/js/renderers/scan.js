// Scan renderer: prediction + ranked key findings — the landing view.

import { barChart } from '../components/charts.js';
import { el, fmtPct } from '../util.js';

export function render(result, container) {
  const pred = result.prediction;
  if (pred && pred.top5_tokens) {
    container.append(
      el('h2', {}, 'Prediction'),
      el(
        'div',
        { class: 'panel' },
        barChart(
          pred.top5_tokens.map((t, i) => ({
            label: JSON.stringify(t),
            value: pred.top5_probs?.[i] ?? 0,
          })),
          { fmt: fmtPct },
        ),
      ),
    );
  }

  const findings = result.key_findings || result.findings || [];
  if (findings.length) {
    container.append(el('h2', {}, 'Key findings'));
    const list = el('div', { class: 'panel' });
    for (const finding of findings) {
      const text = typeof finding === 'string' ? finding : finding.text || JSON.stringify(finding);
      const section = typeof finding === 'object' ? finding.section : null;
      list.append(
        el(
          'div',
          { style: 'padding:6px 0;border-bottom:1px solid var(--accent)' },
          section ? el('span', { style: 'color:var(--green);font-size:0.8em;text-transform:uppercase;margin-right:10px;letter-spacing:0.6px' }, section) : null,
          text,
        ),
      );
    }
    container.append(list);
  }

  container.append(
    el('p', { class: 'empty-hint' }, 'Dig deeper with the dedicated panels in the sidebar — DLA, logit lens, attention, attribution.'),
  );
}
