// DLA renderer: signed bar chart of per-component logit contributions.

import { barChart, dataTable, statCards } from '../components/charts.js';
import { el, fmtNum } from '../util.js';

export function render(result, container) {
  container.append(
    statCards([
      { label: 'Target token', value: JSON.stringify(result.target_token ?? '?'), kind: 'hl' },
      { label: 'Total logit', value: fmtNum(result.total_logit ?? result.total_logit_pre_ln ?? result.model_logit), kind: 'green' },
    ]),
  );

  const contributions = result.contributions || [];
  if (contributions.length) {
    container.append(
      el('h2', {}, 'Component contributions'),
      barChart(
        contributions.map((c) => ({
          label: c.component ?? c.name ?? '?',
          value: c.logit_contribution ?? c.contribution ?? 0,
          tip: `${c.component ?? '?'}  (${c.type ?? 'component'}${c.layer != null ? `, layer ${c.layer}` : ''})\ncontribution: ${fmtNum(c.logit_contribution ?? c.contribution)}`,
        })),
      ),
    );
  }

  const features = result.feature_contributions;
  if (features && features.features && features.features.length) {
    container.append(
      el('h2', {}, `SAE features at ${features.sae_at ?? '?'}`),
      dataTable(
        [
          { key: 'feature', label: 'Feature', numeric: true },
          { key: 'activation', label: 'Activation', numeric: true },
          { key: 'logit_contribution', label: 'Logit contribution', numeric: true },
        ],
        features.features,
      ),
    );
  }
}
