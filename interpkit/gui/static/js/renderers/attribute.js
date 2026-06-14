// Attribution renderer: per-token saliency strip.

import { tokenStrip } from '../components/charts.js';
import { el, fmtNum } from '../util.js';

export function render(result, container) {
  const tokens = result.tokens || [];
  const scores = result.scores || [];

  if (result.interpretation === 'ranking_only') {
    container.append(
      el(
        'div',
        { class: 'panel notice' },
        'Ranking-only attribution for this architecture: the ordering of tokens is reliable, the magnitudes are not quantitative.',
      ),
    );
  }

  if (tokens.length && scores.length) {
    container.append(
      el('p', { class: 'subtitle' }, 'Tokens coloured by attribution score (red = pushes toward the target, blue = against).'),
      el('div', { class: 'panel' }, tokenStrip(tokens, scores, { tip: (i, s) => `${JSON.stringify(tokens[i])}\nscore: ${fmtNum(s)}` })),
    );
  } else {
    container.append(el('p', { class: 'empty-hint' }, 'No token-level scores returned (vision attribution writes figures via the CLI/API save flags).'));
  }
}
