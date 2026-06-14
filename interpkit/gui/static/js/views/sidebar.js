// Sidebar: ops grouped by category; unsupported ops greyed out with the
// library's own explanation as the tooltip.

import { navigate } from '../router.js';
import { getState } from '../store.js';
import { el } from '../util.js';

export function renderSidebar(activeOp) {
  const { ops, categories, session } = getState();
  const support = session?.support || {};
  const root = el('nav', { class: 'sidebar' });

  for (const cat of categories) {
    const inCategory = ops.filter((o) => o.category === cat.id);
    if (!inCategory.length) continue;
    root.append(el('div', { class: 'cat' }, cat.label));
    for (const op of inCategory) {
      const sup = support[op.name];
      const disabled = sup ? !sup.supported : false;
      root.append(
        el(
          'button',
          {
            class: `op${op.name === activeOp ? ' active' : ''}`,
            disabled,
            title: disabled ? sup.reason : op.description,
            onclick: () => navigate(`#/op/${encodeURIComponent(op.name)}`),
          },
          op.title,
        ),
      );
    }
  }
  return root;
}
