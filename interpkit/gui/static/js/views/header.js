// Header bar: brand + active model chip with unload.

import { del } from '../api.js';
import { logoEl } from '../components/logo.js';
import { navigate } from '../router.js';
import { getState, resetSessionState, setState } from '../store.js';
import { currentTheme, toggleTheme } from '../theme.js';
import { el } from '../util.js';

const themeIcon = (t) => (t === 'dark' ? '☀' : '☾');

export function renderHeader() {
  const { session, health } = getState();
  const header = el('header', { class: 'header' },
    logoEl('logo-header'),
    el('div', { class: 'grow' }),
  );

  const themeBtn = el(
    'button',
    {
      class: 'icon-btn',
      title: 'Toggle light / dark theme',
      'aria-label': 'Toggle theme',
      onclick: () => {
        themeBtn.textContent = themeIcon(toggleTheme());
      },
    },
    themeIcon(currentTheme()),
  );
  header.append(themeBtn);

  if (session) {
    header.append(
      el(
        'div',
        { class: 'model-chip' },
        el('span', {}, session.model_id),
        el('span', { class: 'meta' }, `${session.device ?? ''} · ${(session.dtype ?? '').replace('torch.', '')}`),
        el(
          'button',
          {
            title: 'Unload this model and free memory',
            onclick: async () => {
              try {
                await del(`/api/sessions/${session.id}`);
              } catch {
                /* already gone */
              }
              resetSessionState();
              setState({});
              navigate('#/load');
            },
          },
          'unload',
        ),
      ),
    );
  }

  header.append(el('span', { style: 'color:var(--dim);font-size:0.82em;margin-left:14px' }, `v${health?.version ?? ''}`));
  return header;
}
