// Boot: fetch health + op catalog, adopt any existing session (server
// restarts are rare, page reloads are not), then start the router.

import { getJSON } from './api.js';
import { logoEl } from './components/logo.js';
import { effectiveRoute, startRouter } from './router.js';
import { getState, opByName, setState } from './store.js';
import { clear, el } from './util.js';
import { renderChatView } from './views/chat.js';
import { renderHeader } from './views/header.js';
import { renderLoadView } from './views/load-model.js';
import { renderOpPanel } from './views/op-panel.js';
import { renderSidebar } from './views/sidebar.js';

const app = document.getElementById('app');

function renderShell() {
  const route = effectiveRoute();
  clear(app);

  const shell = el('div', { class: 'shell' });
  shell.append(renderHeader());

  const main = el('main', { class: 'main' });

  if (route.view === 'load') {
    // Full-width load gate: sidebar is hidden until a model is ready.
    shell.style.gridTemplateColumns = '0 1fr';
    shell.append(el('div'), main);
    renderLoadView(main);
  } else {
    shell.style.gridTemplateColumns = '';
    shell.append(renderSidebar(route.op), main);
    const op = opByName(route.op);
    if (!op) {
      main.append(el('div', { class: 'inner' }, el('h1', {}, 'Unknown operation'), el('p', { class: 'subtitle' }, `No op named "${route.op}".`)));
    } else if (op.name === 'chat') {
      renderChatView(main, op);
    } else {
      renderOpPanel(main, op);
    }
  }

  app.append(shell);
}

async function boot() {
  try {
    const [health, catalog] = await Promise.all([getJSON('/api/health'), getJSON('/api/ops')]);
    setState({ health, ops: catalog.ops, categories: catalog.categories });

    // Adopt the first ready session if the page was reloaded mid-session.
    const { sessions } = await getJSON('/api/sessions');
    const ready = (sessions || []).find((s) => s.status === 'ready');
    if (ready) {
      const detail = await getJSON(`/api/sessions/${ready.id}`);
      setState({ session: detail });
    }

    startRouter(renderShell);
  } catch (err) {
    clear(app);
    app.append(
      el('div', { class: 'splash' },
        logoEl('logo-splash'),
        el('div', { class: 'panel error-panel', style: 'max-width:480px' },
          el('div', {}, el('span', { class: 'etype' }, 'Connection error')),
          el('div', { class: 'emsg' }, `Could not reach the interpkit server: ${err.message}`),
        ),
      ),
    );
  }
}

boot();

// Auto-run scan the first time the user lands on it with a fresh session:
// deliberately NOT done — scan needs an input text. Instead the scan panel
// is the default route with its form prefilled, one click from results.
