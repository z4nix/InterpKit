// Hash router: #/load and #/op/<name>. The op route falls back to the
// load gate when no model session is active.

import { getState } from './store.js';

let renderFn = null;

export function currentRoute() {
  const hash = window.location.hash || '#/op/scan';
  const parts = hash.replace(/^#\//, '').split('/');
  if (parts[0] === 'load') return { view: 'load' };
  if (parts[0] === 'op' && parts[1]) return { view: 'op', op: decodeURIComponent(parts[1]) };
  return { view: 'op', op: 'scan' };
}

export function navigate(hash) {
  if (window.location.hash === hash) {
    if (renderFn) renderFn();
  } else {
    window.location.hash = hash;
  }
}

export function startRouter(render) {
  renderFn = render;
  window.addEventListener('hashchange', () => renderFn());
  renderFn();
}

/** Route guard: ops need a ready session. */
export function effectiveRoute() {
  const route = currentRoute();
  if (route.view === 'op' && !getState().session) return { view: 'load' };
  return route;
}
