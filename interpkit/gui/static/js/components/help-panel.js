// Collapsible, beginner-facing help card shown atop each op panel.
// Content lives in help.js; this module renders it (incl. a tiny inline
// markup pass) and remembers the open/closed preference across panels.

import { HELP } from '../help.js';
import { el } from '../util.js';

const PREF_KEY = 'ik-help-open';

// **bold** | `code` | [label](op-name)
const TOKEN_RE = /\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)/g;

/** Turn a help string with mini-markup into an array of DOM nodes/strings. */
function renderRich(text) {
  const nodes = [];
  let last = 0;
  let m;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(text))) {
    if (m.index > last) nodes.push(...withBreaks(text.slice(last, m.index)));
    if (m[1] != null) nodes.push(el('strong', {}, m[1]));
    else if (m[2] != null) nodes.push(el('code', {}, m[2]));
    else nodes.push(el('a', { class: 'help-link', href: `#/op/${encodeURIComponent(m[4])}` }, m[3]));
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(...withBreaks(text.slice(last)));
  return nodes;
}

function withBreaks(s) {
  const out = [];
  s.split('\n').forEach((part, i) => {
    if (i) out.push(el('br'));
    out.push(part);
  });
  return out;
}

function section(title, content) {
  return el('div', { class: 'help-section' },
    el('div', { class: 'help-section-title' }, title),
    el('div', {}, content),
  );
}

/** Build the help card for an op, or null if no content exists. */
export function helpPanel(opName) {
  const h = HELP[opName];
  if (!h) return null;

  const body = el('div', { class: 'help-body' });
  if (h.what) body.append(section('What it does', renderRich(h.what)));
  if (h.when) body.append(section('When to use it', renderRich(h.when)));
  if (h.workflow && h.workflow.length) {
    const ul = el('ul', { class: 'help-list' });
    for (const item of h.workflow) ul.append(el('li', {}, renderRich(item)));
    body.append(section('Tips & workflow', ul));
  }
  if (h.example) {
    body.append(
      el('div', { class: 'help-example' },
        el('div', { class: 'help-example-label' }, 'Example'),
        el('div', {}, renderRich(h.example)),
      ),
    );
  }

  const summary = el('summary', {},
    el('span', { class: 'help-badge' }, '?'),
    el('span', {}, h.summary || 'How this works & when to use it'),
  );

  const open = localStorage.getItem(PREF_KEY) !== '0';
  const card = el('details', { class: 'panel help-card', open }, summary, body);
  card.addEventListener('toggle', () => localStorage.setItem(PREF_KEY, card.open ? '1' : '0'));
  return card;
}
