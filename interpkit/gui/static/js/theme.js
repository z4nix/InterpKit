// Light/dark theme: a `data-theme` attribute on <html> drives the CSS
// token overrides. The initial value is set by an inline script in
// index.html (before first paint) to avoid a flash; this module reads and
// flips it, persisting the choice.

const KEY = 'ik-theme';

export function currentTheme() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

export function applyTheme(theme) {
  const t = theme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', t);
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* private mode — keep the in-memory attribute */
  }
}

export function toggleTheme() {
  applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  return currentTheme();
}
