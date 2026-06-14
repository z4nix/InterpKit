// The interpkit wordmark — the exact ANSI Shadow ASCII banner the CLI
// prints (see interpkit/cli/main.py _LOGO_STR), reproduced verbatim so
// the GUI and CLI share one identity. Rendered in a <pre> with the brand
// gradient applied via CSS (.logo-ascii).

import { el } from '../util.js';

export const LOGO_ASCII =
  '██╗███╗   ██╗████████╗███████╗██████╗ ██████╗ ██╗  ██╗██╗████████╗\n' +
  '██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██║ ██╔╝██║╚══██╔══╝\n' +
  '██║██╔██╗ ██║   ██║   █████╗  ██████╔╝██████╔╝█████╔╝ ██║   ██║\n' +
  '██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔═══╝ ██╔═██╗ ██║   ██║\n' +
  '██║██║ ╚████║   ██║   ███████╗██║  ██║██║     ██║  ██╗██║   ██║\n' +
  '╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝   ╚═╝';

/** A <pre> element holding the brand banner. `extra` adds size classes. */
export function logoEl(extra = '') {
  return el('pre', { class: `logo-ascii ${extra}`.trim(), 'aria-label': 'interpkit' }, LOGO_ASCII);
}
