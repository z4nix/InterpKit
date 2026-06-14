// Job lifecycle UI: spinner / progress bar / cancel button / typed errors.

import { cancelJob } from '../api.js';
import { el } from '../util.js';

const FRIENDLY_ERRORS = {
  OperationNotSupportedForArchitecture: 'This operation does not apply to the loaded architecture.',
  WrongInputType: 'The input type does not match what this model consumes.',
  AttentionBackendUnavailable: 'Real attention weights are unavailable for this model/backend.',
  LensPipelineMismatch: 'The lens pipeline failed its correctness check on this model.',
  ArchitectureNotSupported: 'The architecture resolver could not identify this model.',
  ValueError: null,
  KeyError: null,
};

/**
 * Render a live status panel; call .update(job) on every poll tick.
 * The progress bar is always shown — it animates indeterminately until an
 * op reports a real fraction, then switches to a precise fill.
 *   - cancellable: show a Cancel button (ops support it; model loads do not).
 */
export function createJobStatus({ label = 'Running…', onCancel = null, cancellable = true } = {}) {
  const message = el('span', { class: 'msg' }, label);
  const track = el('div', { class: 'progress-track indeterminate' });
  const fill = el('div', { class: 'progress-fill' });
  track.append(fill);

  let jobId = null;
  const cancelBtn = el(
    'button',
    {
      class: 'btn secondary small',
      onclick: async () => {
        if (!jobId) return;
        cancelBtn.disabled = true;
        cancelBtn.textContent = 'Cancelling…';
        try {
          await cancelJob(jobId);
        } catch {
          /* job may have finished already */
        }
        if (onCancel) onCancel();
      },
    },
    'Cancel',
  );

  const root = el(
    'div',
    { class: 'panel job-status' },
    el('div', { class: 'spinner small' }),
    message,
    track,
    cancellable ? cancelBtn : null,
  );

  function update(job) {
    jobId = job.id;
    if (job.status === 'queued') {
      message.textContent = 'Queued…';
      return;
    }
    if (job.status === 'running') {
      const p = job.progress;
      if (p && p.fraction != null) {
        const pct = `${Math.round(p.fraction * 100)}%`;
        track.classList.remove('indeterminate');
        fill.style.width = pct;
        message.textContent = p.message ? `${p.message} · ${pct}` : `${label} ${pct}`;
      } else {
        track.classList.add('indeterminate');
        message.textContent = (p && p.message) || label;
      }
    }
  }

  return { root, update };
}

/** Render a settled error as a typed panel. */
export function renderJobError(error) {
  const type = error?.type || 'Error';
  const friendly = FRIENDLY_ERRORS[type];
  return el(
    'div',
    { class: 'panel error-panel' },
    el('div', {}, el('span', { class: 'etype' }, type)),
    friendly ? el('div', { style: 'margin-bottom:6px' }, friendly) : null,
    el('div', { class: 'emsg' }, error?.message || 'Unknown error'),
  );
}
