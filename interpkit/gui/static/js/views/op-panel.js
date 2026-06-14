// The single generic op page: description → schema-driven form → Run →
// job status → native result rendering (+ raw JSON + history).
// Per-op frontend code lives only in renderers/.

import { pollJob, postJSON } from '../api.js';
import { buildForm } from '../components/form-builder.js';
import { helpPanel } from '../components/help-panel.js';
import { createJobStatus, renderJobError } from '../components/job-status.js';
import { downloadButton, jsonView } from '../components/json-view.js';
import { rendererFor } from '../renderers/index.js';
import { getState, recordResult, rememberFormValues } from '../store.js';
import { clear, el, truncate } from '../util.js';

export function renderOpPanel(main, op) {
  const { session, formValues, results, history } = getState();
  const arch = session?.arch || null;

  clear(main);
  const inner = el('div', { class: 'inner' });
  main.append(inner);

  inner.append(el('h1', {}, op.title), el('p', { class: 'subtitle' }, op.description));

  const help = helpPanel(op.name);
  if (help) inner.append(help);

  const support = session?.support?.[op.name];
  if (support && !support.supported) {
    inner.append(el('div', { class: 'panel notice' }, support.reason));
    return;
  }

  // ---- form -----------------------------------------------------------
  // The landing op (scan) starts prefilled with its example so a fresh
  // session is one click away from real results.
  const initial =
    formValues[op.name] || (op.name === 'scan' ? { text: 'The capital of France is' } : {});
  const form = buildForm(op.fields, arch, initial);
  const runBtn = el('button', { class: 'btn', onclick: run }, 'Run');
  const formPanel = el('div', { class: 'panel' }, form.root, runBtn);
  inner.append(formPanel);

  const statusArea = el('div');
  const resultArea = el('div');
  inner.append(statusArea, resultArea);

  // Restore the last result for this op, if any.
  if (results[op.name]) showResult(results[op.name]);
  renderHistory();

  async function run() {
    const invalid = form.validate();
    if (invalid) {
      clear(statusArea);
      statusArea.append(el('div', { class: 'panel notice' }, invalid));
      return;
    }
    const values = form.getValues();
    rememberFormValues(op.name, values);

    runBtn.disabled = true;
    clear(statusArea);
    clear(resultArea);
    const status = createJobStatus({
      label: op.long_running ? `Running ${op.title}… (this can take a while)` : `Running ${op.title}…`,
    });
    statusArea.append(status.root);

    try {
      const { job_id: jobId } = await postJSON(
        `/api/sessions/${session.id}/ops/${encodeURIComponent(op.name)}`,
        values,
      );
      const job = await pollJob(jobId, { onUpdate: status.update });
      clear(statusArea);
      if (job.status === 'done') {
        recordResult(op.name, job);
        showResult(job);
        renderHistory();
      } else if (job.status === 'cancelled') {
        statusArea.append(el('div', { class: 'panel notice' }, 'Cancelled.'));
      } else {
        statusArea.append(renderJobError(job.error));
      }
    } catch (err) {
      clear(statusArea);
      if (err.status === 422 && Array.isArray(err.detail)) {
        const msgs = err.detail.map((d) => `${(d.loc || []).join('.')}: ${d.msg}`).join('\n');
        statusArea.append(renderJobError({ type: 'ValidationError', message: msgs }));
      } else {
        statusArea.append(renderJobError({ type: `HTTP ${err.status || 'error'}`, message: err.message }));
      }
    } finally {
      runBtn.disabled = false;
    }
  }

  function showResult(job) {
    clear(resultArea);
    const took = job.finished_at && job.started_at ? `${(job.finished_at - job.started_at).toFixed(1)}s` : '';
    resultArea.append(
      el(
        'div',
        { class: 'result-toolbar' },
        el('h2', {}, 'Result'),
        took ? el('span', { class: 'took' }, took) : null,
        downloadButton(job.result, `${op.name}-result.json`),
      ),
    );

    const renderer = rendererFor(op.name);
    const body = el('div');
    resultArea.append(body);
    if (renderer) {
      try {
        renderer(job.result, body, { arch, params: job.params, op: op.name });
      } catch (err) {
        body.append(el('div', { class: 'panel notice' }, `Renderer error (${err.message}) — raw result below.`));
      }
      resultArea.append(jsonView(job.result));
    } else {
      resultArea.append(jsonView(job.result, { open: true, label: 'Result JSON' }));
    }
  }

  function renderHistory() {
    const hist = (getState().history[op.name] || []).slice(0, 10);
    let panel = inner.querySelector('.history-list');
    if (panel) panel.remove();
    if (hist.length < 2) return;
    panel = el('div', { class: 'history-list panel' }, el('div', { class: 'cat', style: 'margin:0 0 6px' }, 'Recent runs'));
    for (const job of hist) {
      const when = new Date(job.created_at * 1000).toLocaleTimeString();
      const label = `${when}  ${truncate(JSON.stringify(job.params), 90)}`;
      panel.append(
        el(
          'button',
          {
            class: job === getState().results[op.name] ? 'current' : '',
            onclick: () => {
              recordSelection(job);
            },
          },
          label,
        ),
      );
    }
    inner.append(panel);

    function recordSelection(job) {
      getState().results[op.name] = job;
      showResult(job);
      renderHistory();
    }
  }
}
