// Model-load gate: the first thing a user sees. One card, sensible
// defaults, suggestion chips — loading a model should feel like one click.

import { getJSON, pollJob, postJSON } from '../api.js';
import { createJobStatus, renderJobError } from '../components/job-status.js';
import { logoEl } from '../components/logo.js';
import { navigate } from '../router.js';
import { getState, setState } from '../store.js';
import { clear, el } from '../util.js';

const SUGGESTIONS = ['gpt2', 'distilgpt2', 'EleutherAI/pythia-70m', 'HuggingFaceTB/SmolLM2-360M-Instruct'];

export function renderLoadView(main) {
  const { health } = getState();
  clear(main);

  const modelInput = el('input', {
    type: 'text',
    placeholder: 'HuggingFace model ID — e.g. gpt2',
    autocomplete: 'off',
    onkeydown: (e) => {
      if (e.key === 'Enter') load();
    },
  });

  const deviceSel = el('select', {});
  for (const [dev, available] of Object.entries(health?.devices || { cpu: true })) {
    if (!available) continue;
    deviceSel.append(el('option', { value: dev, selected: dev === health?.default_device }, dev));
  }

  const dtypeSel = el('select', {});
  for (const dt of ['float32', 'float16', 'bfloat16', 'auto']) {
    dtypeSel.append(el('option', { value: dt }, dt === 'float32' ? 'float32 (default)' : dt));
  }

  const loadBtn = el('button', { class: 'btn', onclick: load }, 'Load model');
  const statusArea = el('div');

  const chips = el(
    'div',
    { class: 'chips' },
    SUGGESTIONS.map((s) =>
      el('button', { class: 'chip', onclick: () => { modelInput.value = s; load(); } }, s),
    ),
  );

  async function load() {
    const modelId = modelInput.value.trim();
    if (!modelId) {
      modelInput.style.borderColor = 'var(--highlight)';
      return;
    }
    modelInput.style.borderColor = '';
    loadBtn.disabled = true;
    clear(statusArea);
    const status = createJobStatus({
      label: `Loading ${modelId}… (first load downloads the weights)`,
      cancellable: false,
    });
    statusArea.append(status.root);

    try {
      const { session, job_id: jobId } = await postJSON('/api/sessions', {
        model_id: modelId,
        device: deviceSel.value || null,
        dtype: dtypeSel.value === 'float32' ? null : dtypeSel.value,
      });
      let detail = session;
      if (session.status !== 'ready' && jobId) {
        const job = await pollJob(jobId, { onUpdate: status.update });
        if (job.status !== 'done') {
          clear(statusArea);
          statusArea.append(renderJobError(job.error || { type: 'Error', message: 'load failed' }));
          loadBtn.disabled = false;
          return;
        }
        detail = job.result;
      } else if (session.status !== 'ready') {
        detail = await getJSON(`/api/sessions/${session.id}`);
      }
      setState({ session: detail });
      navigate('#/op/scan');
    } catch (err) {
      clear(statusArea);
      statusArea.append(renderJobError({ type: `HTTP ${err.status || 'error'}`, message: err.message }));
      loadBtn.disabled = false;
    }
  }

  const banner = el(
    'div',
    { class: 'load-banner' },
    logoEl('logo-load'),
    el('div', { class: 'load-tagline' }, `Mech interp for any HuggingFace model${health?.version ? `  ·  v${health.version}` : ''}`),
  );

  const card = el(
    'div',
    { class: 'load-card panel' },
    el('h1', {}, 'Load a model'),
    el('p', { class: 'subtitle' }, 'Any HuggingFace model — architecture is detected automatically.'),
    el('div', { class: 'field' }, el('label', {}, 'Model'), modelInput, chips),
    el(
      'div',
      { class: 'load-row' },
      el('div', { class: 'field' }, el('label', {}, 'Device'), deviceSel),
      el('div', { class: 'field' }, el('label', {}, 'Dtype'), dtypeSel),
    ),
    loadBtn,
    statusArea,
  );

  main.append(
    el('div', { class: 'load-wrap' }, el('div', { class: 'load-stack' }, banner, card)),
  );
  modelInput.focus();
}
