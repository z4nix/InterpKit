// Dedicated chat view: a conversation thread instead of a form.
// History is kept client-side and sent with every turn.

import { pollJob, postJSON } from '../api.js';
import { helpPanel } from '../components/help-panel.js';
import { renderJobError } from '../components/job-status.js';
import { getState } from '../store.js';
import { clear, el } from '../util.js';

export function renderChatView(main, op) {
  const { session } = getState();
  const chat = getState().chat;

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

  // ---- settings ------------------------------------------------------
  const systemInput = el('input', { type: 'text', placeholder: 'optional system prompt', value: chat.system || '' });
  const maxTokens = el('input', { type: 'number', value: '128', min: '1' });
  const sampleCb = el('input', { type: 'checkbox' });
  const temperature = el('input', { type: 'number', value: '1.0', step: 'any' });
  const settings = el(
    'details',
    { class: 'advanced' },
    el('summary', {}, 'Chat settings'),
    el('div', { class: 'panel' },
      el('div', { class: 'field' }, el('label', {}, 'System prompt'), systemInput),
      el('div', { class: 'load-row' },
        el('div', { class: 'field' }, el('label', {}, 'Max new tokens'), maxTokens),
        el('div', { class: 'field' }, el('label', {}, 'Temperature (when sampling)'), temperature),
      ),
      el('div', { class: 'field checkbox-row' }, sampleCb, el('label', {}, 'Sample (instead of greedy)')),
    ),
  );
  inner.append(settings);

  // ---- thread --------------------------------------------------------
  const thread = el('div', { class: 'chat-thread' });
  const statusArea = el('div');
  const input = el('textarea', {
    rows: 2,
    placeholder: 'Message the model… (Enter to send, Shift+Enter for newline)',
    onkeydown: (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    },
  });
  const sendBtn = el('button', { class: 'btn', onclick: send }, 'Send');
  const resetBtn = el('button', {
    class: 'btn secondary',
    title: 'Clear the conversation',
    onclick: () => {
      chat.history = [];
      drawThread();
    },
  }, 'Reset');

  inner.append(thread, statusArea, el('div', { class: 'chat-compose' }, input, sendBtn, resetBtn));
  drawThread();
  input.focus();

  function drawThread() {
    clear(thread);
    if (chat.system) thread.append(el('div', { class: 'chat-msg system' }, `system: ${chat.system}`));
    for (const turn of chat.history) {
      thread.append(el('div', { class: `chat-msg ${turn.role}` }, turn.content));
    }
    if (!chat.history.length) {
      thread.append(el('p', { class: 'empty-hint' }, 'No messages yet — say something. The model replies through its chat template.'));
    }
  }

  async function send() {
    const message = input.value.trim();
    if (!message) return;
    chat.system = systemInput.value.trim() || null;
    input.value = '';
    sendBtn.disabled = true;
    clear(statusArea);

    const priorHistory = [...chat.history];
    chat.history.push({ role: 'user', content: message });
    drawThread();
    const pending = el('div', { class: 'chat-msg assistant' }, el('div', { class: 'spinner small' }));
    thread.append(pending);
    pending.scrollIntoView({ block: 'end' });

    try {
      const { job_id: jobId } = await postJSON(`/api/sessions/${session.id}/ops/chat`, {
        message,
        system: chat.system,
        history: priorHistory,
        max_new_tokens: parseInt(maxTokens.value, 10) || 128,
        sample: sampleCb.checked,
        temperature: parseFloat(temperature.value) || 1.0,
      });
      const job = await pollJob(jobId);
      pending.remove();
      if (job.status === 'done') {
        chat.history.push({ role: 'assistant', content: job.result.response ?? '' });
        drawThread();
        thread.lastChild?.scrollIntoView({ block: 'end' });
      } else {
        chat.history = priorHistory;
        drawThread();
        statusArea.append(renderJobError(job.error || { type: 'Error', message: 'chat failed' }));
      }
    } catch (err) {
      pending.remove();
      chat.history = priorHistory;
      drawThread();
      statusArea.append(renderJobError({ type: `HTTP ${err.status || 'error'}`, message: err.message }));
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }
}
