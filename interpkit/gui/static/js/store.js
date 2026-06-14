// Shared mutable app state. Views read via getState(); actions mutate via
// setState() and trigger re-renders explicitly through the router.

const state = {
  health: null,        // /api/health payload
  ops: [],             // op catalog entries
  categories: [],      // [{id, label}]
  session: null,       // active session detail (incl. arch + support)
  formValues: {},      // per-op persisted form values: {opName: {...}}
  results: {},         // per-op last shown result: {opName: job}
  history: {},         // per-op job summaries: {opName: [job, ...]} newest first
  chat: { history: [], system: null },
};

export function getState() {
  return state;
}

export function setState(patch) {
  Object.assign(state, patch);
}

export function opByName(name) {
  return state.ops.find((o) => o.name === name) || null;
}

/** Remember the latest finished job for an op and prepend it to history. */
export function recordResult(opName, job) {
  state.results[opName] = job;
  const hist = state.history[opName] || (state.history[opName] = []);
  hist.unshift(job);
  if (hist.length > 20) hist.pop();
}

export function rememberFormValues(opName, values) {
  state.formValues[opName] = values;
}

export function resetSessionState() {
  state.session = null;
  state.formValues = {};
  state.results = {};
  state.history = {};
  state.chat = { history: [], system: null };
}
