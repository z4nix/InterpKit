// Renderer registry: op name → render(result, container, ctx).
// Unmapped ops fall back to the raw JSON view; every result additionally
// shows the JSON view collapsed underneath (see op-panel).

import { render as renderAttention } from './attention.js';
import { render as renderAttribute } from './attribute.js';
import {
  renderAtp,
  renderEap,
  renderFindCircuit,
  renderMaxact,
} from './circuits.js';
import { render as renderDla } from './dla.js';
import { render as renderFeatures } from './features.js';
import { render as renderGenerate } from './generate.js';
import { render as renderInspect } from './inspect.js';
import { render as renderLens } from './lens.js';
import {
  renderActivations,
  renderChatResult,
  renderDecompose,
  renderKv,
  renderPatch,
  renderProbe,
  renderReport,
  renderTunedLens,
} from './misc.js';
import { render as renderScan } from './scan.js';
import { render as renderSteer } from './steer.js';
import { render as renderTrace } from './trace.js';

const RENDERERS = {
  scan: renderScan,
  inspect: renderInspect,
  report: renderReport,
  lens: renderLens,
  dla: renderDla,
  attribute: renderAttribute,
  attention: renderAttention,
  activations: renderActivations,
  trace: renderTrace,
  patch: renderPatch,
  ablate: renderPatch,
  decompose: renderDecompose,
  diff: renderKv,
  probe: renderProbe,
  steer: renderSteer,
  generate: renderGenerate,
  chat: renderChatResult,
  features: renderFeatures,
  'find-circuit': renderFindCircuit,
  atp: renderAtp,
  eap: renderEap,
  maxact: renderMaxact,
  'train-tuned-lens': renderTunedLens,
};

export function rendererFor(opName) {
  return RENDERERS[opName] || null;
}
