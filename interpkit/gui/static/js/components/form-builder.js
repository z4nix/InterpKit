// Schema-driven form builder. The server flattens each op's pydantic
// params model into a field list (see interpkit/gui/ops/base.py); this
// module turns that list into DOM with zero per-op frontend code.

import { el } from '../util.js';

let datalistSeq = 0;

/** Restore a previously entered value onto any of the widget shapes we build. */
function setInputValue(input, field, value) {
  const node = input._input || input; // module-picker wraps its input in a holder
  if (node.type === 'checkbox') node.checked = Boolean(value);
  else node.value = String(value);
}

/**
 * Build a form. Returns {root, getValues, validate}.
 *  - fields: field specs from the op catalog
 *  - arch: serialized arch info (powers module/layer/head pickers); may be null
 *  - initial: previously entered values to restore
 */
export function buildForm(fields, arch, initial = {}) {
  const controls = new Map(); // name -> {field, input, wrap}
  const root = el('div', { class: 'op-form' });

  // ---- partition: plain fields, named groups, advanced ----------------
  const layout = planLayout(fields);

  for (const item of layout.main) root.append(renderItem(item));
  if (layout.advanced.length) {
    const details = el('details', { class: 'advanced' }, el('summary', {}, 'Advanced options'));
    for (const item of layout.advanced) details.append(renderItem(item));
    root.append(details);
  }

  function renderItem(item) {
    if (item.group) {
      const wrap = el('div', { class: 'form-group' }, el('div', { class: 'group-title' }, item.group));
      for (const f of item.fields) wrap.append(renderField(f));
      return wrap;
    }
    return renderField(item.field);
  }

  function renderField(field) {
    const widget = field.ui.widget || defaultWidget(field);
    if (widget === 'hidden') return el('span', { style: 'display:none' });

    const input = makeInput(field, widget);
    const value = initial[field.name];
    if (value !== undefined && value !== null) setInputValue(input, field, value);

    let wrap;
    if (widget === 'checkbox') {
      wrap = el(
        'div',
        { class: 'field checkbox-row' },
        input,
        el('label', { for: input.id }, field.label),
        field.help ? el('div', { class: 'help', style: 'flex-basis:100%' }, field.help) : null,
      );
    } else {
      wrap = el(
        'div',
        { class: 'field' },
        el('label', { for: input.id }, field.label, field.required ? el('span', { class: 'req' }, ' *') : null),
        input,
        field.help ? el('div', { class: 'help' }, field.help) : null,
      );
    }
    controls.set(field.name, { field, input, wrap });
    if (field.ui.show_if) input._showIf = field.ui.show_if;
    return wrap;
  }

  function makeInput(field, widget) {
    const id = `f-${field.name}-${datalistSeq++}`;
    const common = { id, name: field.name };

    if (widget === 'checkbox') {
      const cb = el('input', { ...common, type: 'checkbox' });
      cb.checked = field.default === true;
      cb.addEventListener('change', applyShowIf);
      return cb;
    }
    if (field.type === 'enum') {
      const select = el('select', common);
      if (field.optional) select.append(el('option', { value: '' }, '—'));
      for (const choice of field.choices) {
        select.append(el('option', { value: String(choice), selected: choice === field.default }, String(choice)));
      }
      select.addEventListener('change', applyShowIf);
      return select;
    }
    if (widget === 'layer-select' || widget === 'head-select') {
      const count = widget === 'layer-select' ? arch?.num_layers : arch?.num_attention_heads;
      if (count) {
        const select = el('select', common);
        if (field.optional) select.append(el('option', { value: '' }, widget === 'layer-select' ? 'all layers' : 'all heads'));
        for (let i = 0; i < count; i++) {
          select.append(el('option', { value: String(i) }, `${widget === 'layer-select' ? 'layer' : 'head'} ${i}`));
        }
        return select;
      }
      return el('input', { ...common, type: 'number', placeholder: field.ui.placeholder || '' });
    }
    if (widget === 'module-picker') {
      const listId = `dl-${datalistSeq++}`;
      const input = el('input', {
        ...common,
        type: 'text',
        placeholder: field.ui.placeholder || 'module path…',
        autocomplete: 'off',
      });
      input.setAttribute('list', listId);
      const datalist = el('datalist', { id: listId });
      for (const path of modulePaths(arch)) datalist.append(el('option', { value: path }));
      const holder = el('span', {}, input, datalist);
      holder._input = input;
      holder.id = id;
      return holder;
    }
    if (widget === 'textarea') {
      return el('textarea', { ...common, rows: field.ui.rows || 3, placeholder: field.ui.placeholder || '' });
    }
    if (field.type === 'integer' || field.type === 'number') {
      const input = el('input', { ...common, type: 'number', placeholder: field.ui.placeholder || '' });
      if (field.type === 'number') input.step = 'any';
      if (field.default !== null && field.default !== undefined) input.value = String(field.default);
      return input;
    }
    const input = el('input', { ...common, type: 'text', placeholder: field.ui.placeholder || '' });
    if (field.default) input.value = String(field.default);
    return input;
  }

  function applyShowIf() {
    const values = readValues();
    for (const { input, wrap } of controls.values()) {
      const cond = (input._input || input)._showIf || input._showIf;
      if (!cond) continue;
      const visible = Object.entries(cond).every(([dep, want]) => values[dep] === want);
      wrap.style.display = visible ? '' : 'none';
    }
  }

  function readValues() {
    const values = {};
    for (const { field, input } of controls.values()) {
      const node = input._input || input;
      if (node.type === 'checkbox') {
        values[field.name] = node.checked;
        continue;
      }
      const raw = node.value;
      if (raw === '' || raw == null) {
        values[field.name] = null;
      } else if (field.type === 'integer') {
        values[field.name] = parseInt(raw, 10);
      } else if (field.type === 'number') {
        values[field.name] = parseFloat(raw);
      } else {
        values[field.name] = raw;
      }
    }
    return values;
  }

  function getValues() {
    const values = readValues();
    // Drop nulls so server defaults apply.
    return Object.fromEntries(Object.entries(values).filter(([, v]) => v !== null));
  }

  function validate() {
    let firstBad = null;
    for (const { field, input, wrap } of controls.values()) {
      const node = input._input || input;
      const hidden = wrap.style.display === 'none';
      const missing = field.required && !hidden && (node.value === '' || node.value == null);
      node.style.borderColor = missing ? 'var(--highlight)' : '';
      if (missing && !firstBad) firstBad = field;
    }
    return firstBad ? `"${firstBad.label}" is required.` : null;
  }

  applyShowIf();
  return { root, getValues, validate };
}

function defaultWidget(field) {
  if (field.type === 'boolean') return 'checkbox';
  if (field.type === 'enum') return 'select';
  return 'text';
}

function planLayout(fields) {
  const main = [];
  const advanced = [];
  const groups = new Map();

  for (const field of fields) {
    const groupName = field.ui.group;
    if (groupName) {
      if (!groups.has(groupName)) {
        const item = { group: groupName, fields: [] };
        groups.set(groupName, item);
        // Placement decided when the group completes (below).
        main.push(item); // placeholder position; may move to advanced
      }
      groups.get(groupName).fields.push(field);
    } else if (field.ui.advanced) {
      advanced.push({ field });
    } else {
      main.push({ field });
    }
  }

  // A group belongs in Advanced only if every field in it is advanced.
  const realMain = [];
  for (const item of main) {
    if (item.group && item.fields.every((f) => f.ui.advanced)) advanced.unshift(item);
    else realMain.push(item);
  }
  return { main: realMain, advanced };
}

function modulePaths(arch) {
  if (!arch || !arch.paths) return [];
  const ordered = [
    ...(arch.paths.blocks || []),
    ...(arch.paths.attention || []),
    ...(arch.paths.mlp || []),
    ...(arch.paths.all || []),
  ];
  return [...new Set(ordered)];
}
