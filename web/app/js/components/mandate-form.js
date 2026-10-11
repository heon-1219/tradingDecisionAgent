/* The mandate form: one editable view of the `Mandate` fields, used by the draft card in the
 * conversation and by the Mandate panel.
 *
 * Values are read as text and sent as text: amounts and percentages go to the server as decimal
 * strings (never through a float), counts as digit strings, which the server's model parses. The
 * form only checks the shape of what was typed; the server is the authority, and its 422
 * `fields: [{loc, msg}]` are shown next to the inputs. Symbol lists are comma-separated and
 * upper-cased. `universe` and `version` are not editable: the first is kept as it came, the second
 * is assigned by the server on every save.
 */
import { bindAttr, bindText } from '../i18n.js';
import { el, label, str, uid } from './dom.js';

const DECIMAL = /^\d{1,12}(\.\d{1,6})?$/;
const DIGITS = /^\d{1,6}$/;

// name, kind, label key, and (for text kinds) the unit shown beside the input.
export const MANDATE_FIELDS = [
  { name: 'direction_text', kind: 'textarea', key: 'mandate_direction', wide: true },
  { name: 'budget_usd', kind: 'decimal', key: 'mandate_budget', unit: '$', optional: true },
  { name: 'max_order_usd', kind: 'decimal', key: 'mandate_max_order', unit: '$', optional: true },
  { name: 'max_position_pct', kind: 'decimal', key: 'mandate_max_position', unit: '%' },
  { name: 'cash_buffer_pct', kind: 'decimal', key: 'mandate_cash_buffer', unit: '%' },
  { name: 'price_band_pct', kind: 'decimal', key: 'mandate_price_band', unit: '%' },
  { name: 'max_positions', kind: 'count', key: 'mandate_max_positions' },
  { name: 'max_orders_per_day', kind: 'count', key: 'mandate_max_orders' },
  { name: 'horizon', kind: 'text', key: 'mandate_horizon' },
  { name: 'allow_symbols', kind: 'symbols', key: 'mandate_allow', wide: true },
  { name: 'deny_symbols', kind: 'symbols', key: 'mandate_deny', wide: true },
  { name: 'order_types', kind: 'choices', key: 'mandate_order_types', options: ['limit', 'market'] },
  { name: 'allow_fractional', kind: 'toggle', key: 'mandate_fractional' },
];

/* "voo, vti  bnd" -> ["VOO", "VTI", "BND"] (first-seen order, no duplicates). */
export function parseSymbols(text) {
  const out = [];
  for (const part of str(text).split(/[\s,;]+/)) {
    const symbol = part.trim().toUpperCase();
    if (symbol && !out.includes(symbol)) out.push(symbol);
  }
  return out;
}

/* The text an input shows for a mandate value. */
export function fieldText(field, value) {
  if (value === null || value === undefined) return '';
  if (field.kind === 'symbols') return Array.isArray(value) ? value.join(', ') : '';
  return String(value);
}

/* Read raw input texts into a request body. Returns {payload, errors}, where `errors` maps a field
 * name to an i18n key for what is wrong with its shape. Pure: used by the form and by the tests. */
export function readMandate(raw, base = {}) {
  const payload = { universe: base.universe || 'us_equity' };
  const errors = {};
  for (const field of MANDATE_FIELDS) {
    const value = raw[field.name];
    if (field.kind === 'toggle') {
      payload[field.name] = Boolean(value);
    } else if (field.kind === 'choices') {
      payload[field.name] = field.options.filter((option) => Array.isArray(value) && value.includes(option));
      if (payload[field.name].length === 0) errors[field.name] = 'mandate_err_choice';
    } else if (field.kind === 'symbols') {
      payload[field.name] = parseSymbols(value);
    } else if (field.kind === 'decimal' || field.kind === 'count') {
      const text = str(value).trim().replaceAll(',', '');
      if (text === '') {
        if (field.optional) payload[field.name] = null;
        else errors[field.name] = 'mandate_err_required';
      } else if (!(field.kind === 'decimal' ? DECIMAL : DIGITS).test(text)) {
        errors[field.name] = field.kind === 'decimal' ? 'mandate_err_number' : 'mandate_err_count';
      } else {
        payload[field.name] = text; // a string: the server parses it exactly
      }
    } else {
      payload[field.name] = str(value).trim();
    }
  }
  return { payload, errors };
}

/* Build the form. `onSave(payload)` is called with a shape-checked body; the caller sends it and
 * reports back with `setErrors` (422) or `setMandate` (saved). */
export function createMandateForm({ mandate, onSave, onChange }) {
  const root = el('div', 'mandate-form');
  const controls = new Map(); // name -> {field, input(s), error}
  let base = mandate || {};

  for (const field of MANDATE_FIELDS) {
    const row = el('div', field.wide ? 'field is-wide' : 'field');
    row.setAttribute('data-field', field.name);
    const id = uid('mandate');
    const error = el('p', 'field-error');
    error.id = `${id}-error`;
    error.hidden = true;
    const control = { field, error, inputs: [] };

    if (field.kind === 'toggle' || field.kind === 'choices') {
      const group = el(field.kind === 'choices' ? 'fieldset' : 'div', 'field-checks');
      if (field.kind === 'choices') group.appendChild(label('legend', 'field-label', field.key));
      for (const option of field.kind === 'choices' ? field.options : [null]) {
        const wrap = el('label', 'check');
        const box = el('input');
        box.type = 'checkbox';
        if (option) box.value = option;
        wrap.append(box, label('span', '', option ? `order_type_${option}` : field.key));
        group.appendChild(wrap);
        control.inputs.push(box);
      }
      row.appendChild(group);
    } else {
      const caption = label('label', 'field-label', field.key);
      caption.setAttribute('for', id);
      const input = el(field.kind === 'textarea' ? 'textarea' : 'input', 'field-input');
      input.id = id;
      input.autocomplete = 'off';
      input.spellcheck = false;
      if (field.kind === 'textarea') {
        input.rows = 2;
        input.maxLength = 2000;
        input.spellcheck = true;
      } else {
        input.type = 'text';
        input.maxLength = field.kind === 'symbols' ? 600 : 80;
        if (field.kind === 'decimal') input.inputMode = 'decimal';
        if (field.kind === 'count') input.inputMode = 'numeric';
        if (field.kind === 'symbols') {
          input.autocapitalize = 'characters';
          bindAttr(input, 'placeholder', 'mandate_symbols_hint');
        }
        if (field.optional) bindAttr(input, 'placeholder', 'mandate_no_limit');
      }
      input.setAttribute('aria-describedby', error.id);
      const box = el('div', 'field-box');
      if (field.unit) {
        const unit = el('span', 'field-unit', field.unit);
        unit.setAttribute('aria-hidden', 'true');
        box.classList.add(field.unit === '$' ? 'has-prefix' : 'has-suffix');
        box.append(...(field.unit === '$' ? [unit, input] : [input, unit]));
      } else {
        box.appendChild(input);
      }
      row.append(caption, box);
      control.inputs.push(input);
    }
    row.appendChild(error);
    controls.set(field.name, control);
    root.appendChild(row);
  }

  const actions = el('div', 'mandate-actions');
  const save = el('button', 'btn btn-primary');
  save.type = 'button';
  bindText(save, 'mandate_save');
  const status = el('p', 'mandate-status');
  status.hidden = true;
  actions.append(save, status);
  root.appendChild(actions);

  function fill(values) {
    for (const { field, inputs } of controls.values()) {
      const value = values[field.name];
      if (field.kind === 'toggle') inputs[0].checked = Boolean(value);
      else if (field.kind === 'choices') {
        for (const box of inputs) box.checked = Array.isArray(value) && value.includes(box.value);
      } else inputs[0].value = fieldText(field, value);
    }
  }

  function raw() {
    const out = {};
    for (const { field, inputs } of controls.values()) {
      if (field.kind === 'toggle') out[field.name] = inputs[0].checked;
      else if (field.kind === 'choices') out[field.name] = inputs.filter((b) => b.checked).map((b) => b.value);
      else out[field.name] = inputs[0].value;
    }
    return out;
  }

  /* `errors`: field name -> {key} (an i18n key) or {text} (the server's own message). */
  function showErrors(errors) {
    let first = null;
    for (const [name, control] of controls) {
      const problem = errors[name];
      control.error.hidden = !problem;
      for (const input of control.inputs) input.setAttribute('aria-invalid', problem ? 'true' : 'false');
      if (!problem) {
        control.error.removeAttribute('data-i18n');
        control.error.textContent = '';
        continue;
      }
      if (problem.key) bindText(control.error, problem.key);
      else {
        control.error.removeAttribute('data-i18n');
        control.error.textContent = problem.text;
      }
      first = first || control.inputs[0];
    }
    return first;
  }

  function setStatus(key, tone, text) {
    status.hidden = !key && !text;
    status.setAttribute('data-tone', tone || 'muted');
    if (key) bindText(status, key);
    else {
      status.removeAttribute('data-i18n');
      status.textContent = text || '';
    }
  }

  root.addEventListener('input', () => {
    setStatus(null);
    if (onChange) onChange();
  });
  root.addEventListener('keydown', (event) => {
    // Enter in a one-line field saves; in the text area it adds a line.
    if (event.key === 'Enter' && !event.isComposing && event.target.tagName === 'INPUT'
      && event.target.type === 'text') {
      event.preventDefault();
      save.click();
    }
  });
  save.addEventListener('click', () => {
    const { payload, errors } = readMandate(raw(), base);
    const shaped = Object.fromEntries(Object.entries(errors).map(([name, key]) => [name, { key }]));
    const first = showErrors(shaped);
    if (first) {
      setStatus('mandate_fix', 'bad');
      first.focus();
      return;
    }
    onSave(payload);
  });

  fill(base);
  return {
    el: root,
    setBusy(busy) {
      save.disabled = busy;
      if (busy) setStatus('mandate_saving', 'muted');
    },
    /* Replace the values (after a save, or when the panel loads the stored mandate). */
    setMandate(next) {
      base = next || {};
      fill(base);
      showErrors({});
    },
    /* A 422 from the server: `fields` is [{loc, msg}]; `message` is the fallback line. */
    setErrors(fields, message) {
      const errors = {};
      const other = [];
      for (const item of Array.isArray(fields) ? fields : []) {
        const name = str(item?.loc).split('.')[0];
        if (controls.has(name)) errors[name] = errors[name] || { text: str(item.msg) || str(message) };
        else if (item?.msg) other.push(str(item.msg));
      }
      const first = showErrors(errors);
      setStatus(null, 'bad', other[0] || str(message));
      if (first) first.focus();
    },
    setStatus,
  };
}
