/* The Settings panel: connection status with key entry, the trading switch, the kill switch, the
 * model in use and the activity-log download. It replaces the shell's read-only view.
 *
 * Keys. The server never sends a key back, only status words, and this panel never keeps one: the
 * entry fields are password inputs with autocomplete off, they are never pre-filled, and they are
 * emptied as soon as a save is answered, whatever the answer. A value goes from the input straight
 * into the PUT body: not into localStorage, not onto the bus, not into a log line or an error text.
 * The SEC contact (a name and an e-mail address, not a secret) uses a plain text input, with the
 * same rules.
 *
 * The panel is built once and then updated in place, so a field being typed in is never replaced.
 */
import { request } from '../api.js';
import { bindText, t } from '../i18n.js';
import { announce } from '../notify.js';
import { registerPanel } from '../registry.js';
import { emit, on, setState, state } from '../state.js';
import { actionButton, el, iconButton, iconSpan, label, str, uid } from './dom.js';
import { createKillSwitch } from './settings-kill.js';

// status word from the API -> [i18n key, tone]
const STATUS = {
  connected: ['status_connected', 'ok'],
  set: ['status_set', 'ok'],
  missing: ['status_missing', 'muted'],
  invalid: ['status_invalid', 'bad'],
  unreachable: ['status_unreachable', 'warn'],
  not_paper: ['status_not_paper', 'bad'], // keys of a live account: the server keeps no broker for them
};
// error code from PUT /api/settings -> i18n key
const ERRORS = {
  alpaca_invalid: 'settings_err_alpaca_invalid',
  alpaca_unreachable: 'settings_err_alpaca_unreachable',
  alpaca_pair_required: 'settings_err_alpaca_pair',
  alpaca_not_paper: 'settings_err_alpaca_not_paper',
  invalid_value: 'settings_err_invalid_value',
};
const FIELD_LABEL = {
  alpaca_key_id: 'settings_alpaca_key_id',
  alpaca_secret: 'settings_alpaca_secret',
  tavily_key: 'settings_tavily_key',
  sec_user_agent: 'settings_sec_value',
};

// Each editable connection: the status field it reports, and the request fields it sends.
const CONNECTIONS = [
  { status: 'alpaca', key: 'settings_alpaca', fields: [['alpaca_key_id', true], ['alpaca_secret', true]] },
  { status: 'tavily', key: 'settings_tavily', fields: [['tavily_key', true]] },
  { status: 'sec_user_agent', key: 'settings_sec', fields: [['sec_user_agent', false]] },
];

let container = null;
let view = null;

/* Apply a settings body the server just returned, the same way the shell's own refresh does. */
function adopt(settings) {
  setState({ settings, settingsError: null });
  emit('settings', settings);
}

function statusNode() {
  const node = el('span', 'status');
  node.appendChild(el('span'));
  return node;
}

function setStatus(node, word) {
  const [key, tone] = STATUS[word] || ['status_unknown', 'muted'];
  node.setAttribute('data-tone', tone);
  bindText(node.firstChild, key);
}

/* What to say for a failed settings write. Never includes anything that was typed. */
export function settingsErrorText(err) {
  const key = ERRORS[err?.error];
  if (!key) return err?.message || t('err_generic');
  const field = FIELD_LABEL[err?.body?.field];
  return key === 'settings_err_invalid_value' && field ? `${t(field)}: ${t(key)}` : t(key);
}

function connectionRow(spec) {
  const row = el('div', 'conn');
  const head = el('div', 'conn-head');
  const status = statusNode();
  const edit = actionButton('btn-link', 'settings_edit');
  edit.setAttribute('aria-expanded', 'false');
  head.append(label('span', 'conn-name', spec.key), status, edit);

  const form = el('div', 'conn-form');
  form.hidden = true;
  form.id = uid('conn');
  edit.setAttribute('aria-controls', form.id);
  const inputs = spec.fields.map(([name, secret]) => {
    const id = uid('setting');
    const caption = label('label', 'field-label', FIELD_LABEL[name]);
    caption.setAttribute('for', id);
    const input = el('input', 'field-input');
    input.id = id;
    input.type = secret ? 'password' : 'text';
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.autocapitalize = 'off';
    input.maxLength = 256;
    input.setAttribute('data-field', name);
    const wrap = el('div', 'field');
    wrap.append(caption, input);
    form.appendChild(wrap);
    return input;
  });
  const message = el('p', 'conn-message');
  message.hidden = true;
  const actions = el('div', 'conn-actions');
  const save = actionButton('btn btn-primary btn-small', 'settings_save');
  const clear = actionButton('btn btn-quiet btn-small', 'settings_clear');
  actions.append(save, clear);
  form.append(message, actions);
  row.append(head, form);

  function say(text, tone) {
    message.hidden = !text;
    message.textContent = text || '';
    message.setAttribute('data-tone', tone || 'muted');
  }

  async function send(body, doneKey) {
    save.disabled = true;
    clear.disabled = true;
    say(t('settings_saving'));
    try {
      adopt(await request('PUT', '/api/settings', body));
      say(t(doneKey), 'ok');
      announce(t(doneKey));
      emit('portfolio:refresh');
    } catch (err) {
      say(settingsErrorText(err), 'bad');
    } finally {
      for (const input of inputs) input.value = ''; // never keep what was typed, whatever the answer
      save.disabled = false;
      clear.disabled = false;
    }
  }

  edit.addEventListener('click', () => {
    form.hidden = !form.hidden;
    edit.setAttribute('aria-expanded', String(!form.hidden));
    if (form.hidden) for (const input of inputs) input.value = '';
    else inputs[0].focus();
  });
  save.addEventListener('click', () => {
    if (inputs.some((input) => input.value.trim() === '')) {
      say(t(inputs.length > 1 ? 'settings_err_alpaca_pair' : 'settings_err_empty'), 'bad');
      return;
    }
    send(Object.fromEntries(inputs.map((input) => [input.getAttribute('data-field'), input.value])), 'settings_saved');
  });
  clear.addEventListener('click', () => {
    // "" removes a key from the server's settings and its .env file.
    send(Object.fromEntries(inputs.map((input) => [input.getAttribute('data-field'), ''])), 'settings_cleared');
  });
  form.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.isComposing && event.target.tagName === 'INPUT') {
      event.preventDefault();
      save.click();
    }
  });
  return { el: row, setStatus: (word) => setStatus(status, word) };
}

function build() {
  const head = el('div', 'panel-section-head');
  const refresh = iconButton('refresh', 'settings_refresh');
  refresh.addEventListener('click', () => emit('settings:refresh'));
  head.append(label('h2', 'panel-h', 'settings_heading'), refresh);
  const problem = el('p', 'panel-note');
  problem.hidden = true;

  const connections = CONNECTIONS.map((spec) => ({ spec, row: connectionRow(spec) }));
  const gemini = el('div', 'conn');
  const geminiHead = el('div', 'conn-head');
  const geminiStatus = statusNode();
  geminiHead.append(label('span', 'conn-name', 'settings_gemini'), geminiStatus);
  gemini.appendChild(geminiHead);
  const fallbacks = el('p', 'panel-note');
  fallbacks.hidden = true;
  const connSection = el('section', 'panel-section');
  connSection.append(head, problem, ...connections.map((c) => c.row.el), gemini, fallbacks,
    label('p', 'panel-note', 'settings_keys_note'));

  // Trading: the switch, then the kill switch.
  const toggle = el('button', 'switch');
  toggle.type = 'button';
  toggle.setAttribute('role', 'switch');
  toggle.id = uid('switch');
  const toggleLabel = label('label', 'switch-label', 'settings_trading');
  toggleLabel.setAttribute('for', toggle.id);
  const toggleState = el('span', 'switch-state');
  const thumb = el('span', 'switch-thumb');
  thumb.setAttribute('aria-hidden', 'true');
  toggle.appendChild(thumb);
  const toggleRow = el('div', 'switch-row');
  toggleRow.append(toggleLabel, toggleState, toggle);
  const toggleMessage = el('p', 'conn-message');
  toggleMessage.hidden = true;
  toggle.addEventListener('click', async () => {
    const next = toggle.getAttribute('aria-checked') !== 'true';
    toggle.disabled = true;
    toggleMessage.hidden = true;
    try {
      adopt(await request('PUT', '/api/settings', { trading_enabled: next }));
      announce(t(next ? 'settings_trading_on' : 'settings_trading_off'));
    } catch (err) {
      toggleMessage.hidden = false;
      toggleMessage.setAttribute('data-tone', 'bad');
      toggleMessage.textContent = settingsErrorText(err);
    }
    toggle.disabled = false;
  });
  const kill = createKillSwitch({ adopt });
  const tradingSection = el('section', 'panel-section');
  tradingSection.append(label('h2', 'panel-h', 'settings_trade_heading'), toggleRow, toggleMessage, kill.el);

  const model = el('span', 'kv-value');
  const effort = el('span', 'kv-value');
  const agent = el('dl', 'kv');
  for (const [key, value] of [['settings_model', model], ['settings_effort', effort]]) {
    const row = el('div', 'kv-row');
    const detail = el('dd');
    detail.appendChild(value);
    row.append(label('dt', '', key), detail);
    agent.appendChild(row);
  }
  const agentSection = el('section', 'panel-section');
  agentSection.append(label('h2', 'panel-h', 'settings_agent_heading'), agent);

  const log = el('a', 'btn btn-quiet log-link');
  log.append(iconSpan('download', 16), label('span', 'btn-label', 'settings_log_download'));
  const logNote = label('p', 'panel-note', 'settings_log_note');
  const logSection = el('section', 'panel-section');
  logSection.append(label('h2', 'panel-h', 'settings_log_heading'), log, logNote);

  container.replaceChildren(connSection, tradingSection, agentSection, logSection);
  view = { connections, geminiStatus, fallbacks, problem, toggle, toggleState, kill, model, effort, log, logNote };
}

function update() {
  if (!view) return;
  const settings = state.settings;
  view.problem.hidden = Boolean(settings) || !state.settingsError;
  view.problem.textContent = settings ? '' : str(state.settingsError);
  for (const { spec, row } of view.connections) row.setStatus(settings ? settings[spec.status] : undefined);
  setStatus(view.geminiStatus, settings ? settings.gemini : undefined);

  const names = settings && Array.isArray(settings.env_fallbacks)
    ? settings.env_fallbacks.filter((name) => typeof name === 'string' && /^[A-Z0-9_]{1,64}$/.test(name)) : [];
  view.fallbacks.hidden = names.length === 0;
  if (names.length) bindText(view.fallbacks, 'settings_env_fallbacks', { names: names.join(', ') });

  const enabled = Boolean(settings && settings.trading_enabled);
  view.toggle.setAttribute('aria-checked', String(enabled));
  view.toggle.disabled = !settings;
  bindText(view.toggleState, enabled ? 'status_enabled' : 'status_disabled');
  view.kill.update(settings);
  view.model.textContent = settings && settings.model ? String(settings.model) : '–';
  view.effort.textContent = settings && settings.effort ? String(settings.effort) : '–';

  const sessionId = state.sessionId;
  view.log.hidden = !sessionId;
  bindText(view.logNote, sessionId ? 'settings_log_note' : 'settings_log_none');
  if (sessionId) {
    view.log.setAttribute('href', `/api/ledger/${encodeURIComponent(sessionId)}.jsonl`);
    view.log.setAttribute('download', `ledger-${sessionId}.jsonl`);
  } else {
    view.log.removeAttribute('href');
  }
}

registerPanel('settings', {
  mount(target) {
    container = target;
    build();
    update();
  },
  show() {
    emit('settings:refresh');
  },
});

on('change', (patch) => {
  if ('settings' in patch || 'settingsError' in patch || 'sessionId' in patch) update();
});
