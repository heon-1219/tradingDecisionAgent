/* Side-panel views owned by the shell.
 *
 * Portfolio, Orders and Mandate are static placeholders in index.html until Task 4.2 registers
 * real views with `registerPanel`. Settings shows read-only status from GET /api/settings: status
 * words only, never a key. A component that registered `settings` first is left in charge.
 */
import { bindAttr, bindText } from './i18n.js';
import { icon } from './icons.js';
import { panel, registerPanel } from './registry.js';
import { on, state } from './state.js';

// status word from the API -> [i18n key, tone]. Tone is a colour hint; the word carries the meaning.
const STATUS = {
  connected: ['status_connected', 'ok'],
  set: ['status_set', 'ok'],
  missing: ['status_missing', 'muted'],
  invalid: ['status_invalid', 'bad'],
  unreachable: ['status_unreachable', 'warn'],
};

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function statusNode(word) {
  const [key, tone] = STATUS[word] || ['status_unknown', 'muted'];
  const node = el('span', 'status');
  node.setAttribute('data-tone', tone);
  const label = el('span');
  bindText(label, key);
  node.appendChild(label);
  return node;
}

function row(labelKey, value) {
  const wrap = el('div', 'kv-row');
  const term = el('dt');
  bindText(term, labelKey);
  const detail = el('dd');
  detail.appendChild(value);
  wrap.append(term, detail);
  return wrap;
}

function plain(value) {
  const node = el('span', 'kv-value');
  node.textContent = value == null || value === '' ? '–' : String(value);
  return node;
}

function section(headingKey, rows, action) {
  const box = el('section', 'panel-section');
  const head = el('div', 'panel-section-head');
  const heading = el('h2', 'panel-h');
  bindText(heading, headingKey);
  head.appendChild(heading);
  if (action) head.appendChild(action);
  const list = el('dl', 'kv');
  list.append(...rows);
  box.append(head, list);
  return box;
}

export function initPanels({ refreshSettings }) {
  if (panel('settings')) return;
  let container = null;

  function render() {
    if (!container) return;
    const settings = state.settings;
    const active = document.activeElement;
    const keepFocus = Boolean(active) && container.contains(active) && active.classList.contains('icon-btn');
    const refresh = el('button', 'icon-btn');
    refresh.type = 'button';
    bindAttr(refresh, 'aria-label', 'settings_refresh');
    refresh.appendChild(icon('refresh', { size: 16 }));
    refresh.addEventListener('click', () => refreshSettings());

    if (!settings) {
      const note = el('p', 'panel-note');
      if (state.settingsError) note.textContent = state.settingsError;
      else bindText(note, 'settings_loading');
      const head = el('div', 'panel-section-head');
      const heading = el('h2', 'panel-h');
      bindText(heading, 'settings_heading');
      head.append(heading, refresh);
      container.replaceChildren(head, note);
      if (keepFocus) refresh.focus();
      return;
    }

    const trading = el('span', 'status');
    trading.setAttribute('data-tone', settings.trading_enabled ? 'ok' : 'warn');
    const tradingLabel = el('span');
    bindText(tradingLabel, settings.trading_enabled ? 'status_enabled' : 'status_disabled');
    trading.appendChild(tradingLabel);

    const note = el('p', 'panel-note');
    bindText(note, 'settings_note');
    container.replaceChildren(
      section('settings_heading', [
        row('settings_alpaca', statusNode(settings.alpaca)),
        row('settings_tavily', statusNode(settings.tavily)),
        row('settings_sec', statusNode(settings.sec_user_agent)),
        row('settings_gemini', statusNode(settings.gemini)),
      ], refresh),
      section('settings_trading_heading', [
        row('settings_trading', trading),
        row('settings_model', plain(settings.model)),
        row('settings_effort', plain(settings.effort)),
      ]),
      note,
    );
    if (keepFocus) refresh.focus();
  }

  registerPanel('settings', {
    mount(target) {
      container = target;
      render();
    },
    show() {
      refreshSettings();
    },
  });
  on('change', (patch) => {
    if ('settings' in patch || 'settingsError' in patch) render();
  });
}
