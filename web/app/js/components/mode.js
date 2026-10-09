/* The research depth in the conversation: the mode chip in the top bar, the shallow-mode notice in
 * the empty state, a one-line note above a chat's first answer, and a Settings link under an answer
 * that stopped at its shallow budget.
 *
 * The depth comes from `GET /api/settings` (`mode`, `max_rounds`, `time_budget_sec`) and from every
 * turn's `finish` event, so a switch made in another tab shows after the next reply (and the
 * settings are then read again). Shallow mode means a Gemini key that is not declared paid; the
 * texts point to Settings, where a paid key can be added or the Gemini key marked as paid.
 *
 * The note above the first answer can be dismissed; that is remembered in this browser
 * (`finagent-app.shallow-note`, through state.js's storage, which tolerates a blocked storage).
 * When an answer stopped at its budget in shallow mode, the server's text already says why, so the
 * hint under it is only the mode's name and the link.
 */
import { bindText, formatNumber } from '../i18n.js';
import { emit, on, state, storage, STORAGE_KEYS } from '../state.js';
import { actionButton, el, iconButton, iconSpan, label } from './dom.js';
import { depthOf } from './llm-model.js';
import { closePopover, initPopovers } from './popover.js';

const MODE_KEYS = { shallow: 'mode_shallow', deep: 'mode_deep' };
const DISMISSED = 'dismissed';

let depth = null; // {mode, rounds, seconds}: the newest of the settings and the last finish event
let emptyHost = null; // the empty state's slot for the notice
let popoverReady = false;

const rounds = () => (depth && depth.rounds !== null ? depth.rounds : '–');
const seconds = () => (depth && depth.seconds !== null ? formatNumber(depth.seconds) : '–');

/* Open the side panel on Settings, at the model section. */
export function openModelSettings() {
  closePopover();
  emit('panel:open', 'settings');
  emit('settings:focus', 'model');
}

function settingsLink(key) {
  const link = actionButton('btn-link', key);
  link.addEventListener('click', openModelSettings);
  return link;
}

/* What the chip explains on hover, focus or tap: the limits, and the way to deeper research. */
export function modeExplanation() {
  if (!depth) return null;
  const box = el('div', 'mode-explain');
  const shallow = depth.mode === 'shallow';
  box.append(
    label('p', 'mode-explain-head', MODE_KEYS[depth.mode]),
    label('p', 'mode-explain-text', shallow ? 'mode_shallow_explain' : 'mode_deep_explain', { n: rounds(), s: seconds() }),
    settingsLink(shallow ? 'mode_open_settings' : 'mode_change_settings'),
  );
  return box;
}

function renderChip() {
  const chip = typeof document !== 'undefined' ? document.getElementById('mode-chip') : null;
  if (!chip) return;
  if (!popoverReady && chip.parentElement) {
    popoverReady = true;
    initPopovers(chip.parentElement, { selector: '.chip-mode', build: () => modeExplanation() });
  }
  chip.hidden = !depth;
  if (!depth) return;
  chip.setAttribute('data-mode', depth.mode);
  chip.classList.toggle('is-shallow', depth.mode === 'shallow');
  bindText(chip, MODE_KEYS[depth.mode]);
}

function renderEmpty() {
  if (!emptyHost || !emptyHost.isConnected) return;
  if (!depth || depth.mode !== 'shallow') {
    emptyHost.replaceChildren();
    emptyHost.hidden = true;
    return;
  }
  const notice = el('div', 'notice mode-notice');
  notice.setAttribute('data-tone', 'muted');
  const body = el('div', 'notice-body');
  body.append(label('p', 'notice-text', 'mode_shallow_explain', { n: rounds(), s: seconds() }),
    settingsLink('mode_open_settings'));
  notice.append(iconSpan('info', 18), body);
  emptyHost.replaceChildren(notice);
  emptyHost.hidden = false;
}

function noteDismissed() {
  return storage.get(STORAGE_KEYS.shallowNote) === DISMISSED;
}

/* The one-line shallow note above the chat's first answer: added when a chat's first turn starts
 * or a chat is opened, updated or removed when the mode changes, never added twice. */
function renderNote({ add = false } = {}) {
  const thread = document.getElementById('thread');
  if (!thread) return;
  const existing = thread.querySelector('.mode-note');
  if (!depth || depth.mode !== 'shallow' || noteDismissed()) {
    if (existing) existing.remove();
    return;
  }
  if (existing) {
    bindText(existing.querySelector('.mode-note-text'), 'mode_note', { n: rounds() });
    return;
  }
  const first = thread.querySelector('.turn-assistant');
  if (!add || !first) return;
  const note = el('div', 'mode-note');
  const dismiss = iconButton('close', 'mode_note_dismiss', { size: 14 });
  dismiss.addEventListener('click', () => {
    storage.set(STORAGE_KEYS.shallowNote, DISMISSED);
    note.remove();
  });
  note.append(iconSpan('info', 14), label('span', 'mode-note-text', 'mode_note', { n: rounds() }), dismiss);
  thread.insertBefore(note, first);
}

/* Under an answer that stopped at its shallow budget: the mode and the Settings link. */
function addHint(slot) {
  if (!slot || slot.querySelector('.mode-hint')) return;
  const hint = el('p', 'mode-hint');
  hint.append(iconSpan('info', 14), label('span', 'mode-hint-name', 'mode_shallow'), settingsLink('mode_open_settings'));
  slot.appendChild(hint);
}

function render() {
  renderChip();
  renderEmpty();
  renderNote();
}

on('change', (patch) => {
  if (!('settings' in patch)) return;
  const fresh = depthOf(state.settings);
  if (fresh) depth = fresh;
  render();
});

on('thread:empty', (slot) => {
  emptyHost = slot;
  renderEmpty();
});

on('turn:start', () => renderNote({ add: true }));

on('turn:end', ({ turn, finish } = {}) => {
  const fresh = depthOf(finish);
  if (fresh) {
    const changed = !depth || fresh.mode !== depth.mode || fresh.rounds !== depth.rounds;
    depth = fresh;
    render();
    // Settings changed elsewhere (another tab): read them again, so the panel agrees.
    if (changed && state.settings && state.settings.mode !== fresh.mode) emit('settings:refresh');
  }
  if (finish && finish.budget_exhausted === true && finish.mode === 'shallow') addHint(turn?.slots?.status);
});

on('session:loaded', (detail) => {
  const thread = document.getElementById('thread');
  if (!thread) return;
  // History: the stored answers carry `mode` and `budget_exhausted` in their meta. They pair with
  // the drawn answers by position (thread.js draws one per assistant row).
  const rows = (Array.isArray(detail?.messages) ? detail.messages : [])
    .filter((row) => row && typeof row === 'object' && row.role === 'assistant');
  const turns = [...thread.querySelectorAll('.turn-assistant')];
  if (rows.length === turns.length) {
    rows.forEach((row, index) => {
      const meta = row.meta && typeof row.meta === 'object' ? row.meta : {};
      if (meta.budget_exhausted === true && meta.mode === 'shallow') addHint(turns[index].querySelector('.turn-status'));
    });
  }
  renderNote({ add: true });
});
