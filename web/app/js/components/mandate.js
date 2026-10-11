/* The mandate: the draft card in the conversation and the Mandate panel.
 *
 * `data-mandate {mandate, draft}` (the agent's `draft_mandate` tool) shows an editable card with a
 * "Draft" badge. A draft is not in effect: only Save (`PUT /api/mandate/{session}`) stores it, and
 * the server assigns the version. After a save the badge shows the saved version, and the panel
 * follows. History keeps no copy of a draft, so a reloaded chat shows a short note pointing at the
 * panel instead of the form.
 *
 * The panel shows the mandate in effect for the open chat (the saved one, or the defaults), with
 * the same form.
 */
import { ApiError, request } from '../api.js';
import { bindText, t } from '../i18n.js';
import { announce } from '../notify.js';
import { registerEvent, registerPanel } from '../registry.js';
import { emit, on, state } from '../state.js';
import { actionButton, badge, el, extraSection, iconSpan, label } from './dom.js';
import { createMandateForm } from './mandate-form.js';

/* Send a mandate. Resolves the saved mandate, or reports the failure on the form (and announces one
 * that is not about its fields) and resolves null. */
export async function saveMandate(sessionId, payload, form) {
  form.setBusy(true);
  try {
    const body = await request('PUT', `/api/mandate/${encodeURIComponent(sessionId)}`, payload);
    form.setBusy(false);
    return body && body.mandate ? body.mandate : null;
  } catch (err) {
    form.setBusy(false);
    if (err instanceof ApiError && err.status === 422) form.setErrors(err.fields, err.message);
    else {
      form.setStatus(null, 'bad', err.message || t('err_generic'));
      announce(err.message || t('err_generic'));
    }
    return null;
  }
}

function setBadge(node, tone, key, vars) {
  node.setAttribute('data-tone', tone);
  bindText(node, key, vars);
}

// --- the draft card --------------------------------------------------------------------------------

function draftCard(turn, mandate) {
  const card = el('section', 'card mandate-card');
  const head = el('header', 'card-head');
  const title = el('div', 'card-title');
  title.append(iconSpan('shield', 18), label('h3', 'card-h', 'mandate_card_title'));
  const state_ = badge('warn', 'mandate_draft');
  head.append(title, state_);
  const note = label('p', 'card-note', 'mandate_draft_note');
  const form = createMandateForm({
    mandate,
    onChange() {
      if (card.getAttribute('data-saved') === 'true') setBadge(state_, 'warn', 'mandate_unsaved');
    },
    async onSave(payload) {
      const saved = await saveMandate(turn.sessionId, payload, form);
      if (!saved) return;
      card.setAttribute('data-saved', 'true');
      form.setMandate(saved);
      setBadge(state_, 'ok', 'mandate_saved_version', { version: saved.version });
      bindText(note, 'mandate_saved_note');
      form.setStatus('mandate_saved', 'ok');
      announce(t('mandate_saved'));
      emit('mandate:saved', { sessionId: turn.sessionId, mandate: saved });
    },
  });
  card.append(head, note, form.el);
  return card;
}

registerEvent('data-mandate', (event, turn) => {
  const box = extraSection(turn, 'mandate');
  if (event.mandate && typeof event.mandate === 'object') {
    box.replaceChildren(draftCard(turn, event.mandate));
    return;
  }
  // History: the draft itself was not stored.
  const note = el('div', 'card card-quiet');
  const open = actionButton('btn btn-quiet', 'mandate_open_panel', { iconName: 'shield' });
  open.addEventListener('click', () => emit('panel:open', 'mandate'));
  note.append(label('p', 'card-note', 'mandate_history_note'), open);
  box.replaceChildren(note);
});

// --- the panel -------------------------------------------------------------------------------------

let container = null;
let view = null; // {sessionId, form, badge}
let known = null; // {sessionId, mandate, saved} from the last session load or save

function renderPanel() {
  if (!container) return;
  const sessionId = state.sessionId;
  if (!sessionId) {
    view = null;
    const empty = el('div', 'panel-empty');
    empty.append(iconSpan('shield', 22), label('p', 'panel-note', 'mandate_no_session'));
    container.replaceChildren(empty);
    return;
  }
  const current = known && known.sessionId === sessionId ? known : { sessionId, mandate: null, saved: false };
  const head = el('div', 'panel-section-head');
  const heading = label('h2', 'panel-h', 'mandate_panel_heading');
  const state_ = current.saved
    ? badge('ok', 'mandate_saved_version', { version: current.mandate.version })
    : badge('muted', 'mandate_defaults');
  head.append(heading, state_);
  const form = createMandateForm({
    mandate: current.mandate || {},
    onChange() {
      setBadge(state_, 'warn', 'mandate_unsaved');
    },
    async onSave(payload) {
      const saved = await saveMandate(sessionId, payload, form);
      if (!saved) return;
      known = { sessionId, mandate: saved, saved: true };
      form.setMandate(saved);
      setBadge(state_, 'ok', 'mandate_saved_version', { version: saved.version });
      form.setStatus('mandate_saved', 'ok');
      announce(t('mandate_saved'));
      emit('mandate:saved', { sessionId, mandate: saved, from: 'panel' });
    },
  });
  view = { sessionId, form, badge: state_ };
  container.replaceChildren(head, label('p', 'panel-note panel-lede', 'mandate_panel_note'), form.el);
}

async function loadPanel() {
  const sessionId = state.sessionId;
  if (!sessionId) {
    renderPanel();
    return;
  }
  try {
    const body = await request('GET', `/api/mandate/${encodeURIComponent(sessionId)}`);
    if (state.sessionId !== sessionId) return;
    known = { sessionId, mandate: body.mandate, saved: Boolean(body.saved) };
  } catch {
    if (state.sessionId !== sessionId) return;
  }
  // Do not replace a form the user is typing in for the same chat.
  if (view && view.sessionId === sessionId && container.contains(document.activeElement)) return;
  renderPanel();
}

registerPanel('mandate', {
  mount(target) {
    container = target;
    renderPanel();
  },
  show() {
    loadPanel();
  },
});

on('session:loaded', (detail) => {
  const sessionId = detail?.session?.id;
  if (sessionId && detail.mandate) {
    known = { sessionId, mandate: detail.mandate, saved: Boolean(detail.mandate_saved) };
  }
  renderPanel();
});
on('session:new', () => renderPanel());
on('change', (patch) => {
  // The first message of a new chat creates its session.
  if ('sessionId' in patch && (!view || view.sessionId !== state.sessionId)) renderPanel();
});
on('mandate:saved', ({ sessionId, mandate, from }) => {
  known = { sessionId, mandate, saved: true };
  if (from !== 'panel' && sessionId === state.sessionId) renderPanel();
});
