/* The trade ticket card: where a person approves paper orders.
 *
 * `data-trade-proposal {ticket}` draws the card in the turn; in history the event carries only
 * `ticket_id`, and the ticket is read from `GET /api/tickets/{id}?session_id=`.
 *
 * States (the ticket's stored status, plus the clock): `proposed` shows Confirm, Modify and Reject
 * with a countdown; `expired`, `superseded` and `rejected_by_user` are greyed and can no longer be
 * confirmed; `submitting` and `submitted` show the order timeline; `failed` shows what happened.
 * Only the newest ticket of a chat is confirmable: a new one retires the others on screen.
 *
 * Confirm sends the ticket's `hash` back, is disabled while a request is in flight, has no keyboard
 * shortcut and is never focused by the app. How each answer is handled is decided by
 * `interpretResult` (ticket-model.js); the comments in `settle` below list the cases.
 */
import { ApiError, request } from '../api.js';
import { cmp, formatMoney } from '../decimal.js';
import { bindText, getLang, t } from '../i18n.js';
import { announce, notify } from '../notify.js';
import { registerEvent } from '../registry.js';
import { canSend, emit, on, state } from '../state.js';
import { actionButton, ageParts, badge, clockText, el, extraSection, iconSpan, label, str, timeOf } from './dom.js';
import { ensurePortfolio, snapshot } from './portfolio-store.js';
import { nowMs, onTick } from './ticker.js';
import { confirmLabel, executableLegs, interpretResult, modifyMessage, requoteChanges, ticketState } from './ticket-model.js';
import { createModifyForm } from './ticket-modify.js';
import { noticeNodes, submittedNotice } from './ticket-notice.js';
import { noteTicket, registerCard, supersedeOthers } from './ticket-store.js';
import { createTimeline } from './ticket-timeline.js';
import { ticketBody } from './ticket-view.js';

// state -> [badge key, tone]
const STATE = {
  proposed: ['ticket_state_proposed', 'accent'],
  expired: ['ticket_state_expired', 'muted'],
  superseded: ['ticket_state_superseded', 'muted'],
  rejected_by_user: ['ticket_state_rejected', 'muted'],
  submitting: ['ticket_state_submitting', 'accent'],
  submitted: ['ticket_state_submitted', 'ok'],
  failed: ['ticket_state_failed', 'bad'],
  checking: ['ticket_state_checking', 'warn'],
  gone: ['ticket_state_gone', 'muted'],
};

function panelLink(key, panel, iconName) {
  const button = actionButton('btn btn-quiet btn-small', key, { iconName, iconSize: 14 });
  button.addEventListener('click', () => emit('panel:open', panel));
  return button;
}

export function createTicketCard(initial, { sessionId, orders = [] }) {
  const root = el('article', 'card ticket');
  root.tabIndex = -1; // focusable by script only: where focus lands when a pressed button is replaced
  const notice = el('div', 'notice ticket-notice');
  notice.hidden = true;
  const timelineHost = el('div', 'ticket-timeline');
  const modifyHost = el('div', 'ticket-modify-host');
  let ticket = initial;
  let override = null; // a state the UI knows before (or apart from) the stored status
  let busy = false;
  let timeline = null;
  let body = null;
  let clockNodes = null;

  const currentState = () => override || ticketState(ticket, nowMs());

  /* Show a message in the card (see ticket-notice.js for the description it takes); `null` clears it. */
  function setNotice(spec) {
    notice.hidden = !spec;
    if (!spec) {
      notice.replaceChildren();
      return;
    }
    notice.setAttribute('data-tone', spec.tone || 'muted');
    notice.replaceChildren(...noticeNodes(spec));
  }

  function showTimeline(known) {
    if (timeline) return;
    timeline = createTimeline({
      ticket, sessionId, orders: known,
      onTicket(next) {
        if (next && next.id === ticket.id && next.status !== ticket.status) {
          ticket = next;
          render();
        }
      },
    });
    timelineHost.replaceChildren(timeline.el);
  }

  /* "After the trade" figures only make sense while the trade is still a proposal. */
  function showEstimates(data) {
    if (body) body.setPortfolio(currentState() === 'proposed' ? data : null);
  }

  function tick(now) {
    if (!clockNodes) return;
    const quoted = timeOf(ticket.quote_as_of);
    if (quoted !== null) {
      const [key, vars] = ageParts(now - quoted);
      bindText(clockNodes.age, 'ticket_quoted', { age: t(key, vars) });
    }
    const expires = timeOf(ticket.expires_at);
    if (clockNodes.left && expires !== null) {
      const seconds = Math.max(0, Math.ceil((expires - now) / 1000));
      bindText(clockNodes.left, 'ticket_expires_in', { time: clockText(seconds) });
      clockNodes.left.setAttribute('data-urgent', String(seconds <= 30));
      if (seconds === 0 && !override && ticket.status === 'proposed') render(); // it has just expired
    }
  }

  function actionsFor(current) {
    const row = el('footer', 'ticket-actions');
    if (current === 'proposed') {
      const name = confirmLabel(ticket);
      const confirm = actionButton('btn btn-confirm', name.key, { vars: name.vars });
      confirm.disabled = busy || !name.safe; // a ticket the card cannot describe is not confirmable here
      confirm.addEventListener('click', () => act('confirm'));
      if (!name.safe && executableLegs(ticket).length) row.appendChild(label('p', 'ticket-caveat', 'ticket_unreadable'));
      const modify = actionButton('btn btn-modify', 'ticket_modify');
      modify.disabled = busy;
      modify.addEventListener('click', openModify);
      const reject = actionButton('btn btn-quiet', 'ticket_reject');
      reject.disabled = busy;
      reject.addEventListener('click', () => act('reject'));
      row.append(confirm, modify, reject);
    } else if (current === 'expired') {
      const again = actionButton('btn', 'ticket_refresh', { iconName: 'refresh' });
      again.disabled = busy;
      again.addEventListener('click', () => act('refresh'));
      row.appendChild(again);
    } else {
      return null;
    }
    if (busy) row.appendChild(label('span', 'ticket-working shimmer', 'ticket_working'));
    return row;
  }

  function quoteLine(current) {
    const line = el('div', 'ticket-quote');
    const age = el('span', 'ticket-age');
    line.append(iconSpan('clock', 14), age);
    clockNodes = { age, left: null };
    if (current === 'proposed') {
      clockNodes.left = el('span', 'ticket-left');
      const again = actionButton('btn-link', 'ticket_refresh');
      again.disabled = busy;
      again.addEventListener('click', () => act('refresh'));
      line.append(clockNodes.left, again);
    }
    return line;
  }

  function render() {
    const current = currentState();
    const hadFocus = root.contains(document.activeElement);
    root.setAttribute('data-ticket-id', str(ticket.id));
    root.setAttribute('data-state', current);
    root.setAttribute('aria-busy', String(busy));
    const [stateKey, tone] = STATE[current] || STATE.gone;
    const head = el('header', 'card-head');
    const title = el('div', 'card-title');
    title.append(iconSpan('ticket', 18), label('h3', 'card-h', 'ticket_title'), el('span', 'ticket-id', str(ticket.id)));
    head.append(title, badge(tone, stateKey));

    body = ticketBody(ticket, sessionId);
    showEstimates(snapshot());
    if (current !== 'proposed') modifyHost.replaceChildren();
    if (current === 'submitted' || current === 'submitting') showTimeline(orders);
    const actions = actionsFor(current);
    root.replaceChildren(head, quoteLine(current), ...body.nodes, notice, timelineHost, modifyHost,
      ...(actions ? [actions] : []));
    // The pressed button was rebuilt: keep focus in the card, but never on Confirm.
    if (hadFocus && !root.contains(document.activeElement)) root.focus({ preventScroll: true });
    tick(nowMs());
  }

  function openModify() {
    if (busy || currentState() !== 'proposed') return;
    const form = createModifyForm({
      legs: executableLegs(ticket),
      onCancel() {
        modifyHost.replaceChildren();
        root.querySelector('.ticket-actions .btn-modify')?.focus(); // back to where the fields were opened
      },
      onSend(message) {
        if (!canSend(state)) return false;
        modifyHost.replaceChildren();
        emit('chat:send', message);
        return true;
      },
    });
    modifyHost.replaceChildren(form.el);
    form.focus();
  }

  /* Re-read the ticket after an answer that left its state unknown. */
  async function sync() {
    try {
      const fresh = await request('GET',
        `/api/tickets/${encodeURIComponent(ticket.id)}?session_id=${encodeURIComponent(sessionId)}`);
      if (fresh && fresh.ticket) {
        ticket = fresh.ticket;
        orders = fresh.orders || [];
        override = null;
      }
    } catch {
      /* still unknown: the card keeps its buttons hidden and offers "Check again" */
    }
    render();
  }

  async function act(action) {
    if (busy) return;
    if (action === 'confirm' && !confirmLabel(ticket).safe) return;
    busy = true;
    render();
    const path = `/api/tickets/${encodeURIComponent(ticket.id)}/${action}`;
    const payload = action === 'confirm' ? { session_id: sessionId, hash: ticket.hash } : { session_id: sessionId };
    let result;
    try {
      result = { ok: true, status: 200, body: await request('POST', path, payload) };
    } catch (err) {
      const known = err instanceof ApiError;
      result = { ok: false, status: known ? err.status : 0, error: known ? err.error : '',
        body: (known && err.body) || { message: err?.message || '' } };
    }
    busy = false;
    settle(action, interpretResult(action, result));
  }

  function addSuccessor(next) {
    const card = createTicketCard(next, { sessionId });
    root.after(card.el);
    noteTicket(sessionId, next.id);
    registerCard(card);
    supersedeOthers(sessionId, next.id);
    card.el.scrollIntoView({ block: 'nearest' });
  }

  function settle(action, outcome) {
    const said = outcome.message;
    if (outcome.ticket && outcome.ticket.id === ticket.id) ticket = outcome.ticket;
    switch (outcome.kind) {
      case 'submitted': // 200: show the order timeline
        override = null;
        if (ticket.status !== 'submitted') ticket = { ...ticket, status: 'submitted' };
        orders = outcome.orders;
        setNotice(submittedNotice(outcome.orders, executableLegs(ticket).length, said));
        announce(t('ticket_announce_submitted'));
        emit('portfolio:refresh');
        break;
      case 'requote': { // 409 on confirm, 200 on refresh: old -> new price, and a fresh card to confirm
        const lines = requoteChanges(ticket, outcome.newTicket)
          .filter((change) => change.from && change.to && cmp(change.from, change.to) !== 0)
          .map((change) => `${change.symbol}: ${formatMoney(change.from)} → ${formatMoney(change.to)}`);
        override = 'superseded';
        setNotice({ tone: 'warn', key: action === 'confirm' ? 'ticket_requote_confirm' : 'ticket_requote_refresh', lines });
        render();
        addSuccessor(outcome.newTicket);
        announce(t('ticket_announce_requote'));
        return;
      }
      case 'dismissed': // reject, 200
        override = null;
        if (ticket.status === 'proposed') ticket = { ...ticket, status: 'rejected_by_user' };
        setNotice(null);
        announce(t('ticket_announce_rejected'));
        break;
      case 'rejected': // 409: the returned ticket says where things stand; the server says why
        setNotice({ tone: 'bad', key: ticket.status === 'proposed' ? 'ticket_changed' : 'ticket_conflict', detail: said });
        if (action === 'confirm') announce(t('ticket_announce_not_placed'));
        break;
      case 'expired': // 410: grey the card; a refresh that came too late offers a new proposal
        override = 'expired';
        if (action === 'refresh') {
          const ask = actionButton('btn btn-small', 'ticket_ask_again', { iconName: 'refresh', iconSize: 14 });
          ask.addEventListener('click', () => {
            if (canSend(state)) emit('chat:send', modifyMessage(executableLegs(ticket), getLang()));
            else notify(t('ticket_modify_busy'));
          });
          setNotice({ tone: 'warn', key: 'ticket_expired_late', actions: [ask] });
        } else {
          setNotice({ tone: 'warn', key: 'ticket_expired_note' });
          if (action === 'confirm') announce(t('ticket_announce_not_placed'));
        }
        break;
      case 'disabled': // 423: trading is off (kill switch)
        setNotice({ tone: 'warn', key: 'ticket_disabled', actions: [panelLink('open_settings', 'settings', 'sliders')] });
        announce(t('ticket_announce_not_placed'));
        break;
      case 'no_broker': // 503 broker_not_connected
        setNotice({ tone: 'warn', key: 'ticket_no_broker', actions: [panelLink('open_settings', 'settings', 'sliders')] });
        if (action === 'confirm') announce(t('ticket_announce_not_placed'));
        break;
      case 'uncertain': { // 502 / 500 / no answer on confirm: orders may exist; never offer a blind retry
        override = 'checking';
        const check = actionButton('btn btn-small', 'ticket_check_again', { iconName: 'refresh', iconSize: 14 });
        check.addEventListener('click', () => sync());
        setNotice({
          tone: 'bad', key: 'ticket_uncertain', also: 'ticket_check_orders', detail: said,
          actions: [panelLink('open_orders', 'orders', 'receipt'), check],
        });
        announce(t('ticket_announce_uncertain'));
        emit('portfolio:refresh');
        render();
        sync();
        return;
      }
      case 'not_found': // 404
        override = 'gone';
        setNotice({ tone: 'bad', key: 'ticket_not_found' });
        break;
      default: // reject or refresh failed (5xx, network): nothing changed
        setNotice({ tone: 'bad', key: 'ticket_action_failed', detail: said });
    }
    render();
  }

  const card = {
    el: root,
    get id() { return ticket.id; },
    sessionId,
    state: currentState,
    supersede() {
      override = 'superseded';
      render();
    },
  };

  render();
  onTick(root, tick);
  ensurePortfolio();
  const stopPortfolio = on('portfolio:data', (data) => {
    if (!root.isConnected) stopPortfolio();
    else showEstimates(data);
  });
  const stopSources = on('sources:loaded', () => {
    if (!root.isConnected) stopSources();
    else if (body) body.refreshEvidence();
  });
  return card;
}

function mount(ticket, sessionId, orders) {
  const card = createTicketCard(ticket, { sessionId, orders });
  registerCard(card);
  return card;
}

const ticketPath = (id, sessionId) => `/api/tickets/${encodeURIComponent(id)}?session_id=${encodeURIComponent(sessionId)}`;

registerEvent('data-trade-proposal', (event, turn) => {
  const sessionId = turn.sessionId;
  const box = extraSection(turn, 'ticket');
  if (event.ticket && typeof event.ticket === 'object' && typeof event.ticket.id === 'string') {
    noteTicket(sessionId, event.ticket.id);
    box.replaceChildren(mount(event.ticket, sessionId, []).el);
    supersedeOthers(sessionId, event.ticket.id);
    return;
  }
  const id = str(event.ticket_id);
  if (!id || !sessionId) return;
  // History: only the id was stored.
  noteTicket(sessionId, id);
  const waiting = label('p', 'card card-quiet card-note', 'ticket_loading');
  waiting.setAttribute('data-ticket-id', id);
  box.replaceChildren(waiting);
  request('GET', ticketPath(id, sessionId))
    .then((found) => {
      if (!waiting.isConnected || !found || !found.ticket) return;
      waiting.replaceWith(mount(found.ticket, sessionId, found.orders || []).el);
    })
    .catch(() => {
      if (waiting.isConnected) bindText(waiting, 'ticket_unavailable');
    });
});

// A reloaded chat: the active ticket may be a requote, which no turn's events mention. Show it under
// the last answer so that it can still be confirmed.
on('session:loaded', (detail) => {
  const active = detail?.active_ticket;
  const sessionId = detail?.session?.id;
  if (!active || typeof active.id !== 'string' || !sessionId) return;
  noteTicket(sessionId, active.id);
  const thread = document.getElementById('thread');
  const shown = [...thread.querySelectorAll('[data-ticket-id]')].some((n) => n.getAttribute('data-ticket-id') === active.id);
  const last = [...thread.querySelectorAll('.turn-assistant .turn-extras')].pop();
  if (shown || !last) return;
  extraSection({ slots: { extras: last } }, 'ticket').appendChild(mount(active, sessionId, []).el);
});
