/* Where ticket cards go in the thread (the card itself is ticket.js).
 *
 * Live: `data-trade-proposal {ticket}` draws the card in its turn. History: the event carries only
 * `ticket_id`; a "Loading the ticket…" line holds the place while `GET /api/tickets/{id}?session_id=`
 * reads it.
 *
 * A reloaded chat: `GET /api/sessions/{id}` lists every ticket of the chat (`tickets`, oldest first),
 * including the requotes that Refresh quote or a confirm made, which no turn's events mention. Each
 * one not on screen yet goes under the last answer written before it (by `created_at`; a requote
 * comes after the answer whose card was pressed), or under the last answer, in the listed order. So a
 * requoted and then confirmed ticket still shows, with its orders, after a reload.
 */
import { request } from '../api.js';
import { bindText } from '../i18n.js';
import { registerEvent } from '../registry.js';
import { on } from '../state.js';
import { extraSection, label, str, timeOf } from './dom.js';
import { createTicketCard } from './ticket.js';
import { noteTicket, registerCard, supersedeOthers } from './ticket-store.js';

function mount(ticket, sessionId, orders) {
  const card = createTicketCard(ticket, { sessionId, orders });
  registerCard(card);
  return card;
}

const ticketPath = (id, sessionId) => `/api/tickets/${encodeURIComponent(id)}?session_id=${encodeURIComponent(sessionId)}`;

/* Append a placeholder for ticket `id` to `box`; it becomes the card once the ticket is read. */
function placeholder(box, id, sessionId) {
  const waiting = label('p', 'card card-quiet card-note', 'ticket_loading');
  waiting.setAttribute('data-ticket-id', id);
  box.appendChild(waiting);
  request('GET', ticketPath(id, sessionId))
    .then((found) => {
      if (!waiting.isConnected || !found || !found.ticket) return;
      waiting.replaceWith(mount(found.ticket, sessionId, found.orders || []).el);
    })
    .catch(() => {
      if (waiting.isConnected) bindText(waiting, 'ticket_unavailable');
    });
}

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
  noteTicket(sessionId, id); // history: only the id was stored
  box.replaceChildren();
  placeholder(box, id, sessionId);
});

/* The answers of a reloaded chat, in order: each one's extras slot and when it was written. The
 * history's assistant rows and the drawn answers pair up by position (thread.js draws one per row);
 * if they do not, no answer has a time and every ticket goes under the last one. */
function answersOf(thread, messages) {
  const rows = (Array.isArray(messages) ? messages : [])
    .filter((row) => row && typeof row === 'object' && row.role === 'assistant');
  const turns = [...thread.querySelectorAll('.turn-assistant')];
  return turns
    .map((turn, i) => ({ extras: turn.querySelector('.turn-extras'), at: rows.length === turns.length ? timeOf(rows[i].created_at) : null }))
    .filter((answer) => answer.extras);
}

function anchorFor(item, answers) {
  const at = timeOf(item.created_at);
  let found = null;
  if (at !== null) for (const answer of answers) if (answer.at !== null && answer.at <= at) found = answer;
  return (found || answers[answers.length - 1] || null)?.extras || null;
}

on('session:loaded', (detail) => {
  const sessionId = detail?.session?.id;
  const thread = document.getElementById('thread');
  if (!sessionId || !thread) return;
  const active = detail.active_ticket && typeof detail.active_ticket.id === 'string' ? detail.active_ticket : null;
  const listed = Array.isArray(detail.tickets)
    ? detail.tickets.filter((item) => item && typeof item.id === 'string' && item.id)
    : (active ? [active] : []); // a server without the list: the active ticket only
  const answers = answersOf(thread, detail.messages);
  const shown = new Set([...thread.querySelectorAll('[data-ticket-id]')].map((node) => node.getAttribute('data-ticket-id')));
  for (const item of listed) {
    const extras = shown.has(item.id) ? null : anchorFor(item, answers);
    if (!extras) continue;
    shown.add(item.id);
    noteTicket(sessionId, item.id);
    const box = extraSection({ slots: { extras } }, 'ticket');
    if (active && item.id === active.id) box.appendChild(mount(active, sessionId, []).el);
    else placeholder(box, item.id, sessionId);
  }
});
