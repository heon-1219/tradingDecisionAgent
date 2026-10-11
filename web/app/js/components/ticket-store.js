/* What the UI remembers about tickets: the ids of each chat's tickets (for the Orders panel), and
 * the ticket cards currently on screen (so that a new ticket can retire the older ones).
 *
 * A chat's ids come from the server when it is opened (`GET /api/sessions/{id}` -> tickets, every
 * ticket of the chat oldest first, requotes included); the tickets that arrive afterwards are added
 * as they are seen.
 *
 * Only the newest ticket of a chat can be confirmed: the server supersedes every other proposed
 * ticket of the session when it creates one. `supersedeOthers` mirrors that on screen at once.
 */
import { on } from '../state.js';

const seen = new Map(); // session id -> [ticket id, …] in the order they were first seen
const cards = new Set(); // ticket card controllers: {id, sessionId, el, state(), supersede()}

export function noteTicket(sessionId, ticketId) {
  if (!sessionId || typeof ticketId !== 'string' || !ticketId) return;
  if (!seen.has(sessionId)) seen.set(sessionId, []);
  const ids = seen.get(sessionId);
  if (!ids.includes(ticketId)) ids.push(ticketId);
}

export function ticketIds(sessionId) {
  return [...(seen.get(sessionId) || [])];
}

/* The server's list for `sessionId` (`[{id, …}]`, oldest first) replaces what was seen. An id seen
 * here but not listed (the list holds the newest 50) is older, so it stays in front. */
export function setSessionTickets(sessionId, listed) {
  if (!sessionId || !Array.isArray(listed)) return;
  const ids = [];
  for (const item of listed) {
    const id = item && typeof item.id === 'string' ? item.id : '';
    if (id && !ids.includes(id)) ids.push(id);
  }
  const older = (seen.get(sessionId) || []).filter((id) => !ids.includes(id));
  seen.set(sessionId, [...older, ...ids]);
}

// Registered when this module loads, before any panel's listener: the Orders panel reads the list.
on('session:loaded', (detail) => setSessionTickets(detail?.session?.id, detail?.tickets));

export function registerCard(card) {
  cards.add(card);
}

function liveCards() {
  for (const card of [...cards]) if (!card.el.isConnected) cards.delete(card);
  return [...cards];
}

/* A new ticket arrived for `sessionId`: every other card of that chat that could still be
 * confirmed becomes "superseded". */
export function supersedeOthers(sessionId, keepId) {
  for (const card of liveCards()) {
    if (card.sessionId === sessionId && card.id !== keepId && card.state() === 'proposed') card.supersede();
  }
}

export function cardFor(ticketId) {
  return liveCards().find((card) => card.id === ticketId) || null;
}

/* The newest card on screen of a ticket of `sessionId` seen after `ticketId`, or null. */
export function newerCard(sessionId, ticketId) {
  const ids = ticketIds(sessionId);
  for (const id of ids.slice(ids.indexOf(ticketId) + 1).reverse()) {
    const card = cardFor(id);
    if (card && card.sessionId === sessionId) return card;
  }
  return null;
}
