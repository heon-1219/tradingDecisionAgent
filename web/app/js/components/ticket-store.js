/* What the UI remembers about tickets: the ids seen in each chat (for the Orders panel), and the
 * ticket cards currently on screen (so that a new ticket can retire the older ones).
 *
 * Only the newest ticket of a chat can be confirmed: the server supersedes every other proposed
 * ticket of the session when it creates one. `supersedeOthers` mirrors that on screen at once.
 */

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
