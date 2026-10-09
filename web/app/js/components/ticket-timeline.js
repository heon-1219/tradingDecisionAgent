/* The order timeline of a submitted ticket.
 *
 * One row per order: direction, symbol, size, a three-step track (sent, working, filled) and the
 * status in words, with what has filled so far. A leg whose outcome the server could not establish
 * has no order; it is shown as "uncertain: check open orders" and is never offered for
 * resubmission.
 *
 * While any order is not final, the ticket is re-read every 3 s for up to 60 s
 * (`GET /api/tickets/{id}?session_id=`; the server re-reads those orders from the broker). Polling
 * stops when every order is final, when the time is up, when the card leaves the document, or when
 * another chat is opened. `createPoller` takes its timer and clock as arguments so the rule is
 * tested without waiting.
 */
import { request } from '../api.js';
import { bindText } from '../i18n.js';
import { emit, state } from '../state.js';
import { actionButton, badge, el, label, str } from './dom.js';
import { fillNode, isFinalOrder, orderStatusBadge, orderStatusInfo, sideBadge, sizeNode } from './order-view.js';

export const POLL_EVERY_MS = 3000;
export const POLL_FOR_MS = 60000;

/* Orders in the two shapes the API uses: a confirm answer lists order snapshots, a ticket read
 * lists {client_order_id, broker_order_id, order}. Both become the second shape. */
export function normalizeOrders(orders) {
  return (Array.isArray(orders) ? orders : []).filter((row) => row && typeof row === 'object').map((row) => (
    'order' in row
      ? row
      : { client_order_id: str(row.client_order_id), broker_order_id: str(row.id) || null, order: row }));
}

/* True while something may still change: an order that is not final. An uncertain leg (no order)
 * will not resolve by polling. */
export function anyPending(rows) {
  return rows.some((row) => row.order && !isFinalOrder(row.order));
}

/* Poll `read()` every `every` ms for at most `limit` ms, while `alive()` and `pending(result)`.
 * `onResult(result)` gets every successful read; `onStop(reason)` is called once with
 * 'settled' | 'timeout' | 'gone'. A failed read is skipped, not fatal. Returns {stop()}. */
export function createPoller({ read, pending, alive, onResult, onStop, every = POLL_EVERY_MS,
  limit = POLL_FOR_MS, now = () => Date.now(), delay = (fn, ms) => setTimeout(fn, ms),
  cancel = (id) => clearTimeout(id) }) {
  const started = now();
  let timer = null;
  let stopped = false;

  function stop(reason) {
    if (stopped) return;
    stopped = true;
    if (timer !== null) cancel(timer);
    if (onStop) onStop(reason);
  }

  async function step() {
    timer = null;
    if (stopped) return;
    if (!alive()) return stop('gone');
    let result = null;
    try {
      result = await read();
    } catch {
      result = null;
    }
    if (stopped) return;
    if (!alive()) return stop('gone');
    if (result !== null) {
      onResult(result);
      if (!pending(result)) return stop('settled');
    }
    if (now() - started + every > limit) return stop('timeout');
    timer = delay(step, every);
  }

  timer = delay(step, every);
  return { stop: () => stop('gone') };
}

function track(step, tone) {
  const node = el('span', 'order-track');
  node.setAttribute('aria-hidden', 'true');
  node.setAttribute('data-tone', tone);
  for (let i = 1; i <= 3; i += 1) {
    const dot = el('span', 'order-step');
    if (i <= step) dot.classList.add('is-on');
    node.appendChild(dot);
  }
  return node;
}

function row(tracked) {
  const item = el('li', 'timeline-row');
  const order = tracked.order;
  if (!order) {
    item.classList.add('is-uncertain');
    item.append(badge('warn', 'order_uncertain'), label('span', 'order-note', 'order_uncertain_note'));
    return item;
  }
  const info = orderStatusInfo(order.status);
  const main = el('div', 'timeline-main');
  main.append(sideBadge(order.side), el('span', 'leg-symbol', str(order.symbol)), sizeNode(order));
  const status = el('div', 'timeline-status');
  status.append(track(info.step, info.tone), orderStatusBadge(order.status));
  const fill = fillNode(order);
  if (fill) status.appendChild(fill);
  item.append(main, status);
  return item;
}

/* Build the timeline for `ticket`. `orders` is what is known now (either API shape). It keeps
 * itself up to date and calls `onTicket(ticket)` when a read returns the ticket. */
export function createTimeline({ ticket, sessionId, orders, onTicket }) {
  const root = el('section', 'timeline');
  const list = el('ol', 'timeline-list');
  const foot = el('p', 'timeline-foot');
  const again = actionButton('btn btn-quiet btn-small', 'timeline_refresh', { iconName: 'refresh', iconSize: 14 });
  again.hidden = true;
  root.append(label('h4', 'ticket-h', 'timeline_heading'), list, foot, again);
  let rows = normalizeOrders(orders);
  let poller = null;

  const path = `/api/tickets/${encodeURIComponent(ticket.id)}?session_id=${encodeURIComponent(sessionId)}`;
  const alive = () => root.isConnected && state.sessionId === sessionId;

  function draw() {
    if (rows.length === 0) list.replaceChildren(label('li', 'timeline-row is-empty', 'timeline_empty'));
    else list.replaceChildren(...rows.map(row));
  }

  function apply(body) {
    if (Array.isArray(body?.orders)) rows = normalizeOrders(body.orders);
    if (body?.ticket && onTicket) onTicket(body.ticket);
    draw();
  }

  function start() {
    if (poller) poller.stop();
    again.hidden = true;
    bindText(foot, 'timeline_updating');
    poller = createPoller({
      read: () => request('GET', path),
      pending: (body) => anyPending(normalizeOrders(body?.orders)) || body?.ticket?.status === 'submitting',
      alive,
      onResult: apply,
      onStop(reason) {
        poller = null;
        if (reason === 'gone') return;
        const settled = reason === 'settled';
        bindText(foot, settled ? 'timeline_settled' : 'timeline_paused');
        again.hidden = settled;
        emit('portfolio:refresh');
      },
    });
  }

  again.addEventListener('click', async () => {
    again.disabled = true;
    try {
      apply(await request('GET', path));
    } catch {
      /* the next click tries again */
    }
    again.disabled = false;
    if (anyPending(rows)) start();
    else {
      bindText(foot, 'timeline_settled');
      again.hidden = true;
    }
  });

  draw();
  // The first read also picks up legs the confirm answer does not list (uncertain ones).
  request('GET', path).then((body) => {
    if (!alive()) return;
    apply(body);
    if (anyPending(rows) || body?.ticket?.status === 'submitting') start();
    else bindText(foot, 'timeline_settled');
  }).catch(() => {
    if (!alive()) return;
    if (anyPending(rows)) start();
    else {
      bindText(foot, 'timeline_paused');
      again.hidden = false;
    }
  });
  bindText(foot, 'timeline_updating');
  return { el: root };
}
