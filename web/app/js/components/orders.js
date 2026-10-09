/* The Orders panel: the account's open orders, and the orders placed from this chat's tickets.
 *
 * Open orders come with the portfolio (`GET /api/portfolio` -> open_orders). The chat's own orders
 * are read per ticket (`GET /api/tickets/{id}?session_id=`) for the last 20 tickets of the open chat
 * (ticket-store.js: the server's list when the chat was opened, requotes included, plus the ones
 * since; the panel says so when there are more), and only the newest load is shown. The server
 * re-reads non-final orders from the broker on each call. A leg whose outcome the server
 * could not establish has no order and is shown as "uncertain". It reloads when the tab is shown,
 * after a confirm, and from its refresh button.
 */
import { request } from '../api.js';
import { registerPanel } from '../registry.js';
import { emit, on, state } from '../state.js';
import { actionButton, badge, dateTimeText, el, iconButton, iconSpan, label, str } from './dom.js';
import { fillNode, orderStatusBadge, sideBadge, sizeNode, termsNode } from './order-view.js';
import { refreshPortfolio, snapshot } from './portfolio-store.js';
import { ticketIds } from './ticket-store.js';

const MAX_TICKETS = 20; // the newest tickets of the chat; the panel says when there are more
let container = null;
let shown = false; // the tab has been opened at least once
let chat = { sessionId: null, groups: [], loading: false, total: 0 };
let generation = 0; // only the newest load may write `chat`

/* One order as a row. `tracked` is {client_order_id, broker_order_id, order} from a ticket, or
 * {order} for an open order. */
export function orderRow(tracked) {
  const order = tracked.order;
  const row = el('li', 'order-row');
  if (!order) {
    row.classList.add('is-uncertain');
    const body = el('div', 'order-main');
    body.append(badge('warn', 'order_uncertain'), label('span', 'order-note', 'order_uncertain_note'));
    row.appendChild(body);
    return row;
  }
  const main = el('div', 'order-main');
  main.append(sideBadge(order.side), el('span', 'leg-symbol', str(order.symbol)), sizeNode(order));
  const detail = el('div', 'order-detail');
  detail.append(termsNode(order), orderStatusBadge(order.status));
  const fill = fillNode(order);
  if (fill) detail.appendChild(fill);
  const when = dateTimeText(order.updated_at || order.submitted_at);
  if (when) detail.appendChild(el('time', 'order-time', when));
  row.append(main, detail);
  return row;
}

function list(rows) {
  const node = el('ul', 'order-list');
  node.append(...rows.map(orderRow));
  return node;
}

function section(headingKey, vars, ...children) {
  const box = el('section', 'panel-section');
  box.append(label('h2', 'panel-h', headingKey, vars), ...children);
  return box;
}

async function loadChatOrders() {
  const mine = ++generation;
  const sessionId = state.sessionId;
  const all = ticketIds(sessionId);
  const ids = all.slice(-MAX_TICKETS).reverse();
  chat = { sessionId, groups: chat.sessionId === sessionId ? chat.groups : [], loading: ids.length > 0, total: all.length };
  render();
  const groups = [];
  for (const id of ids) {
    try {
      const body = await request('GET',
        `/api/tickets/${encodeURIComponent(id)}?session_id=${encodeURIComponent(sessionId)}`);
      if (Array.isArray(body?.orders) && body.orders.length > 0) groups.push({ id, orders: body.orders });
    } catch {
      /* a ticket that cannot be read is simply not listed */
    }
    // A newer load (a refresh, another chat) started meanwhile: its answer is the one to show.
    if (mine !== generation || state.sessionId !== sessionId) return;
  }
  chat = { sessionId, groups, loading: false, total: all.length };
  render();
}

function render() {
  if (!container) return;
  const data = snapshot();
  const keepFocus = container.contains(document.activeElement);
  const head = el('div', 'panel-section-head');
  const refresh = iconButton('refresh', 'orders_refresh');
  refresh.addEventListener('click', () => reload());
  head.append(label('h2', 'panel-h', 'orders_heading'), refresh);
  const nodes = [head];

  if (data.status === 'disconnected') {
    const box = el('div', 'notice');
    box.setAttribute('data-tone', 'muted');
    const body = el('div', 'notice-body');
    const open = actionButton('btn btn-quiet', 'open_settings', { iconName: 'sliders' });
    open.addEventListener('click', () => emit('panel:open', 'settings'));
    body.append(label('p', 'notice-text', 'portfolio_disconnected'), open);
    box.append(iconSpan('wallet', 18), body);
    nodes.push(box);
  } else if (data.status === 'error' && !data.body) {
    const box = el('div', 'notice');
    box.setAttribute('data-tone', 'bad');
    const body = el('div', 'notice-body');
    const retry = actionButton('btn btn-quiet', 'retry', { iconName: 'refresh' });
    retry.addEventListener('click', () => reload());
    body.append(el('p', 'notice-text', data.message), retry);
    box.append(iconSpan('alert', 18), body);
    nodes.push(box);
  } else if (data.body) {
    const open = Array.isArray(data.body.open_orders) ? data.body.open_orders : [];
    nodes.push(section('orders_open', { n: open.length },
      open.length ? list(open.map((order) => ({ order }))) : label('p', 'panel-note', 'orders_open_empty')));
  } else {
    nodes.push(label('p', 'panel-note', 'portfolio_loading'));
  }

  const mine = chat.sessionId === state.sessionId ? chat.groups : [];
  const children = [];
  if (chat.sessionId === state.sessionId && chat.total > MAX_TICKETS) {
    children.push(label('p', 'panel-note', 'orders_more', { shown: MAX_TICKETS, total: chat.total }));
  }
  if (mine.length === 0) {
    children.push(label('p', 'panel-note', chat.loading ? 'portfolio_loading' : 'orders_chat_empty'));
  }
  for (const group of mine) {
    const box = el('div', 'order-group');
    box.append(label('p', 'order-group-h', 'orders_ticket', { id: group.id }), list(group.orders));
    children.push(box);
  }
  nodes.push(section('orders_chat', undefined, ...children));

  container.replaceChildren(...nodes);
  if (keepFocus) container.querySelector('.icon-btn')?.focus();
}

function reload() {
  refreshPortfolio();
  loadChatOrders();
}

registerPanel('orders', {
  mount(target) {
    container = target;
    render();
  },
  show() {
    shown = true;
    reload();
  },
});

on('portfolio:data', render);
on('portfolio:refresh', () => {
  if (shown) loadChatOrders();
});
on('session:loaded', () => {
  if (shown) loadChatOrders();
  else render();
});
on('session:new', () => {
  generation += 1;
  chat = { sessionId: null, groups: [], loading: false, total: 0 };
  render();
});
