/* How an order is shown: the side badge, the status chip and the one-line description. Shared by
 * the ticket card (legs and the order timeline) and the Orders panel.
 *
 * Direction and status are always written in words; colour and the ▲/▼ mark only repeat them.
 * Symbols, statuses and messages come from the broker and are rendered as text.
 */
import { formatMoney, formatQty } from '../decimal.js';
import { badge, el, label, str } from './dom.js';

/* Alpaca statuses after which an order will not change again today (as on the server). */
export const FINAL_ORDER_STATUSES = new Set(['filled', 'canceled', 'expired', 'rejected', 'replaced', 'done_for_day']);

// status -> [i18n key, tone, step on the 3-step track (1 sent, 2 working, 3 filled)]
const ORDER_STATUS = {
  pending_new: ['order_pending_new', 'accent', 1],
  accepted: ['order_accepted', 'accent', 2],
  accepted_for_bidding: ['order_accepted', 'accent', 2],
  new: ['order_new', 'accent', 2],
  partially_filled: ['order_partially_filled', 'accent', 2],
  filled: ['order_filled', 'ok', 3],
  done_for_day: ['order_done_for_day', 'muted', 2],
  canceled: ['order_canceled', 'muted', 2],
  pending_cancel: ['order_pending_cancel', 'muted', 2],
  expired: ['order_expired', 'muted', 2],
  replaced: ['order_replaced', 'muted', 2],
  pending_replace: ['order_pending_replace', 'accent', 2],
  rejected: ['order_rejected', 'bad', 1],
  stopped: ['order_stopped', 'accent', 2],
  suspended: ['order_suspended', 'warn', 2],
  held: ['order_held', 'accent', 1],
};

export function orderStatusInfo(status) {
  const name = str(status).toLowerCase();
  const known = Object.prototype.hasOwnProperty.call(ORDER_STATUS, name) ? ORDER_STATUS[name] : null;
  return {
    name,
    key: known ? known[0] : null,
    tone: known ? known[1] : 'muted',
    step: known ? known[2] : 1,
    final: FINAL_ORDER_STATUSES.has(name),
  };
}

export function isFinalOrder(order) {
  return Boolean(order) && FINAL_ORDER_STATUSES.has(str(order.status).toLowerCase());
}

/* "BUY ▲" / "SELL ▼": the word first, the mark for a quick scan. */
export function sideBadge(side) {
  const sell = str(side).toLowerCase() === 'sell';
  const node = el('span', 'side-badge');
  node.setAttribute('data-side', sell ? 'sell' : 'buy');
  const mark = el('span', 'side-mark', sell ? '▼' : '▲');
  mark.setAttribute('aria-hidden', 'true');
  node.append(label('span', '', sell ? 'side_sell' : 'side_buy'), mark);
  return node;
}

export function orderStatusBadge(status) {
  const info = orderStatusInfo(status);
  if (info.key) return badge(info.tone, info.key);
  const node = el('span', 'badge', info.name || '–'); // a status this app does not know: shown as sent
  node.setAttribute('data-tone', 'muted');
  return node;
}

/* The size of a leg or an order: "3 shares" or "$500.00 notional". */
export function sizeNode(item) {
  if (item && item.qty !== null && item.qty !== undefined) {
    return label('span', 'leg-size', 'size_shares', { qty: formatQty(str(item.qty)) });
  }
  return label('span', 'leg-size', 'size_notional', { amount: formatMoney(str(item?.notional)) });
}

/* The terms of a leg or an order: "Limit $612.00 · Day" or "Market · Day". */
export function termsNode(item) {
  const node = el('span', 'leg-terms');
  const type = str(item?.type).toLowerCase();
  if (type === 'limit') node.appendChild(label('span', '', 'terms_limit', { price: formatMoney(str(item.limit_price)) }));
  else if (type === 'market') node.appendChild(label('span', '', 'terms_market'));
  else node.appendChild(el('span', '', type || '–'));
  const tif = str(item?.time_in_force).toLowerCase();
  node.appendChild(el('span', 'sep', ' · '));
  if (tif === 'day') node.appendChild(label('span', '', 'tif_day'));
  else if (tif === 'gtc') node.appendChild(label('span', '', 'tif_gtc'));
  else node.appendChild(el('span', '', tif.toUpperCase() || '–'));
  return node;
}

/* "2 of 3 filled at $610.12", or null when nothing has filled. */
export function fillNode(order) {
  const filled = str(order?.filled_qty);
  if (!filled || /^0(\.0+)?$/.test(filled)) return null;
  const vars = { filled: formatQty(filled), price: formatMoney(str(order.filled_avg_price)) };
  if (order.qty !== null && order.qty !== undefined) {
    return label('span', 'order-fill', 'order_fill_of', { ...vars, qty: formatQty(str(order.qty)) });
  }
  return label('span', 'order-fill', 'order_fill', vars);
}
