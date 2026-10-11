/* The ticket card's message area, built from a plain description:
 *   {tone, key, vars, also, orders, lines, detail, actions}
 * `key`/`also` are i18n keys, so the message follows the interface language. `orders` are order
 * snapshots from a confirm answer, shown one per line (side, size, symbol, terms, status in words).
 * `lines` are short lines built here from structured data (old → new prices). `detail` is the
 * server's own English message: it is shown only in a collapsed "Details from the server" row, and
 * only when it adds something the localized text cannot say (errors, uncertain outcomes). Every
 * piece of server or broker text is set with textContent.
 */
import { el, iconSpan, label, str } from './dom.js';
import { orderStatusBadge, sideBadge, sizeNode, termsNode } from './order-view.js';

const ICON = { ok: 'check', muted: 'info' };

function orderLines(orders) {
  const list = el('ul', 'notice-orders');
  for (const order of orders) {
    if (!order || typeof order !== 'object') continue;
    const item = el('li', 'notice-order');
    item.append(sideBadge(order.side), el('span', 'leg-symbol', str(order.symbol)), sizeNode(order),
      termsNode(order), orderStatusBadge(order.status));
    list.appendChild(item);
  }
  return list;
}

function serverDetail(text) {
  const details = el('details', 'notice-detail');
  details.append(label('summary', '', 'ticket_server_detail'), el('p', 'notice-line', text));
  return details;
}

/* The notice for a confirm that went through (200): summarised from the structured result, in the
 * interface language. The server's text is kept only when a leg was not placed (it says why). */
export function submittedNotice(orders, legCount, serverText) {
  const placed = (Array.isArray(orders) ? orders : []).filter((order) => order && typeof order === 'object');
  const total = legCount || placed.length;
  const partial = placed.length < total;
  return {
    tone: partial ? 'warn' : 'ok', key: 'ticket_result_placed', vars: { placed: placed.length, total },
    orders: placed, also: partial ? 'ticket_result_partial' : null, detail: partial ? serverText : '',
  };
}

/* The notice's children: an icon for the tone and the message body. */
export function noticeNodes(spec) {
  const content = el('div', 'notice-body');
  content.appendChild(spec.key ? label('p', 'notice-text', spec.key, spec.vars) : el('p', 'notice-text', str(spec.text)));
  if (spec.also) content.appendChild(label('p', 'notice-text', spec.also));
  if (Array.isArray(spec.orders) && spec.orders.length) content.appendChild(orderLines(spec.orders));
  for (const line of spec.lines || []) content.appendChild(el('p', 'notice-line', line));
  if (str(spec.detail).trim()) content.appendChild(serverDetail(str(spec.detail).trim()));
  if (spec.actions && spec.actions.length) {
    const row = el('div', 'notice-actions');
    row.append(...spec.actions);
    content.appendChild(row);
  }
  return [iconSpan(ICON[spec.tone] || 'alert', 18), content];
}
