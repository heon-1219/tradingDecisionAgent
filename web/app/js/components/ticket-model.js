/* The ticket card's logic without the DOM: what the Confirm button says, the estimated totals, what
 * a confirm response means for the card, the "Modify: …" message. Pure functions, covered by
 * `scripts/test_components.mjs`.
 *
 * A ticket's `legs` are exactly its executable legs (the server derives them from the decisions
 * that are not rejected, in order); `decisions` keeps every leg, including the rejected ones, which
 * are shown as information only. Money is the server's decimal strings; the few sums the server
 * does not send are computed with exact decimal arithmetic and labelled "est." by the card.
 */
import {
  add, cmp, div, formatMoney, formatQty, isPositiveDecimal, mul, parseDecimal, sub, toTrimmed,
} from '../decimal.js';

const text = (value) => (typeof value === 'string' ? value : '');
const PLAIN_POSITIVE = /^\d{1,12}(\.\d{1,6})?$/; // what a person may type into the Modify fields

export function executableLegs(ticket) {
  return Array.isArray(ticket?.legs) ? ticket.legs : [];
}

/* 'buy' or 'sell' (any case), else null. */
export function legSide(leg) {
  const side = text(leg?.side).toLowerCase();
  return side === 'buy' || side === 'sell' ? side : null;
}

/* Can the card say exactly what this leg does? A known side, and a quantity or an amount that is a
 * positive decimal. */
export function readableLeg(leg) {
  if (!legSide(leg)) return false;
  const hasQty = leg.qty !== null && leg.qty !== undefined;
  const size = parseDecimal(text(hasQty ? leg.qty : leg.notional));
  return Boolean(size) && size.int > 0n && Boolean(text(leg.symbol));
}

/* The Confirm button's full label, as an i18n key and its variables: "Confirm buy 3 VOO", "Confirm
 * sell $500.00 of BND", or "Confirm 2 orders". `safe` is false when a leg cannot be read (an
 * unknown side, a size that does not parse): the label is then the generic "Confirm ticket" and
 * the card keeps Confirm disabled, rather than naming an order it cannot vouch for. */
export function confirmLabel(ticket) {
  const legs = executableLegs(ticket);
  if (legs.length === 0) return { key: 'ticket_confirm_none', vars: {}, safe: false };
  if (!legs.every(readableLeg)) return { key: 'ticket_confirm_generic', vars: {}, safe: false };
  if (legs.length > 1) return { key: 'ticket_confirm_many', vars: { n: legs.length }, safe: true };
  const leg = legs[0];
  const side = legSide(leg);
  if (leg.qty !== null && leg.qty !== undefined) {
    return { key: `ticket_confirm_${side}_qty`, vars: { qty: formatQty(text(leg.qty)), symbol: text(leg.symbol) }, safe: true };
  }
  return {
    key: `ticket_confirm_${side}_notional`,
    vars: { amount: formatMoney(text(leg.notional)), symbol: text(leg.symbol) },
    safe: true,
  };
}

/* Estimated buy cost and sell value over the executable decisions. `complete` is false when a leg
 * has no estimate, so the card can say that the totals leave something out. */
export function ticketTotals(ticket) {
  let buy = '0';
  let sell = '0';
  let buys = 0;
  let sells = 0;
  let complete = true;
  for (const decision of Array.isArray(ticket?.decisions) ? ticket.decisions : []) {
    if (!decision || decision.verdict === 'reject') continue;
    const side = legSide(decision.leg);
    const cost = text(decision.est_cost_usd);
    if (side === 'sell') sells += 1;
    else buys += 1;
    const sum = add(side === 'sell' ? sell : buy, cost);
    if (!sum) {
      complete = false;
      continue;
    }
    if (side === 'sell') sell = sum;
    else buy = sum;
  }
  return { buy, sell, buys, sells, complete };
}

/* Buying power left after the buys (sale proceeds are not counted: they settle later). */
export function buyingPowerAfter(buyingPower, totals) {
  return sub(text(buyingPower), totals.buy);
}

/* A position's share of equity after this leg fills, in percent (est.), or null. */
export function positionPercentAfter(decision, positions, equity) {
  const symbol = text(decision?.leg?.symbol);
  const cost = text(decision?.est_cost_usd);
  if (!symbol || !cost || !isPositiveDecimal(text(equity))) return null;
  const held = (Array.isArray(positions) ? positions : []).find((p) => text(p?.symbol) === symbol);
  const current = held ? text(held.market_value) : '0';
  const after = legSide(decision.leg) === 'sell' ? sub(current, cost) : add(current, cost);
  if (!after) return null;
  const floor = cmp(after, '0') < 0 ? '0' : after;
  const share = div(floor, text(equity), 6);
  return share ? mul(share, '100') : null;
}

/* What risk changed on a clamped leg: [{field, from, to}] for qty, notional and limit_price. */
export function clampChanges(decision) {
  const out = [];
  if (!decision || decision.verdict !== 'clamp') return out;
  for (const field of ['qty', 'notional', 'limit_price']) {
    const from = decision.original?.[field];
    const to = decision.leg?.[field];
    if (typeof from === 'string' && typeof to === 'string' && cmp(from, to) !== 0) out.push({ field, from, to });
  }
  return out;
}

/* Old and new estimated prices per leg after a requote: [{symbol, side, from, to}]. Legs are paired
 * by position among the executable decisions; a leg that disappeared has `to: null`. */
export function requoteChanges(oldTicket, newTicket) {
  const live = (ticket) => (Array.isArray(ticket?.decisions) ? ticket.decisions : [])
    .filter((d) => d && d.verdict !== 'reject');
  const next = live(newTicket);
  return live(oldTicket).map((decision, index) => {
    const match = next[index] && text(next[index].leg?.symbol) === text(decision.leg?.symbol) ? next[index] : null;
    return {
      symbol: text(decision.leg?.symbol),
      side: legSide(decision.leg) || '',
      from: text(decision.est_price) || null,
      to: match ? text(match.est_price) || null : null,
    };
  });
}

/* The card's state from the ticket's stored status and the clock. A proposed ticket past its
 * expiry is `expired` on screen even before the server has swept it. */
export function ticketState(ticket, nowMs) {
  const status = text(ticket?.status) || 'proposed';
  if (status === 'proposed') {
    const expires = Date.parse(text(ticket.expires_at));
    if (!Number.isNaN(expires) && nowMs >= expires) return 'expired';
  }
  return status;
}

/* What a confirm, reject or refresh answer means for the card. `result` is
 * {ok, status (HTTP), error (code), body}. Returns {kind, …}; the card acts on `kind`:
 *   submitted | requote | rejected (message) | expired | disabled | no_broker | uncertain | dismissed
 *   | not_found | failed (message; nothing is known to have been placed) */
export function interpretResult(action, result) {
  const body = result.body && typeof result.body === 'object' ? result.body : {};
  const message = text(body.message);
  const base = { message, ticket: body.ticket || null, newTicket: body.new_ticket || null, orders: body.orders || [],
    reason: text(body.reason), blockCodes: Array.isArray(body.block_codes) ? body.block_codes : [] };
  const http = result.status;
  if (result.ok) {
    if (action === 'confirm') return { ...base, kind: 'submitted' };
    if (action === 'reject') return { ...base, kind: 'dismissed' };
    return base.newTicket ? { ...base, kind: 'requote' } : { ...base, kind: 'rejected' };
  }
  if (http === 409 && body.status === 'requote' && base.newTicket) return { ...base, kind: 'requote' };
  if (http === 409) return { ...base, kind: 'rejected' };
  if (http === 410) return { ...base, kind: 'expired' };
  if (http === 423) return { ...base, kind: 'disabled' };
  if (http === 503 && result.error === 'broker_not_connected') return { ...base, kind: 'no_broker' };
  if (http === 404) return { ...base, kind: 'not_found' };
  // 502 (an executor error), 500 confirm_failed, or no answer at all: a confirm may have reached the
  // broker, so the card says "check open orders" rather than "failed".
  if (action === 'confirm' && (http === 502 || http === 500 || http === 0)) return { ...base, kind: 'uncertain' };
  return { ...base, kind: 'failed' };
}

// Risk block codes (src/finagent/trading/risk.py) that come from the mandate, and those that only
// say the prices do not fit right now. (One string each: scripts/check_i18n.mjs reads a quoted
// `order_…` word as an i18n key.)
const MANDATE_CODES = new Set('denied_symbol order_type_not_allowed max_positions budget orders_per_day'.split(' '));
const PRICE_CODES = new Set('price_band no_quote bad_tick'.split(' '));

/* Why a refresh was refused (409), from the result's `reason` and `block_codes`:
 *   newer      the chat has a newer ticket (`not_newest`)
 *   mandate    a mandate rule blocks a leg
 *   prices     only the price checks block the legs (prices may move back)
 *   not_paper  the connected account is not a paper account
 *   other      anything else, or a server that gives no reason */
export function refusalKind(reason, blockCodes) {
  const codes = Array.isArray(blockCodes) ? blockCodes.filter((code) => typeof code === 'string') : [];
  if (reason === 'not_newest') return 'newer';
  if (reason === 'not_paper') return 'not_paper';
  if (reason === 'no_executable_legs' && codes.length) {
    if (codes.some((code) => MANDATE_CODES.has(code))) return 'mandate';
    if (codes.every((code) => PRICE_CODES.has(code))) return 'prices';
  }
  return 'other';
}

/* The user message that asks the agent for a changed ticket. `edits` is one entry per executable
 * leg: {side, symbol, qty | notional, limit_price?}. It starts with "Modify:" in both languages (the
 * agent treats it as an instruction to re-propose); `lang === 'ko'` adds a Korean closing line so
 * the reply comes back in Korean. null when a leg's side cannot be read: it is never guessed. */
export function modifyMessage(edits, lang = 'en') {
  if (!edits.every((edit) => legSide(edit))) return null;
  const parts = edits.map((edit) => {
    const side = legSide(edit);
    const size = edit.qty !== undefined && edit.qty !== null
      ? `${toTrimmed(edit.qty) ?? edit.qty} ${edit.symbol}`
      : `$${edit.notional} of ${edit.symbol}`;
    const price = edit.limit_price ? ` limit ${edit.limit_price}` : ' at market';
    return `${side} ${size}${price}`;
  });
  const tail = lang === 'ko' ? ' 이 내용으로 주문 제안을 수정해 주세요.' : '';
  return `Modify: ${parts.join('; ')}${tail}`;
}

/* Check the Modify fields. Returns {edits, errors}; `errors` maps "index.field" to an i18n key. */
export function readModify(legs, values) {
  const edits = [];
  const errors = {};
  legs.forEach((leg, index) => {
    const typed = values[index] || {};
    const edit = { side: legSide(leg) || '', symbol: text(leg.symbol) };
    const sizeField = leg.qty !== null && leg.qty !== undefined ? 'qty' : 'notional';
    const size = text(typed[sizeField]).trim().replaceAll(',', '');
    if (!PLAIN_POSITIVE.test(size) || !isPositiveDecimal(size)) errors[`${index}.${sizeField}`] = 'ticket_modify_err';
    edit[sizeField] = size;
    if (text(leg.type) === 'limit') {
      const price = text(typed.limit_price).trim().replaceAll(',', '');
      if (!PLAIN_POSITIVE.test(price) || !isPositiveDecimal(price)) errors[`${index}.limit_price`] = 'ticket_modify_err';
      edit.limit_price = price;
    }
    edits.push(edit);
  });
  return { edits, errors };
}
