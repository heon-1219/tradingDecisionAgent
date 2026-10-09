/* The parts of a ticket card that only display the ticket: the legs with the risk engine's verdict,
 * the estimated totals, the reasoning and the evidence. `ticket.js` owns the state and the buttons.
 *
 * Everything the model or the broker wrote (thesis, risks, invalidation, reason messages, symbols)
 * is rendered as text. Estimated figures the server computed are shown as sent; the two the UI
 * derives (buying power and position share after the trade) are labelled "est." and use exact
 * decimal arithmetic.
 */
import { formatMoney, formatPercent, formatQty } from '../decimal.js';
import { badge, el, label, str } from './dom.js';
import { sideBadge, sizeNode, termsNode } from './order-view.js';
import { sourceChip } from './sources.js';
import { buyingPowerAfter, clampChanges, legSide, positionPercentAfter, ticketTotals } from './ticket-model.js';

const VERDICT = {
  accept: ['verdict_accept', 'ok'],
  clamp: ['verdict_clamp', 'warn'],
  reject: ['verdict_reject', 'bad'],
};
const SEVERITY = { info: 'severity_info', warn: 'severity_warn', block: 'severity_block' };
const CLAMP_KEY = { qty: 'clamp_qty', notional: 'clamp_notional', limit_price: 'clamp_limit' };

function figure(key, value, className = '') {
  const box = el('div', `figure ${className}`.trim());
  box.append(label('dt', 'figure-label', key), el('dd', 'figure-value', value));
  return box;
}

function reasonItem(reason) {
  const item = el('li', 'reason');
  const severity = str(reason?.severity);
  item.setAttribute('data-severity', severity || 'info');
  const tag = SEVERITY[severity] ? label('span', 'reason-sev', SEVERITY[severity]) : el('span', 'reason-sev', severity);
  item.append(tag, el('code', 'reason-code', str(reason?.code)), el('span', 'reason-text', str(reason?.message)));
  return item;
}

function clampLine(change) {
  const money = change.field !== 'qty';
  const show = (value) => (money ? formatMoney(value) : formatQty(value));
  return label('p', 'leg-clamp', CLAMP_KEY[change.field], { from: show(change.from), to: show(change.to) });
}

/* One leg with its verdict. Returns {el, setPortfolio(body)}. */
function legItem(decision) {
  const leg = decision.leg || {};
  const verdict = str(decision.verdict);
  const item = el('li', 'leg');
  item.setAttribute('data-side', legSide(leg) || 'unknown');
  item.setAttribute('data-verdict', verdict);

  const head = el('div', 'leg-head');
  head.append(sideBadge(leg.side), el('span', 'leg-symbol', str(leg.symbol)), sizeNode(leg), termsNode(leg));

  const figures = el('dl', 'leg-figures');
  figures.appendChild(figure('ticket_est_price', formatMoney(str(decision.est_price))));
  figures.appendChild(figure(legSide(leg) === 'sell' ? 'ticket_est_proceeds' : 'ticket_est_cost',
    formatMoney(str(decision.est_cost_usd))));
  const share = figure('ticket_position_after', '–', 'is-derived');
  share.hidden = true;
  figures.appendChild(share);

  const [verdictKey, tone] = VERDICT[verdict] || ['verdict_unknown', 'muted'];
  const verdictRow = el('div', 'leg-verdict');
  verdictRow.appendChild(badge(tone, verdictKey));
  if (verdict === 'reject') verdictRow.appendChild(label('span', 'leg-note', 'leg_not_placed'));
  for (const change of clampChanges(decision)) verdictRow.appendChild(clampLine(change));

  item.append(head, figures, verdictRow);
  const reasons = Array.isArray(decision.reasons) ? decision.reasons : [];
  if (reasons.length) {
    const list = el('ul', 'leg-reasons');
    list.append(...reasons.map(reasonItem));
    item.appendChild(list);
  }
  return {
    el: item,
    setPortfolio(body) {
      const percent = verdict !== 'reject' && body
        ? positionPercentAfter(decision, body.positions, body.account?.equity) : null;
      share.hidden = !percent;
      if (percent) share.lastChild.textContent = formatPercent(percent, { places: 1 });
    },
  };
}

function reasoning(ticket) {
  const details = el('details', 'ticket-why');
  details.appendChild(label('summary', 'ticket-why-summary', 'ticket_why'));
  const list = el('dl', 'ticket-why-list');
  for (const [key, value] of [['ticket_thesis', ticket.thesis], ['ticket_risks', ticket.risks],
    ['ticket_invalidation', ticket.invalidation]]) {
    list.append(label('dt', '', key), el('dd', '', str(value).trim() || '–'));
  }
  details.appendChild(list);
  return details;
}

/* The display parts of a ticket. Returns {nodes, setPortfolio(snapshot), refreshEvidence()}. */
export function ticketBody(ticket, sessionId) {
  const decisions = (Array.isArray(ticket.decisions) ? ticket.decisions : []).filter((d) => d && d.leg);
  const legs = decisions.map(legItem);
  const list = el('ol', 'ticket-legs');
  list.append(...legs.map((leg) => leg.el));

  const totals = ticketTotals(ticket);
  const sums = el('dl', 'ticket-totals');
  if (totals.buys > 0) sums.appendChild(figure('ticket_total_buy', formatMoney(totals.buy)));
  if (totals.sells > 0) sums.appendChild(figure('ticket_total_sell', formatMoney(totals.sell)));
  const power = figure('ticket_power_after', '–', 'is-derived');
  power.hidden = true;
  sums.appendChild(power);
  const nodes = [list, sums];
  if (!totals.complete) nodes.push(label('p', 'ticket-caveat', 'ticket_totals_partial'));
  nodes.push(reasoning(ticket));

  const ids = (Array.isArray(ticket.evidence_ids) ? ticket.evidence_ids : []).filter((id) => /^E\d{1,4}$/.test(id));
  const evidence = el('div', 'ticket-evidence');
  const chips = el('span', 'ticket-evidence-chips');
  const refreshEvidence = () => chips.replaceChildren(...ids.map((id) => sourceChip(sessionId, id)));
  if (ids.length) {
    refreshEvidence();
    evidence.append(label('span', 'ticket-evidence-label', 'ticket_evidence'), chips);
    nodes.push(evidence);
  }

  return {
    nodes,
    refreshEvidence,
    setPortfolio(snapshot) {
      const body = snapshot && snapshot.status === 'ready' ? snapshot.body : null;
      for (const leg of legs) leg.setPortfolio(body);
      const after = body && totals.buys > 0 ? buyingPowerAfter(body.account?.buying_power, totals) : null;
      power.hidden = !after;
      if (after) {
        power.lastChild.textContent = formatMoney(after);
        power.setAttribute('data-short', String(after.int < 0n));
      }
    },
  };
}
