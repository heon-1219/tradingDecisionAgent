/* The Portfolio panel: the paper account's equity, cash, buying power and day change, and its
 * positions. Three states besides the data: not connected (a link to Settings), a broker error
 * (a retry button) and loading. It reloads when the tab is shown, after a confirm, and from its
 * refresh button.
 *
 * Paper status: "PAPER (verified)" when the account says it is a paper account, "PAPER" when it
 * does not say, and a prominent warning when it is not one (in the data, or when the Settings
 * report `alpaca: "not_paper"` and the server therefore shows no account at all).
 *
 * Amounts are the server's decimal strings, formatted on the digits. The two derived figures (the
 * day change and its percentage) are computed with exact decimal arithmetic. A gain or a loss is
 * written with its sign and a ▲/▼ mark, never by colour alone.
 */
import { cmp, div, formatMoney, formatPercent, formatQty, isZero, mul, sub } from '../decimal.js';
import { registerPanel } from '../registry.js';
import { emit, on, state } from '../state.js';
import { actionButton, badge, el, iconButton, iconSpan, label, str } from './dom.js';
import { refreshPortfolio, snapshot } from './portfolio-store.js';

let container = null;

/* A signed amount with its direction mark: "▲ +$12.50", "▼ −$3.10", "$0.00". */
export function changeParts(value) {
  const direction = cmp(value, '0');
  if (direction === null) return { mark: '', tone: 'muted', text: '–' };
  return {
    mark: direction > 0 ? '▲' : direction < 0 ? '▼' : '',
    tone: direction > 0 ? 'up' : direction < 0 ? 'down' : 'flat',
    text: formatMoney(value, { signed: true }),
  };
}

function changeNode(value, percent) {
  const parts = changeParts(value);
  const node = el('span', 'change');
  node.setAttribute('data-tone', parts.tone);
  if (parts.mark) {
    const mark = el('span', 'change-mark', parts.mark);
    mark.setAttribute('aria-hidden', 'true');
    node.appendChild(mark);
  }
  node.appendChild(el('span', 'change-amount', parts.text));
  if (percent) node.appendChild(el('span', 'change-pct', `(${formatPercent(percent, { signed: true })})`));
  return node;
}

function stat(key, valueNode) {
  const box = el('div', 'stat');
  box.append(label('dt', 'stat-label', key), valueNode);
  return box;
}

function money(value) {
  return el('dd', 'stat-value', formatMoney(str(value)));
}

/* The account's paper status: verified by the broker's own account number, assumed, or NOT paper. */
function paperBadge(account) {
  if (account && account.paper === true) return badge('warn', 'paper_verified');
  if (account && account.paper === false) return null; // the warning below replaces the badge
  return badge('warn', 'paper');
}

function header(body) {
  const head = el('div', 'panel-section-head');
  const title = el('div', 'panel-title');
  title.appendChild(label('h2', 'panel-h', 'portfolio_heading'));
  const paper = body ? paperBadge(body.account) : null;
  if (paper) title.appendChild(paper);
  const refresh = iconButton('refresh', 'portfolio_refresh');
  refresh.addEventListener('click', () => refreshPortfolio());
  head.append(title, refresh);
  return head;
}

function notice(tone, iconName, key, action) {
  const box = el('div', 'notice');
  box.setAttribute('data-tone', tone);
  const text = el('div', 'notice-body');
  text.appendChild(typeof key === 'string' ? label('p', 'notice-text', key) : key);
  if (action) text.appendChild(action);
  box.append(iconSpan(iconName, 18), text);
  return box;
}

function positionsTable(positions) {
  const wrap = el('div', 'table-scroll');
  wrap.tabIndex = 0;
  wrap.setAttribute('role', 'region');
  wrap.setAttribute('aria-labelledby', 'positions-heading');
  const table = el('table', 'data-table');
  const headRow = el('tr');
  for (const [key, numeric] of [['col_symbol', false], ['col_qty', true], ['col_avg_entry', true],
    ['col_price', true], ['col_value', true], ['col_pl', true]]) {
    const th = label('th', numeric ? 'num' : '', key);
    th.scope = 'col';
    headRow.appendChild(th);
  }
  const thead = el('thead');
  thead.appendChild(headRow);
  const tbody = el('tbody');
  for (const position of positions) {
    const row = el('tr');
    const symbol = el('th', 'symbol', str(position.symbol));
    symbol.scope = 'row';
    const pl = el('td', 'num');
    const percent = mul(str(position.unrealized_plpc), '100');
    pl.appendChild(changeNode(str(position.unrealized_pl), percent));
    row.append(
      symbol,
      el('td', 'num', formatQty(str(position.qty))),
      el('td', 'num', formatMoney(str(position.avg_entry_price))),
      el('td', 'num', formatMoney(str(position.current_price))),
      el('td', 'num', formatMoney(str(position.market_value))),
      pl,
    );
    tbody.appendChild(row);
  }
  table.append(thead, tbody);
  wrap.appendChild(table);
  return wrap;
}

function accountView(body) {
  const account = body.account || {};
  const nodes = [];
  if (account.paper === false) {
    nodes.push(notice('bad', 'alert', 'paper_not_verified'));
  }
  const change = sub(str(account.equity), str(account.last_equity));
  const base = str(account.last_equity);
  const percent = change && isZero(base) === false ? mul(div(change, base, 6), '100') : null;
  const stats = el('dl', 'stats');
  const day = el('dd', 'stat-value');
  day.appendChild(changeNode(change, percent));
  stats.append(
    stat('stat_equity', money(account.equity)),
    stat('stat_cash', money(account.cash)),
    stat('stat_buying_power', money(account.buying_power)),
    stat('stat_day_change', day),
  );
  nodes.push(stats);
  if (account.trading_blocked || account.account_blocked || account.trade_suspended_by_user) {
    nodes.push(notice('warn', 'alert', 'account_blocked'));
  }
  const positions = Array.isArray(body.positions) ? body.positions : [];
  const section = el('section', 'panel-section');
  const heading = label('h2', 'panel-h', 'positions_heading', { n: positions.length });
  heading.id = 'positions-heading';
  section.appendChild(heading);
  if (positions.length === 0) section.appendChild(label('p', 'panel-note', 'positions_empty'));
  else section.appendChild(positionsTable(positions));
  nodes.push(section);
  return nodes;
}

function settingsLink() {
  const button = actionButton('btn btn-quiet', 'open_settings', { iconName: 'sliders' });
  button.addEventListener('click', () => emit('panel:open', 'settings'));
  return button;
}

function retryButton() {
  const button = actionButton('btn btn-quiet', 'retry', { iconName: 'refresh' });
  button.addEventListener('click', () => refreshPortfolio());
  return button;
}

function render() {
  if (!container) return;
  const data = snapshot();
  const keepFocus = container.contains(document.activeElement);
  const nodes = [header(data.body)];
  if (data.status === 'disconnected' && state.settings && state.settings.alpaca === 'not_paper') {
    // The keys belong to a live account: the server keeps no broker for them and shows no data.
    nodes.push(notice('bad', 'alert', 'paper_not_verified', settingsLink()));
  } else if (data.status === 'disconnected') {
    nodes.push(notice('muted', 'wallet', 'portfolio_disconnected', settingsLink()));
  } else if (data.status === 'error') {
    const message = el('p', 'notice-text', data.message);
    nodes.push(notice('bad', 'alert', message, retryButton()));
    if (data.body) nodes.push(label('p', 'panel-note', 'portfolio_stale'), ...accountView(data.body));
  } else if (data.body) {
    nodes.push(...accountView(data.body));
  } else {
    nodes.push(label('p', 'panel-note', 'portfolio_loading'));
  }
  container.setAttribute('aria-busy', String(data.status === 'loading'));
  container.replaceChildren(...nodes);
  if (keepFocus) container.querySelector('.icon-btn')?.focus();
}

registerPanel('portfolio', {
  mount(target) {
    container = target;
    render();
  },
  show() {
    refreshPortfolio();
  },
});

on('portfolio:data', render);
on('change', (patch) => {
  if ('settings' in patch) render();
});
