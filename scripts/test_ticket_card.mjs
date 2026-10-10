// Tests for the trade ticket card (web/app/js/components/ticket.js) on the fake DOM.
// Run: node scripts/test_ticket_card.mjs
//
// The real card is driven through every answer the ticket endpoints can give (confirm 200, 409
// requote, 409 rejected, 410, 423, 503, 502, 404, and no answer at all), with `fetch` replaced by a
// scripted stand-in: what the Confirm button says, that it sends the hash exactly once, which state
// the card ends in, and that no state offers a blind retry. The setup is scripts/component_harness.mjs.
import assert from 'node:assert/strict';
import {
  advance, clock, decision, dict, dispatch, document, emit, finish, flush, leg, liveTurn, net, on, order, panel,
  refreshPortfolio,
  replayTurn,
  setState, sources, t, test, thread, ticker, ticket,
} from './component_harness.mjs';
import { setLang } from '../web/app/js/i18n.js';
import { ticketIds } from '../web/app/js/components/ticket-store.js';

const { calls } = net;
const cardOf = (turn) => turn.slots.extras.querySelector('.ticket');
const button = (card, cls) => card.querySelector(`.${cls}`);
const noticeText = (card) => card.querySelector('.ticket-notice').textContent;
// The server's own words, kept in a collapsed detail row (only when they add something).
const serverDetail = (card) => card.querySelector('.ticket-notice').querySelector('.notice-detail');
const confirmPosts = () => calls.filter((c) => c.method === 'POST' && c.path.endsWith('/confirm'));
// The live region of index.html. notify.js clears it, then fills it 60 ms later, so one announce() is one
// write of '' and the last write is what a screen reader reads; the spy keeps every write.
const announcer = document.createElement('div');
announcer.id = 'announcer';
document.body.appendChild(announcer);
const writes = [];
let spoken = '';
Object.defineProperty(announcer, 'textContent', {
  get: () => spoken,
  set(value) {
    spoken = String(value);
    writes.push(spoken);
  },
});
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
/* Let every pending announcement land, then forget what was announced so far. */
const quietAnnouncer = async () => {
  await wait(90);
  writes.length = 0;
};
/* How many announcements were made since, and the text now in the live region. */
const announced = async () => {
  await wait(90);
  return { calls: writes.filter((w) => w === '').length, text: spoken };
};
// E1 is a known source of chat s1; the tickets below also cite E9, which is not.
sources.seedSources('s1',[{ id: 'E1', title: 'Alpaca IEX quotes', publisher: 'Alpaca Markets', tier: 1 }]);

// --- the trade ticket card ------------------------------------------------------------------------------------

function propose(tk, sessionId = 's1') {
  const turn = liveTurn(sessionId);
  dispatch({ type: 'data-trade-proposal', ticket: tk }, turn);
  return { turn, card: cardOf(turn) };
}
const ticketRead = (tk, orders = []) => ({ method: 'GET', match: `/api/tickets/${tk.id}?`, status: 200, body: { ticket: tk, orders } });
const portfolio = { method: 'GET', match: '/api/portfolio', status: 200, body: { connected: true, paper: true,
  account: { equity: '132100.00', cash: '100000.00', buying_power: '100000.00', non_marginable_buying_power: '100000.00',
    last_equity: '131000.00', paper: true },
  positions: [{ symbol: 'VOO', market_value: '12000.00' }], open_orders: [] } };
setState({ sessionId: 's1' });

await test('ticket card: says exactly what will be placed, with the verdicts as text', async () => {
  net.routes = [portfolio];
  const tk = ticket({
    thesis: '<img src=x onerror=alert(1)> thesis', risks: '**bold** [link](javascript:alert(1))',
    decisions: [
      decision(0, { qty: '6' }, { verdict: 'clamp', original: leg({ qty: '10' }), est_cost_usd: '3660.00',
        reasons: [{ code: 'max_position_pct', message: 'Cut to fit <b>20%</b>', severity: 'warn' }] }),
      decision(1, { symbol: 'XYZ' }, { verdict: 'reject', reasons: [{ code: 'not_tradable', message: 'Not tradable', severity: 'block' }] }),
    ],
  });
  const { card } = propose(tk);
  await flush();
  assert.equal(card.getAttribute('data-state'), 'proposed');
  const confirm = button(card, 'btn-confirm');
  assert.equal(confirm.textContent, 'Confirm buy 6 VOO');
  assert.equal(confirm.disabled, false);
  assert.notEqual(document.activeElement, confirm, 'Confirm is never focused by the app');
  const legs = card.querySelectorAll('.leg');
  assert.equal(legs.length, 2);
  assert.equal(legs[0].querySelector('.side-badge').textContent, 'BUY▲');
  assert.equal(legs[0].querySelector('.leg-verdict').querySelector('.badge').textContent, 'Clamped');
  assert.equal(legs[0].querySelector('.leg-clamp').textContent, 'Quantity changed from 10 to 6');
  assert.equal(legs[0].querySelector('.reason').textContent, 'Warningmax_position_pctCut to fit <b>20%</b>');
  assert.equal(legs[1].getAttribute('data-verdict'), 'reject');
  assert.equal(legs[1].querySelector('.leg-note').textContent, 'This leg will not be placed.');
  const figures = card.querySelector('.ticket-totals').querySelectorAll('.figure');
  assert.equal(figures[0].textContent, 'Est. total buy cost$3,660.00');
  assert.equal(figures[1].textContent, 'Buying power after (est.)$96,340.00');
  assert.equal(figures[1].hidden, false);
  assert.equal(card.querySelector('.ticket-age').textContent, 'Quoted 12 s ago');
  assert.equal(card.querySelector('.ticket-left').textContent, 'Expires in 4:48');
  // model text is text: no element was created from it, and it is not run through markdown
  const why = card.querySelector('.ticket-why-list').querySelectorAll('dd');
  assert.equal(why[0].textContent, '<img src=x onerror=alert(1)> thesis');
  assert.equal(why[1].textContent, '**bold** [link](javascript:alert(1))');
  assert.ok(!document.created.includes('img'));
  assert.equal(card.querySelectorAll('a').length, 0);
  // evidence chips: E1 is known, E9 is not
  const chips = card.querySelector('.ticket-evidence-chips').children;
  assert.deepEqual(chips.map((c) => [c.localName, c.textContent]), [['button', 'E1'], ['span', 'E9']]);
});

await test('countdown: one shared ticker; at expiry the card greys and offers Refresh quote', async () => {
  net.routes = [portfolio];
  const before = ticker.tickerStats().running;
  const tk = ticket();
  const { card } = propose(tk);
  assert.equal(ticker.tickerStats().running, true);
  assert.equal(before || clock.ticks.length === 1, true);
  assert.equal(clock.ticks.length, 1, 'every card shares one interval');
  advance(260000);
  assert.equal(card.querySelector('.ticket-left').textContent, 'Expires in 0:28');
  assert.equal(card.querySelector('.ticket-left').getAttribute('data-urgent'), 'true');
  assert.equal(card.querySelector('.ticket-age').textContent, 'Quoted 4 min ago');
  advance(28000);
  assert.equal(card.getAttribute('data-state'), 'expired');
  assert.equal(button(card, 'btn-confirm'), null);
  const actions = card.querySelector('.ticket-actions').querySelectorAll('button');
  assert.deepEqual(actions.map((b) => b.textContent), ['Refresh quote']);
  // The chat's newest ticket is re-quoted even after it expired (200): it stays expired, and the
  // new ticket follows it with its own Confirm.
  const next = ticket({ decisions: [decision(0, {}, { est_price: '611.00', est_cost_usd: '1833.00' })] });
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/refresh`, status: 200, body: { status: 'requote',
    message: 'Re-quoted at current prices; review the new ticket.', ticket: { ...tk, status: 'expired' }, orders: [], new_ticket: next } }];
  actions[0].click();
  await flush();
  assert.equal(card.getAttribute('data-state'), 'expired');
  assert.equal(card.querySelector('.ticket-actions'), null, 'no second Refresh quote: a newer ticket exists');
  assert.ok(noticeText(card).includes('Re-quoted at current prices'));
  const inBox = card.parentNode.querySelectorAll('.ticket');
  const fresh = inBox[inBox.indexOf(card) + 1]; // right after the expired card
  assert.equal(fresh.getAttribute('data-ticket-id'), next.id);
  assert.equal(fresh.getAttribute('data-state'), 'proposed');
  assert.equal(button(fresh, 'btn-confirm').disabled, false);
  fresh.remove();
  clock.time -= 288000; // back to "now" for the following tests
});

/* Press Refresh quote on an expired card and have the server refuse it (409) with `reason`. */
async function refuseRefresh(tk, reason, blockCodes, message) {
  const { card } = propose(tk);
  assert.equal(card.getAttribute('data-state'), 'expired');
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/refresh`, status: 409, body: { status: 'rejected',
    message, ticket: tk, orders: [], new_ticket: null, ...(reason === undefined ? {} : { reason, block_codes: blockCodes }) } }];
  card.querySelector('.ticket-actions').querySelector('button').click();
  await flush();
  assert.equal(card.getAttribute('data-state'), 'expired');
  return card;
}
const noticeButtons = (card) => card.querySelector('.ticket-notice').querySelectorAll('button');
const refreshLeft = (card) => Boolean(card.querySelector('.ticket-actions'));

await test('refused refresh, a newer ticket exists: points to it and offers a new ticket, no second Refresh', async () => {
  const tk = ticket({ status: 'expired' });
  const { card } = propose(tk);
  const newer = propose(ticket()).card; // the chat's newer ticket, further down
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/refresh`, status: 409, body: { status: 'rejected',
    message: 'ticket is expired and a newer ticket exists; nothing was re-quoted', ticket: tk, orders: [], new_ticket: null,
    reason: 'not_newest', block_codes: [] } }];
  card.querySelector('.ticket-actions').querySelector('button').click();
  await flush();
  assert.ok(noticeText(card).includes('This chat has a newer ticket'));
  assert.equal(serverDetail(card), null, 'the localized text says it all');
  assert.equal(refreshLeft(card), false);
  assert.deepEqual(noticeButtons(card).map((b) => b.textContent), ['Show the newer ticket', 'Ask for a new ticket']);
  noticeButtons(card)[0].click();
  assert.equal(document.activeElement, newer, 'the newer ticket gets the focus');
  const sent = [];
  const stop = on('chat:send', (text) => sent.push(text));
  noticeButtons(card)[1].click();
  stop();
  assert.deepEqual(sent, ['Modify: buy 3 VOO limit 612.00']);
});

await test('refused refresh, other reasons: neutral words, the server\'s reason in the detail row, the right offer', async () => {
  // prices alone: Refresh stays (prices move), and a new ticket is offered
  let card = await refuseRefresh(ticket({ status: 'expired' }), 'no_executable_legs', ['price_band'],
    'No leg can be placed at current prices: leg 1 VOO: limit is 9% away from the last price');
  assert.ok(noticeText(card).includes('price checks'));
  assert.ok(!noticeText(card).includes('newer ticket'));
  assert.ok(serverDetail(card).textContent.includes('9% away'));
  assert.deepEqual(noticeButtons(card).map((b) => b.textContent), ['Ask for a new ticket']);
  assert.equal(refreshLeft(card), true);
  // the mandate (the reviewer's case): explain, link the Mandate panel, no new-ticket offer
  card = await refuseRefresh(ticket({ status: 'expired' }), 'no_executable_legs', ['denied_symbol'],
    "No leg can be placed at current prices: leg 1 VOO: VOO is on the mandate's deny list");
  assert.ok(noticeText(card).includes('mandate does not allow'));
  assert.ok(!noticeText(card).includes('newer ticket'));
  assert.ok(serverDetail(card).textContent.includes('deny list'));
  assert.deepEqual(noticeButtons(card).map((b) => b.textContent), ['Open Mandate']);
  const opened = [];
  const stop = on('panel:open', (name) => opened.push(name));
  noticeButtons(card)[0].click();
  // not a paper account: explain, link Settings
  card = await refuseRefresh(ticket({ status: 'expired' }), 'not_paper', [],
    'The connected Alpaca account is not a paper account, so nothing was re-quoted.');
  assert.ok(noticeText(card).includes('not a paper account'));
  assert.deepEqual(noticeButtons(card).map((b) => b.textContent), ['Open Settings']);
  noticeButtons(card)[0].click();
  stop();
  assert.deepEqual(opened, ['mandate', 'settings']);
  // anything else, and an older server without `reason`: neutral, never "a newer ticket"
  for (const [reason, codes] of [['no_executable_legs', ['account_blocked']], ['ticket_state', []], [undefined, undefined]]) {
    card = await refuseRefresh(ticket({ status: 'expired' }), reason, codes, 'No leg can be placed: the account cannot trade');
    assert.ok(noticeText(card).includes('could not be re-quoted'), String(reason));
    assert.ok(!noticeText(card).includes('newer ticket'), String(reason));
    assert.ok(serverDetail(card).textContent.includes('cannot trade'));
    assert.deepEqual(noticeButtons(card), []);
  }
});

await test('a card whose order cannot be read offers neither Modify nor "Ask for a new ticket"', async () => {
  net.routes = [portfolio];
  const { card } = propose(ticket({ decisions: [decision(0, { side: 'short' })] }));
  assert.equal(button(card, 'btn-confirm').disabled, true);
  assert.equal(button(card, 'btn-modify').disabled, true);
  button(card, 'btn-modify').click();
  assert.equal(card.querySelector('.ticket-modify-host').children.length, 0, 'no Modify fields open');
  const expired = await refuseRefresh(ticket({ status: 'expired', decisions: [decision(0, { qty: 'abc' })] }),
    'no_executable_legs', ['price_band'], 'No leg can be placed at current prices');
  const [ask] = noticeButtons(expired);
  assert.equal(ask.textContent, 'Ask for a new ticket');
  assert.equal(ask.disabled, true);
  const sent = [];
  const stop = on('chat:send', (text) => sent.push(text));
  ask.click();
  stop();
  assert.deepEqual(sent, []);
});

await test('confirm 200: sends the hash once, shows the order timeline, no Confirm left', async () => {
  const tk = ticket();
  const done = { ...tk, status: 'submitted', broker_order_ids: ['o1'] };
  net.routes = [portfolio,
    { method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status: 200, body: { status: 'submitted', message: '1 of 1 orders placed.', ticket: done, orders: [order()], new_ticket: null } },
    ticketRead(done, [{ client_order_id: 'c0', broker_order_id: 'o1', order: order() }])];
  const { card } = propose(tk);
  calls.length = 0;
  const confirm = button(card, 'btn-confirm');
  confirm.click();
  // while the request is in flight every button is disabled and a second click does nothing
  assert.equal(card.getAttribute('aria-busy'), 'true');
  assert.ok(card.querySelector('.ticket-actions').querySelectorAll('button').every((b) => b.disabled));
  button(card, 'btn-confirm').click();
  await flush();
  const posts = calls.filter((c) => c.method === 'POST');
  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].body, { session_id: 's1', hash: tk.hash });
  assert.equal(posts[0].headers['X-FinAgent-Client'], '1');
  assert.equal(card.getAttribute('data-state'), 'submitted');
  assert.equal(button(card, 'btn-confirm'), null);
  assert.equal(card.querySelector('.ticket-actions'), null);
  // the banner is built from the structured result, in the interface language
  const summary = card.querySelector('.ticket-notice');
  assert.equal(summary.querySelector('.notice-text').textContent, '1 of 1 orders were sent to the paper account.');
  assert.equal(summary.querySelector('.notice-order').textContent, 'BUY▲VOO3 sharesLimit $612.00 · DayFilled');
  assert.equal(serverDetail(card), null, 'everything placed: the server text adds nothing');
  assert.ok(!noticeText(card).includes('orders placed.'));
  const row = card.querySelector('.timeline-row');
  assert.equal(row.querySelector('.timeline-main').textContent, 'BUY▲VOO3 shares');
  assert.ok(row.querySelector('.timeline-status').textContent.includes('Filled'));
  assert.ok(row.querySelector('.timeline-status').textContent.includes('3 of 3 filled at $610.10'));
  assert.equal(card.querySelector('.timeline-foot').textContent, 'Every order is final.');
});

await test('a click queued against a button that was already replaced sends nothing while busy', async () => {
  const tk = ticket();
  let open;
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status: 200,
    gate: new Promise((resolve) => { open = resolve; }),
    body: { status: 'submitted', message: 'ok', ticket: { ...tk, status: 'submitted' }, orders: [order()], new_ticket: null } },
  ticketRead({ ...tk, status: 'submitted' }, [{ client_order_id: 'c0', broker_order_id: 'o1', order: order() }])];
  const { card } = propose(tk);
  calls.length = 0;
  const confirm = button(card, 'btn-confirm');
  confirm.click();
  // The card re-rendered at once; a second click event aimed at the old node (a double click the
  // browser had already queued) still reaches its listener. The busy guard must drop it.
  assert.notEqual(button(card, 'btn-confirm'), confirm);
  confirm.dispatch('click');
  open();
  await flush();
  assert.equal(confirmPosts().length, 1);
});

await test('confirm 409 requote: old → new price, the old card retires, the new one needs its own Confirm', async () => {
  const tk = ticket();
  const next = ticket({ decisions: [decision(0, {}, { est_price: '612.40', est_cost_usd: '1837.20' })] });
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status: 409, body: { status: 'requote', message: 'Prices moved', ticket: { ...tk, status: 'superseded' }, orders: [], new_ticket: next } }];
  const { turn, card } = propose(tk);
  button(card, 'btn-confirm').click();
  await flush();
  assert.equal(card.getAttribute('data-state'), 'superseded');
  assert.equal(button(card, 'btn-confirm'), null);
  assert.ok(noticeText(card).includes('VOO: $610.00 → $612.40'));
  assert.ok(noticeText(card).includes('nothing was submitted'));
  const cards = turn.slots.extras.querySelectorAll('.ticket');
  assert.equal(cards.length, 2);
  assert.equal(cards[1].getAttribute('data-ticket-id'), next.id);
  assert.equal(cards[1].getAttribute('data-state'), 'proposed');
  assert.equal(button(cards[1], 'btn-confirm').disabled, false);
  assert.equal(calls.filter((c) => c.method === 'POST' && c.path.includes(next.id)).length, 0, 'the new ticket is not confirmed for the user');
  // confirming the new card sends the NEW ticket's id and hash, never the old one's
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${next.id}/confirm`, status: 200, body: {
    status: 'submitted', message: 'ok', ticket: { ...next, status: 'submitted' }, orders: [order()], new_ticket: null } },
  ticketRead({ ...next, status: 'submitted' }, [{ client_order_id: 'c0', broker_order_id: 'o1', order: order() }])];
  calls.length = 0;
  button(cards[1], 'btn-confirm').click();
  await flush();
  assert.deepEqual(confirmPosts().map((c) => [c.path, c.body.hash]), [[`/api/tickets/${next.id}/confirm`, next.hash]]);
  assert.notEqual(next.hash, tk.hash);
  cards[1].remove();
});

await test('confirm 409 rejected, 410, 423, 503, 404: each has its own message and state', async () => {
  const run = async (status, body, error) => {
    const tk = ticket();
    net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status,
      body: { error, message: 'server says no', orders: [], new_ticket: null, ...body(tk) } }];
    const { card } = propose(tk);
    button(card, 'btn-confirm').click();
    await flush();
    return card;
  };
  // 409 rejected (the hash no longer matches): the message is shown, the card stays confirmable
  let changedHash = null;
  let card = await run(409, (tk) => {
    changedHash = 'f'.repeat(64);
    return { status: 'rejected', ticket: { ...tk, hash: changedHash } };
  });
  assert.equal(card.querySelector('.ticket-notice').querySelector('.notice-text').textContent,
    'The ticket changed on the server, so nothing was submitted. Review it and confirm again.');
  assert.equal(serverDetail(card).textContent, 'Details from the serverserver says no');
  assert.equal(serverDetail(card).hasAttribute('open'), false);
  // the next confirm sends the ticket's new hash, as the server now has it
  const id = card.getAttribute('data-ticket-id');
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${id}/confirm`, status: 409,
    body: { status: 'rejected', message: 'again', ticket: null, orders: [], new_ticket: null } }];
  calls.length = 0;
  button(card, 'btn-confirm').click();
  await flush();
  assert.deepEqual(confirmPosts().map((c) => c.body.hash), [changedHash]);
  assert.equal(card.getAttribute('data-state'), 'proposed');
  assert.equal(button(card, 'btn-confirm').disabled, false);
  card.remove();
  // 409 rejected because the ticket was already submitted: the returned ticket decides the state
  card = await run(409, (tk) => ({ status: 'rejected', ticket: { ...tk, status: 'rejected_by_user' } }));
  assert.equal(card.getAttribute('data-state'), 'rejected_by_user');
  assert.ok(noticeText(card).startsWith('This ticket can no longer be confirmed.'));
  assert.equal(button(card, 'btn-confirm'), null);
  // 410: greyed, with Refresh quote
  card = await run(410, (tk) => ({ status: 'expired', ticket: { ...tk, status: 'expired' } }));
  assert.equal(card.getAttribute('data-state'), 'expired');
  assert.deepEqual(card.querySelector('.ticket-actions').querySelectorAll('button').map((b) => b.textContent), ['Refresh quote']);
  // 423: trading is disabled, a link opens Settings, nothing else changes
  card = await run(423, () => ({ status: 'disabled', ticket: null }));
  assert.ok(noticeText(card).includes('Trading is disabled'));
  assert.equal(serverDetail(card), null, 'the localized text says it all');
  const opened = [];
  const stop = on('panel:open', (name) => opened.push(name));
  card.querySelector('.ticket-notice').querySelector('button').click();
  assert.deepEqual(opened, ['settings']);
  assert.equal(card.getAttribute('data-state'), 'proposed');
  card.remove();
  // 503 broker_not_connected: a link to Settings
  card = await run(503, () => ({}), 'broker_not_connected');
  assert.ok(noticeText(card).includes('not connected'));
  card.querySelector('.ticket-notice').querySelector('button').click();
  assert.deepEqual(opened, ['settings', 'settings']);
  stop();
  card.remove();
  // 404
  card = await run(404, () => ({ status: 'not_found', ticket: null }));
  assert.equal(card.getAttribute('data-state'), 'gone');
  assert.equal(card.querySelector('.ticket-actions'), null);
});

await test('confirm 502: "check open orders", the portfolio reloads, and no blind retry is offered', async () => {
  const tk = ticket();
  net.routes = [portfolio,
    { method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status: 502, body: { status: 'error', message: '0 of 1 orders placed, 1 uncertain.', ticket: { ...tk, status: 'submitted' }, orders: [], new_ticket: null } },
    ticketRead({ ...tk, status: 'submitted' }, [{ client_order_id: 'c0', broker_order_id: null, order: null }])];
  const { card } = propose(tk);
  const refreshed = [];
  const stop = on('portfolio:refresh', () => refreshed.push(1));
  button(card, 'btn-confirm').click();
  await flush();
  stop();
  assert.ok(refreshed.length >= 1);
  assert.ok(noticeText(card).startsWith('The confirmation did not finish, and orders may have been placed.'));
  assert.ok(noticeText(card).includes('Check your open orders'));
  assert.equal(serverDetail(card).querySelector('.notice-line').textContent, '0 of 1 orders placed, 1 uncertain.');
  assert.equal(button(card, 'btn-confirm'), null);
  assert.equal(card.getAttribute('data-state'), 'submitted');
  assert.equal(card.querySelector('.timeline-row').textContent.startsWith('Uncertain'), true);
  // no answer at all (the server is unreachable): the outcome is unknown, Confirm is not offered
  const lost = ticket();
  net.routes = [portfolio];
  const second = propose(lost).card;
  button(second, 'btn-confirm').click();
  await flush();
  assert.equal(second.getAttribute('data-state'), 'checking');
  assert.equal(button(second, 'btn-confirm'), null);
  assert.ok(second.querySelector('.ticket-notice').querySelectorAll('button').map((b) => b.textContent).includes('Check again'));
  second.remove();
});

await test('Korean: the confirm result is summarised in Korean; the server text only in the detail row', async () => {
  setLang('ko');
  try {
    const tk = ticket({ decisions: [decision(0), decision(1, { symbol: 'BND', qty: '33', limit_price: '74.01' })] });
    const message = '1 of 2 orders placed, 1 failed. Leg 2 BUY 33 BND @ limit $74.01: error - insufficient buying power.';
    net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status: 200, body: {
      status: 'submitted', message, ticket: { ...tk, status: 'submitted' }, orders: [order({ status: 'accepted', filled_qty: '0' })],
      new_ticket: null } },
    ticketRead({ ...tk, status: 'submitted' }, [{ client_order_id: 'c0', broker_order_id: 'o1', order: order({ status: 'accepted', filled_qty: '0' }) }])];
    const { card } = propose(tk);
    assert.equal(button(card, 'btn-confirm').textContent, '주문 2건 승인');
    button(card, 'btn-confirm').click();
    await flush();
    const summary = card.querySelector('.ticket-notice');
    assert.equal(summary.querySelector('.notice-text').textContent, '주문 2건 중 1건을 모의 계좌로 보냈습니다.');
    assert.equal(summary.querySelector('.notice-order').textContent, '매수▲VOO3주지정가 $612.00 · 당일접수됨');
    assert.ok(summary.textContent.includes('일부 주문은 들어가지 않았습니다'));
    // English appears only inside the collapsed detail row
    const outside = [...summary.querySelectorAll('.notice-text, .notice-order')].map((n) => n.textContent).join(' ');
    assert.ok(!/orders placed|insufficient/.test(outside));
    assert.equal(serverDetail(card).querySelector('.notice-line').textContent, message);
  } finally {
    setLang('en');
  }
});

await test('a ticket the app cannot read is never confirmable under a wrong label', async () => {
  net.routes = [portfolio];
  // a side in capitals is still a buy; an exponent quantity is still a number
  const upper = propose(ticket({ decisions: [decision(0, { side: 'BUY', qty: '1E+1' })] })).card;
  assert.equal(button(upper, 'btn-confirm').textContent, 'Confirm buy 10 VOO');
  assert.equal(button(upper, 'btn-confirm').disabled, false);
  assert.equal(upper.querySelector('.side-badge').textContent, 'BUY▲');
  for (const bad of [{ side: 'short' }, { qty: 'abc' }, { qty: null, notional: null }, { side: undefined }]) {
    const card = propose(ticket({ decisions: [decision(0, bad)] })).card;
    const confirm = button(card, 'btn-confirm');
    assert.equal(confirm.textContent, 'Confirm ticket', JSON.stringify(bad));
    assert.equal(confirm.disabled, true, JSON.stringify(bad));
    assert.ok(card.querySelector('.ticket-caveat').textContent.includes('cannot read'), JSON.stringify(bad));
    calls.length = 0;
    confirm.dispatch('click'); // even a click that reaches the listener sends nothing
    await flush();
    assert.equal(confirmPosts().length, 0, JSON.stringify(bad));
    card.remove();
  }
  // one unreadable leg among good ones makes the whole ticket unconfirmable
  const mixed = propose(ticket({ decisions: [decision(0), decision(1, { side: 'hold' })] })).card;
  assert.equal(button(mixed, 'btn-confirm').textContent, 'Confirm ticket');
  assert.equal(button(mixed, 'btn-confirm').disabled, true);
});

await test('reject, and Modify through the chat', async () => {
  const tk = ticket({ decisions: [decision(0), decision(1, { symbol: 'BND', side: 'sell', qty: '10', limit_price: '73.50' })] });
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/reject`, status: 200, body: { status: 'rejected', message: 'rejected', ticket: { ...tk, status: 'rejected_by_user' }, orders: [], new_ticket: null } }];
  const { card } = propose(tk);
  assert.equal(button(card, 'btn-confirm').textContent, 'Confirm 2 orders');
  const [, modify, reject] = card.querySelector('.ticket-actions').querySelectorAll('button');
  modify.click();
  const inputs = card.querySelector('.ticket-modify').querySelectorAll('input');
  assert.deepEqual(inputs.map((i) => i.value), ['3', '612.00', '10', '73.50']);
  inputs[0].value = '2';
  const sent = [];
  const stop = on('chat:send', (text) => sent.push(text));
  // while a response is streaming the message is not sent, and the fields stay
  setState({ streaming: true });
  card.querySelector('.ticket-modify').querySelector('.btn-primary').click();
  assert.deepEqual(sent, []);
  assert.ok(card.querySelector('.ticket-modify'));
  setState({ streaming: false });
  card.querySelector('.ticket-modify').querySelector('.btn-primary').click();
  stop();
  assert.deepEqual(sent, ['Modify: buy 2 VOO limit 612.00; sell 10 BND limit 73.50']);
  assert.equal(card.querySelector('.ticket-modify'), null);
  assert.equal(calls.filter((c) => c.method === 'POST' && c.path.includes('/confirm') && c.path.includes(tk.id)).length, 0);
  reject.click();
  await flush();
  assert.equal(card.getAttribute('data-state'), 'rejected_by_user');
  assert.equal(card.querySelector('.ticket-actions'), null);
});

await test('a new ticket retires the older ones of the same chat, and only those', async () => {
  net.routes = [portfolio];
  const first = propose(ticket()).card;
  const other = propose(ticket(), 's2').card;
  const second = propose(ticket()).card;
  assert.equal(first.getAttribute('data-state'), 'superseded');
  assert.equal(button(first, 'btn-confirm'), null);
  assert.equal(first.querySelector('.ticket-actions'), null);
  assert.equal(other.getAttribute('data-state'), 'proposed');
  assert.equal(second.getAttribute('data-state'), 'proposed');
});

await test('history: the ticket is read by id and shown in its stored state', async () => {
  const tk = ticket();
  const stored = { ...tk, status: 'submitted', broker_order_ids: ['o1'] };
  net.routes = [portfolio, ticketRead(stored, [{ client_order_id: 'c0', broker_order_id: 'o1', order: order({ status: 'canceled', filled_qty: '0' }) }])];
  const replay = replayTurn();
  dispatch({ type: 'data-trade-proposal', ticket_id: tk.id }, replay);
  assert.equal(replay.slots.extras.textContent, 'Loading the ticket…');
  await flush();
  const card = cardOf(replay);
  assert.equal(card.getAttribute('data-state'), 'submitted');
  assert.equal(button(card, 'btn-confirm'), null);
  assert.ok(card.querySelector('.timeline-status').textContent.includes('Canceled'));
  // a ticket the server no longer has
  net.routes = [portfolio];
  const gone = replayTurn();
  dispatch({ type: 'data-trade-proposal', ticket_id: 'MISSING' }, gone);
  await flush();
  assert.equal(gone.slots.extras.textContent, 'This ticket could not be loaded.');
});

// --- a reloaded chat -------------------------------------------------------------------------------------------

const answerRow = (createdAt) => ({ role: 'assistant', content: 'answer', meta: {}, created_at: createdAt });
const listedOf = (...tickets) => tickets.map((tk) => ({ id: tk.id, status: tk.status, created_at: tk.created_at }));
const iso = (ms) => new Date(clock.time + ms).toISOString();
const byId = (id) => thread.querySelectorAll('[data-ticket-id]').find((node) => node.getAttribute('data-ticket-id') === id);

await test('reload: a requoted, then confirmed ticket comes back with its orders, after its original', async () => {
  thread.replaceChildren();
  const view = document.createElement('section');
  document.body.appendChild(view);
  panel('orders').mount(view);
  setState({ sessionId: 'reload-chat' });
  const original = ticket({ session_id: 'reload-chat', status: 'superseded', created_at: iso(-60000) });
  const requote = ticket({ session_id: 'reload-chat', status: 'submitted', created_at: iso(-30000), broker_order_ids: ['o9'],
    decisions: [decision(0, { symbol: 'QQQ' })] });
  const placed = [{ client_order_id: 'c0', broker_order_id: 'o9', order: order({ id: 'o9', symbol: 'QQQ', status: 'filled' }) }];
  const later = ticket({ session_id: 'reload-chat', status: 'expired', created_at: iso(-20000) });
  net.routes = [portfolio, ticketRead(original), ticketRead(requote, placed), ticketRead(later)];
  // what thread.js draws from the history: two answers and their proposal events (no requote)
  const answer = replayTurn('reload-chat');
  dispatch({ type: 'data-trade-proposal', ticket_id: original.id }, answer);
  const next = replayTurn('reload-chat');
  dispatch({ type: 'data-trade-proposal', ticket_id: later.id }, next);
  const user = (createdAt) => ({ role: 'user', content: 'Buy QQQ', meta: null, created_at: createdAt });
  emit('session:loaded', {
    session: { id: 'reload-chat', title: '' },
    messages: [user(iso(-90000)), answerRow(iso(-59000)), user(iso(-25000)), answerRow(iso(-19000))],
    active_ticket: null, tickets: listedOf(original, requote, later), mandate: {}, mandate_saved: false,
  });
  await flush();
  const cards = answer.slots.extras.querySelectorAll('.ticket');
  assert.deepEqual(cards.map((c) => [c.getAttribute('data-ticket-id'), c.getAttribute('data-state')]),
    [[original.id, 'superseded'], [requote.id, 'submitted']]);
  assert.ok(cards[1].querySelector('.timeline-status').textContent.includes('Filled'));
  assert.deepEqual(next.slots.extras.querySelectorAll('.ticket').map((c) => c.getAttribute('data-ticket-id')), [later.id]);
  // the chat's tickets in the server's order (the history alone would put the requote last)
  assert.deepEqual(ticketIds('reload-chat'), [original.id, requote.id, later.id]);
  // the Orders panel's "From this chat" reads the same list
  panel('orders').show(view);
  await flush();
  assert.ok(view.textContent.includes('QQQ'), 'the requote\'s order is listed');
  // loading the same chat again draws nothing twice
  emit('session:loaded', { session: { id: 'reload-chat' }, messages: [answerRow(iso(-59000)), answerRow(iso(-19000))],
    active_ticket: null, tickets: listedOf(original, requote, later) });
  await flush();
  assert.equal(thread.querySelectorAll('.ticket').length, 3);
  view.remove();
  thread.replaceChildren();
  setState({ sessionId: 's1' });
});

await test('reload: each unlisted ticket goes under the answer written before it; older ones under the first', async () => {
  thread.replaceChildren();
  const first = replayTurn('s1');
  const second = replayTurn('s1');
  const early = ticket({ status: 'expired', created_at: iso(-50000) });   // after the first answer
  const late = ticket({ status: 'expired', created_at: iso(-10000) });    // after the second
  const before = ticket({ status: 'expired', created_at: iso(-99000) });  // older than every answer (its turn was cut)
  const own = ticket({ status: 'superseded', created_at: iso(-61000) });  // the first answer's own proposal
  const active = ticket({ created_at: iso(-5000) });                      // drawn from the body, no read
  net.routes = [portfolio, ticketRead(early), ticketRead(late), ticketRead(before), ticketRead(own)];
  net.calls.length = 0;
  dispatch({ type: 'data-trade-proposal', ticket_id: own.id }, first);
  emit('session:loaded', {
    session: { id: 's1' }, messages: [answerRow(iso(-60000)), answerRow(iso(-20000))],
    active_ticket: active, tickets: listedOf(before, own, early, late, active),
  });
  // the active ticket is drawn from the body at once (no read); the others start as placeholders
  assert.ok(byId(active.id).classList.contains('ticket'));
  assert.ok(byId(early.id).classList.contains('card-note'));
  await flush();
  const ids = (turn) => turn.slots.extras.querySelectorAll('.ticket').map((c) => c.getAttribute('data-ticket-id'));
  // in time order: the cut turn's ticket first, under the first answer, never below the active one
  assert.deepEqual(ids(first), [before.id, own.id, early.id]);
  assert.deepEqual(ids(second), [late.id, active.id]);
  assert.equal(button(byId(active.id), 'btn-confirm').disabled, false);
  // a server without the list: the active ticket alone, under the last answer
  thread.replaceChildren();
  const only = replayTurn('s1');
  const live = ticket();
  emit('session:loaded', { session: { id: 's1' }, messages: [], active_ticket: live });
  assert.deepEqual(ids(only), [live.id]);
  thread.replaceChildren();
});

await test('after the buys: cash only, never margin; a missing figure is said as "Cash after"', async () => {
  const withAccount = (account, openOrders = []) => ({ ...portfolio, body: { ...portfolio.body, account: { ...portfolio.body.account, ...account }, open_orders: openOrders } });
  const after = async (route) => {
    net.routes = [route];
    await refreshPortfolio();
    const { card } = propose(ticket());
    await flush();
    const figure = card.querySelector('.ticket-totals').querySelectorAll('.figure')[1];
    const text = figure.hidden ? null : figure.textContent;
    card.remove();
    return text;
  };
  // a margin account: buying power is twice the cash; the card counts the cash (1,830 of buys)
  assert.equal(await after(withAccount({ cash: '10000.00', buying_power: '20000.00', non_marginable_buying_power: '10000.00' })),
    'Buying power after (est.)$8,170.00');
  // a pending open buy is spoken for
  assert.equal(await after(withAccount({ cash: '10000.00', buying_power: '20000.00', non_marginable_buying_power: '10000.00' },
    [{ side: 'buy', type: 'limit', qty: '2', filled_qty: '0', limit_price: '500.00' }])), 'Buying power after (est.)$7,170.00');
  // the account does not send the non-marginable figure: cash, labelled as such
  assert.equal(await after(withAccount({ cash: '10000.00', buying_power: '20000.00', non_marginable_buying_power: undefined })),
    'Cash after (est.)$8,170.00');
  net.routes = [portfolio];
  await refreshPortfolio();
});

await test('a 200 "submitted" with a leg the broker did not confirm is uncertain, never "placed"', async () => {
  const sent = (n, total) => `${n} of ${total} orders were sent to the paper account.`;
  const answer = (tk, placed, message) => ({ method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status: 200, body: {
    status: 'submitted', message, ticket: { ...tk, status: 'submitted' }, orders: placed, new_ticket: null, reason: null, block_codes: [] } });
  const confirmed = async (tk, placed, rows, message) => {
    net.routes = [portfolio, answer(tk, placed, message), ticketRead({ ...tk, status: 'submitted' }, rows)];
    calls.length = 0;
    const { card } = propose(tk);
    button(card, 'btn-confirm').click();
    await flush();
    return card;
  };
  // every leg uncertain: "0 of 1 orders placed, 1 uncertain"
  const one = ticket();
  let card = await confirmed(one, [], [{ client_order_id: 'c0', broker_order_id: null, order: null }],
    '0 of 1 orders placed, 1 uncertain. Leg 1 BUY 3 VOO @ limit $612.00: uncertain - not found at lookup time; do not resubmit.');
  assert.equal(noticeText(card).startsWith(dict.en.ticket_result_uncertain), true);
  assert.ok(noticeText(card).includes(dict.en.ticket_check_orders));
  assert.ok(!noticeText(card).includes('were sent'), 'never "placed"');
  assert.equal(card.querySelector('.ticket-notice').getAttribute('data-tone'), 'bad');
  assert.deepEqual(card.querySelector('.ticket-notice').querySelectorAll('button').map((b) => b.textContent), ['Open Orders', 'Check again']);
  assert.equal(button(card, 'btn-confirm'), null, 'nothing to confirm again');
  assert.ok(calls.some((c) => c.method === 'GET' && c.path.includes(one.id)), 'the ticket is read again');
  assert.ok(card.querySelector('.timeline-row.is-uncertain'), 'the timeline shows the uncertain leg');
  card.remove();
  // one placed, one uncertain
  const two = ticket({ decisions: [decision(0), decision(1, { symbol: 'VTI' })] });
  card = await confirmed(two, [order()], [{ client_order_id: 'c0', broker_order_id: 'o1', order: order() },
    { client_order_id: 'c1', broker_order_id: null, order: null }], '1 of 2 orders placed, 1 uncertain.');
  assert.equal(noticeText(card).startsWith(dict.en.ticket_result_uncertain), true);
  assert.ok(!noticeText(card).includes(sent(1, 2)));
  card.remove();
  // one placed, one failed outright (no row): after the re-read, the ordinary partial result
  const failed = ticket({ decisions: [decision(0), decision(1, { symbol: 'VTI' })] });
  card = await confirmed(failed, [order()], [{ client_order_id: 'c0', broker_order_id: 'o1', order: order() }],
    '1 of 2 orders placed, 1 failed. Leg 2 BUY 3 VTI @ limit $612.00: error - insufficient buying power.');
  assert.ok(noticeText(card).startsWith(sent(1, 2)), noticeText(card));
  assert.ok(noticeText(card).includes(dict.en.ticket_result_partial));
  card.remove();
  // every leg placed: the plain success
  const all = ticket();
  card = await confirmed(all, [order()], [{ client_order_id: 'c0', broker_order_id: 'o1', order: order() }], '1 of 1 orders placed.');
  assert.ok(noticeText(card).startsWith(sent(1, 1)));
  card.remove();
});

await test('while any row is uncertain the header says "Outcome unknown" and the footer never says every order is final', async () => {
  const badgeOf = (card) => card.querySelector('.card-head').querySelector('.badge');
  const footOf = (card) => card.querySelector('.timeline-foot');
  const checkAgain = (card) => card.querySelector('.ticket-notice').querySelectorAll('button').find((b) => b.textContent === 'Check again');
  const row = (n, placed) => ({ client_order_id: `c${n}`, broker_order_id: placed ? `o${n}` : null, order: placed ? order({ id: `o${n}`, client_order_id: `c${n}` }) : null });
  const confirmAnswer = (tk, status, orders) => ({ method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status, body: {
    status: status === 200 ? 'submitted' : 'error', message: '1 uncertain.', ticket: { ...tk, status: 'submitted' }, orders, new_ticket: null,
    reason: null, block_codes: [] } });
  const two = () => ticket({ decisions: [decision(0), decision(1, { symbol: 'VTI' })] });
  const flows = [
    { name: '200, no order listed for the one leg', tk: ticket(), status: 200, listed: [], rows: [row(0, false)], settled: [row(0, true)] },
    { name: '200, one order listed for two legs', tk: two(), status: 200, listed: [order()], rows: [row(0, true), row(1, false)], settled: [row(0, true), row(1, true)] },
    { name: '502 with an uncertain row', tk: ticket(), status: 502, listed: [], rows: [row(0, false)], settled: [row(0, true)] },
  ];
  for (const { name, tk, status, listed, rows, settled } of flows) {
    net.routes = [portfolio, confirmAnswer(tk, status, listed), ticketRead({ ...tk, status: 'submitted' }, rows)];
    const { card } = propose(tk);
    button(card, 'btn-confirm').click();
    await flush();
    assert.equal(badgeOf(card).textContent, dict.en.ticket_state_checking, name);
    assert.equal(badgeOf(card).getAttribute('data-tone'), 'warn', name);
    assert.equal(footOf(card).textContent, dict.en.timeline_unsettled, name);
    assert.ok(!/final/i.test(footOf(card).textContent), name);
    assert.ok(card.querySelector('.timeline-row.is-uncertain'), name);
    setLang('ko');
    try {
      assert.equal(badgeOf(card).textContent, dict.ko.ticket_state_checking, name);
      assert.equal(footOf(card).textContent, dict.ko.timeline_unsettled, name);
    } finally {
      setLang('en');
    }
    // a re-read that is still uncertain changes nothing
    checkAgain(card).click();
    await flush();
    assert.equal(badgeOf(card).textContent, dict.en.ticket_state_checking, name);
    assert.equal(footOf(card).textContent, dict.en.timeline_unsettled, name);
    // a re-read that settles every row brings the ordinary display back
    net.routes = [portfolio, ticketRead({ ...tk, status: 'submitted' }, settled)];
    checkAgain(card).click();
    await flush();
    assert.equal(badgeOf(card).textContent, dict.en.ticket_state_submitted, name);
    assert.equal(badgeOf(card).getAttribute('data-tone'), 'ok', name);
    assert.equal(footOf(card).textContent, dict.en.timeline_settled, name);
    assert.equal(card.querySelectorAll('.timeline-row.is-uncertain').length, 0, name);
    card.remove();
  }
  // one leg placed, one failed outright: unsure until the first re-read answers, then the ordinary display
  const half = two();
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  net.routes = [portfolio, confirmAnswer(half, 200, [order()]), { ...ticketRead({ ...half, status: 'submitted' }, [row(0, true)]), gate }];
  const { card } = propose(half);
  button(card, 'btn-confirm').click();
  await flush();
  assert.equal(badgeOf(card).textContent, dict.en.ticket_state_checking, 'unsure while the re-read is outstanding');
  release();
  await flush();
  assert.equal(badgeOf(card).textContent, dict.en.ticket_state_submitted);
  assert.equal(footOf(card).textContent, dict.en.timeline_settled);
  card.remove();
});

await test('an uncertain answer that later settles is announced once, when it settles', async () => {
  const lost = { client_order_id: 'c1', broker_order_id: null, order: null };
  const placed = (n) => ({ client_order_id: `c${n}`, broker_order_id: `o${n}`, order: order({ id: `o${n}`, client_order_id: `c${n}` }) });
  const answerOf = (tk, status) => ({ method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status, body: {
    status: status === 200 ? 'submitted' : 'error', message: '1 of 2 orders placed.', ticket: { ...tk, status: 'submitted' },
    orders: status === 200 ? [order()] : [], new_ticket: null, reason: null, block_codes: [] } });
  const checkAgain = (card) => card.querySelector('.ticket-notice').querySelectorAll('button').find((b) => b.textContent === 'Check again');
  const confirmed = async (tk, status, rows) => {
    net.routes = [portfolio, answerOf(tk, status), ticketRead({ ...tk, status: 'submitted' }, rows)];
    await quietAnnouncer();
    const { card } = propose(tk);
    button(card, 'btn-confirm').click();
    await flush();
    return card;
  };
  // one leg placed, one failed: the first answer is uncertain ("did not finish"), the re-read settles it
  const half = ticket({ decisions: [decision(0), decision(1, { symbol: 'VTI' })] });
  let card = await confirmed(half, 200, [placed(0), lost]);
  assert.deepEqual(await announced(), { calls: 1, text: dict.en.ticket_announce_uncertain });
  // still uncertain on the next look: nothing new to say
  checkAgain(card).click();
  await flush();
  assert.deepEqual(await announced(), { calls: 1, text: dict.en.ticket_announce_uncertain });
  // the second leg turns out to have failed: the ordinary partial result is announced, once, even when two
  // re-reads are in flight together
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  net.routes = [portfolio, { ...ticketRead({ ...half, status: 'submitted' }, [placed(0)]), gate }];
  checkAgain(card).click();
  checkAgain(card).click();
  await flush();
  release();
  await flush();
  assert.ok(noticeText(card).startsWith('1 of 2 orders were sent'), noticeText(card));
  const partial = await announced();
  assert.equal(partial.text, '1 of 2 orders were submitted. The rest were not placed.');
  assert.equal(partial.calls, 2, 'the uncertain answer, then the settled one: no repeat');
  assert.ok(!partial.text.includes('Orders submitted'), 'a partial result is not announced as plain success');
  card.remove();
  // the missing leg was placed after all: the plain success is announced
  const both = ticket({ decisions: [decision(0), decision(1, { symbol: 'VTI' })] });
  card = await confirmed(both, 200, [placed(0), lost]);
  net.routes = [portfolio, ticketRead({ ...both, status: 'submitted' }, [placed(0), placed(1)])];
  checkAgain(card).click();
  await flush();
  assert.deepEqual(await announced(), { calls: 2, text: dict.en.ticket_announce_submitted });
  card.remove();
  // the Korean wording takes the same numbers
  setLang('ko');
  try {
    assert.equal(t('ticket_announce_partial', { placed: 1, total: 2 }), '주문 2건 중 1건을 전송했습니다. 나머지는 들어가지 않았습니다.');
  } finally {
    setLang('en');
  }
});

await test('a 502 that later settles: the "did not finish" notice gives way to the ordinary result, announced once', async () => {
  const lost = { client_order_id: 'c1', broker_order_id: null, order: null };
  const placed = (n) => ({ client_order_id: `c${n}`, broker_order_id: `o${n}`, order: order({ id: `o${n}`, client_order_id: `c${n}` }) });
  const badAnswer = (tk) => ({ method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status: 502, body: {
    status: 'error', message: '1 of 2 orders placed, 1 uncertain.', ticket: { ...tk, status: 'submitted' }, orders: [], new_ticket: null,
    reason: null, block_codes: [] } });
  const checkAgain = (card) => card.querySelector('.ticket-notice').querySelectorAll('button').find((b) => b.textContent === 'Check again');
  const badgeOf = (card) => card.querySelector('.card-head').querySelector('.badge');
  const two = () => ticket({ decisions: [decision(0), decision(1, { symbol: 'VTI' })] });
  const confirmed = async (tk, rows) => {
    net.routes = [portfolio, badAnswer(tk), ticketRead({ ...tk, status: 'submitted' }, rows)];
    await quietAnnouncer();
    const { card } = propose(tk);
    button(card, 'btn-confirm').click();
    await flush();
    return card;
  };
  // one leg placed, the other failed: the 502 is uncertain until a re-read shows how it ended
  const half = two();
  let card = await confirmed(half, [placed(0), lost]);
  assert.ok(noticeText(card).startsWith(dict.en.ticket_uncertain), noticeText(card));
  assert.deepEqual(await announced(), { calls: 1, text: dict.en.ticket_announce_uncertain });
  // still uncertain on the next look: the notice stays, nothing new is announced
  checkAgain(card).click();
  await flush();
  assert.ok(noticeText(card).startsWith(dict.en.ticket_uncertain), noticeText(card));
  assert.equal(badgeOf(card).textContent, dict.en.ticket_state_checking);
  assert.deepEqual(await announced(), { calls: 1, text: dict.en.ticket_announce_uncertain });
  // the second leg turns out to have failed; two re-reads are in flight together
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  net.routes = [portfolio, { ...ticketRead({ ...half, status: 'submitted' }, [placed(0)]), gate }];
  checkAgain(card).click();
  checkAgain(card).click();
  await flush();
  release();
  await flush();
  assert.ok(noticeText(card).startsWith('1 of 2 orders were sent'), noticeText(card));
  assert.ok(noticeText(card).includes(dict.en.ticket_result_partial));
  assert.ok(!noticeText(card).includes(dict.en.ticket_uncertain), 'the stale "did not finish" text is gone');
  assert.equal(checkAgain(card), undefined, 'nothing left to check');
  assert.equal(card.querySelector('.ticket-notice').getAttribute('data-tone'), 'warn');
  assert.equal(badgeOf(card).textContent, dict.en.ticket_state_submitted);
  const partial = await announced();
  assert.equal(partial.text, '1 of 2 orders were submitted. The rest were not placed.');
  assert.equal(partial.calls, 2, 'the uncertain answer, then the settled one, once');
  setLang('ko');
  try {
    assert.ok(noticeText(card).startsWith(t('ticket_result_placed', { placed: 1, total: 2 })), noticeText(card));
  } finally {
    setLang('en');
  }
  card.remove();
  // every leg was placed after all: the plain result, with the orders listed, and the plain success announced
  const all = ticket();
  card = await confirmed(all, [lost]);
  assert.ok(noticeText(card).startsWith(dict.en.ticket_uncertain), noticeText(card));
  net.routes = [portfolio, ticketRead({ ...all, status: 'submitted' }, [placed(0)])];
  checkAgain(card).click();
  await flush();
  assert.ok(noticeText(card).startsWith('1 of 1 orders were sent'), noticeText(card));
  assert.ok(!noticeText(card).includes(dict.en.ticket_result_partial));
  assert.equal(card.querySelector('.ticket-notice').getAttribute('data-tone'), 'ok');
  assert.equal(card.querySelectorAll('.notice-order').length, 1);
  assert.deepEqual(await announced(), { calls: 2, text: dict.en.ticket_announce_submitted });
  card.remove();
  // a re-read that finds the ticket still proposed leaves nothing unconfirmed: a later clean 200 is plain "Submitted"
  const again = ticket();
  net.routes = [portfolio, badAnswer(again), ticketRead(again)];
  ({ card } = propose(again));
  button(card, 'btn-confirm').click();
  await flush();
  assert.equal(noticeText(card), dict.en.ticket_uncertain_none);
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${again.id}/confirm`, status: 200, body: {
    status: 'submitted', message: '1 of 1 orders placed.', ticket: { ...again, status: 'submitted' }, orders: [order()],
    new_ticket: null, reason: null, block_codes: [] } }, ticketRead({ ...again, status: 'submitted' }, [placed(0)])];
  button(card, 'btn-confirm').click();
  await flush();
  assert.ok(noticeText(card).startsWith('1 of 1 orders were sent'), noticeText(card));
  assert.equal(badgeOf(card).textContent, dict.en.ticket_state_submitted);
  assert.equal(card.querySelector('.timeline-foot').textContent, dict.en.timeline_settled);
  card.remove();
});

await test('uncertain confirm: a re-read that finds the ticket still proposed says nothing was placed', async () => {
  const tk = ticket();
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/confirm`, status: 504, body: { error: 'http_504' } },
    ticketRead(tk)];
  const { card } = propose(tk);
  button(card, 'btn-confirm').click();
  await flush();
  assert.equal(card.getAttribute('data-state'), 'proposed', 'the re-read found it still proposed');
  assert.equal(noticeText(card), 'The server shows nothing was placed for this ticket.');
  assert.ok(!noticeText(card).includes('may have been placed'));
  assert.equal(button(card, 'btn-confirm').disabled, false);
  setLang('ko');
  assert.equal(noticeText(card), '서버 기록상 이 티켓으로는 주문이 들어가지 않았습니다.');
  setLang('en');
  card.remove();
});

finish('ticket card checks');
