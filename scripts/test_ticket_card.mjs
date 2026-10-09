// Tests for the trade ticket card (web/app/js/components/ticket.js) on the fake DOM.
// Run: node scripts/test_ticket_card.mjs
//
// The real card is driven through every answer the ticket endpoints can give (confirm 200, 409
// requote, 409 rejected, 410, 423, 503, 502, 404, and no answer at all), with `fetch` replaced by a
// scripted stand-in: what the Confirm button says, that it sends the hash exactly once, which state
// the card ends in, and that no state offers a blind retry. The setup is scripts/component_harness.mjs.
import assert from 'node:assert/strict';
import {
  advance, clock, decision, dispatch, document, finish, flush, leg, liveTurn, net, on, order, replayTurn, setState,
  sources, test, ticker, ticket,
} from './component_harness.mjs';

const { calls } = net;
const cardOf = (turn) => turn.slots.extras.querySelector('.ticket');
const button = (card, cls) => card.querySelector(`.${cls}`);
const noticeText = (card) => card.querySelector('.ticket-notice').textContent;
// E1 is a known source of chat s1; the tickets below also cite E9, which is not.
sources.seedSources('s1', [{ id: 'E1', title: 'Alpaca IEX quotes', publisher: 'Alpaca Markets', tier: 1 }]);

// --- the trade ticket card ------------------------------------------------------------------------------------

function propose(tk, sessionId = 's1') {
  const turn = liveTurn(sessionId);
  dispatch({ type: 'data-trade-proposal', ticket: tk }, turn);
  return { turn, card: cardOf(turn) };
}
const ticketRead = (tk, orders = []) => ({ method: 'GET', match: `/api/tickets/${tk.id}?`, status: 200, body: { ticket: tk, orders } });
const portfolio = { method: 'GET', match: '/api/portfolio', status: 200, body: { connected: true, paper: true,
  account: { equity: '132100.00', cash: '100000.00', buying_power: '100000.00', last_equity: '131000.00', paper: true },
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
  // An expired ticket cannot be re-quoted by the server (410): the card offers a new proposal.
  net.routes = [{ method: 'POST', match: `/api/tickets/${tk.id}/refresh`, status: 410, body: { status: 'rejected', message: 'ticket is expired', ticket: { ...tk, status: 'expired' }, orders: [], new_ticket: null } }];
  actions[0].click();
  await flush();
  assert.equal(card.getAttribute('data-state'), 'expired');
  assert.ok(noticeText(card).includes('cannot be re-quoted'));
  const sent = [];
  const stop = on('chat:send', (text) => sent.push(text));
  card.querySelector('.ticket-notice').querySelector('button').click();
  stop();
  assert.deepEqual(sent, ['Modify: buy 3 VOO limit 612.00']);
  clock.time -= 288000; // back to "now" for the following tests
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
  assert.equal(noticeText(card), '1 of 1 orders placed.');
  const row = card.querySelector('.timeline-row');
  assert.equal(row.querySelector('.timeline-main').textContent, 'BUY▲VOO3 shares');
  assert.ok(row.querySelector('.timeline-status').textContent.includes('Filled'));
  assert.ok(row.querySelector('.timeline-status').textContent.includes('3 of 3 filled at $610.10'));
  assert.equal(card.querySelector('.timeline-foot').textContent, 'Every order is final.');
});

await test('confirm 409 requote: old → new price, the old card retires, the new one needs its own Confirm', async () => {
  const tk = ticket();
  const next = ticket({ decisions: [decision(0, {}, { est_price: '612.40', est_cost_usd: '1837.20' })], hash: 'b'.repeat(64) });
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
  let card = await run(409, (tk) => ({ status: 'rejected', ticket: tk }));
  assert.equal(noticeText(card), 'server says no');
  assert.equal(card.getAttribute('data-state'), 'proposed');
  assert.equal(button(card, 'btn-confirm').disabled, false);
  card.remove();
  // 409 rejected because the ticket was already submitted: the returned ticket decides the state
  card = await run(409, (tk) => ({ status: 'rejected', ticket: { ...tk, status: 'rejected_by_user' } }));
  assert.equal(card.getAttribute('data-state'), 'rejected_by_user');
  assert.equal(button(card, 'btn-confirm'), null);
  // 410: greyed, with Refresh quote
  card = await run(410, (tk) => ({ status: 'expired', ticket: { ...tk, status: 'expired' } }));
  assert.equal(card.getAttribute('data-state'), 'expired');
  assert.deepEqual(card.querySelector('.ticket-actions').querySelectorAll('button').map((b) => b.textContent), ['Refresh quote']);
  // 423: trading is disabled, a link opens Settings, nothing else changes
  card = await run(423, () => ({ status: 'disabled', ticket: null }));
  assert.ok(noticeText(card).includes('Trading is disabled'));
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
  assert.ok(noticeText(card).includes('0 of 1 orders placed, 1 uncertain.'));
  assert.ok(noticeText(card).includes('Check your open orders'));
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

finish('ticket card checks');
