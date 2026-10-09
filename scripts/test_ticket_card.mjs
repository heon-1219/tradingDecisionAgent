// Tests for the trade ticket card (web/app/js/components/ticket.js) on the fake DOM.
// Run: node scripts/test_ticket_card.mjs
//
// The real card is driven through every answer the ticket endpoints can give (confirm 200, 409
// requote, 409 rejected, 410, 423, 503, 502, 404, and no answer at all), with `fetch` replaced by a
// scripted stand-in: what the Confirm button says, that it sends the hash exactly once, which state
// the card ends in, and that no state offers a blind retry. The setup is scripts/component_harness.mjs.
import assert from 'node:assert/strict';
import {
  advance, clock, decision, dispatch, document, emit, finish, flush, leg, liveTurn, net, on, order, panel, replayTurn,
  setState, sources, test, thread, ticker, ticket,
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

await test('an expired ticket that is not the newest: 409, no second Refresh, "Ask for a new ticket"', async () => {
  net.routes = [portfolio];
  const tk = ticket({ status: 'expired' });
  const { card } = propose(tk);
  assert.equal(card.getAttribute('data-state'), 'expired');
  net.routes = [portfolio, { method: 'POST', match: `/api/tickets/${tk.id}/refresh`, status: 409, body: { status: 'rejected',
    message: 'ticket is expired and a newer ticket exists; nothing was re-quoted', ticket: tk, orders: [], new_ticket: null } }];
  button(card, 'ticket-actions').querySelector('button').click();
  await flush();
  assert.equal(card.getAttribute('data-state'), 'expired');
  assert.ok(noticeText(card).includes('cannot be re-quoted: this chat has a newer ticket'));
  assert.equal(card.querySelector('.ticket-actions'), null);
  const sent = [];
  const stop = on('chat:send', (text) => sent.push(text));
  card.querySelector('.ticket-notice').querySelector('button').click();
  stop();
  assert.deepEqual(sent, ['Modify: buy 3 VOO limit 612.00']);
  // a 409 on refresh for any other reason is the ordinary conflict, without the offer
  const other = ticket({ status: 'superseded' });
  const second = propose(other).card;
  net.routes = [portfolio];
  assert.equal(second.querySelector('.ticket-actions'), null);
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

await test('reload: each unlisted ticket goes under the answer written before it, else the last answer', async () => {
  thread.replaceChildren();
  const first = replayTurn('s1');
  const second = replayTurn('s1');
  const early = ticket({ status: 'expired', created_at: iso(-50000) });   // after the first answer
  const late = ticket({ status: 'expired', created_at: iso(-10000) });    // after the second
  const before = ticket({ status: 'expired', created_at: iso(-99000) });  // older than every answer
  const active = ticket({ created_at: iso(-5000) });                      // drawn from the body, no read
  net.routes = [portfolio, ticketRead(early), ticketRead(late), ticketRead(before)];
  net.calls.length = 0;
  emit('session:loaded', {
    session: { id: 's1' }, messages: [answerRow(iso(-60000)), answerRow(iso(-20000))],
    active_ticket: active, tickets: listedOf(before, early, late, active),
  });
  // the active ticket is drawn from the body at once (no read); the others start as placeholders
  assert.ok(byId(active.id).classList.contains('ticket'));
  assert.ok(byId(early.id).classList.contains('card-note'));
  await flush();
  const ids = (turn) => turn.slots.extras.querySelectorAll('.ticket').map((c) => c.getAttribute('data-ticket-id'));
  assert.deepEqual(ids(first), [early.id]);
  assert.deepEqual(ids(second), [before.id, late.id, active.id]);
  assert.equal(button(byId(active.id), 'btn-confirm').disabled, false);
  // a server without the list: the active ticket alone, under the last answer
  thread.replaceChildren();
  const only = replayTurn('s1');
  const live = ticket();
  emit('session:loaded', { session: { id: 's1' }, messages: [], active_ticket: live });
  assert.deepEqual(ids(only), [live.id]);
  thread.replaceChildren();
});

finish('ticket card checks');
