// Tests for the conversation components (web/app/js/components). Run: node scripts/test_components.mjs
//
// Pure logic first (the ticket model, tool-output splitting, the mandate reader, the poller, the
// shared ticker, popover placement), then the real components on the fake DOM: tool cards and the
// Thinking block through the registry, grounding, citation chips, the Portfolio panel and the
// mandate form. The trade ticket card has its own file, scripts/test_ticket_card.mjs. The setup
// (fake DOM, scripted fetch, manual clock, fixtures) is scripts/component_harness.mjs.
import assert from 'node:assert/strict';
import {
  advance, changeParts, clock, decision, dict, dispatch, document, emit, finish, leg, liveTurn, mandateForm, model,
  net, order, orderView, panel, placePopover, refreshPortfolio, replayTurn, setState, settingsErrorText, sources, t,
  test, ticker, ticket, timeline, toolcard,
} from './component_harness.mjs';

// --- pure: the ticket model ------------------------------------------------------------------------------

await test('confirm label: one leg is spelled out, several are counted, and every key exists', () => {
  const one = model.confirmLabel(ticket());
  assert.deepEqual(one, { key: 'ticket_confirm_buy_qty', vars: { qty: '3', symbol: 'VOO' } });
  assert.equal(t(one.key, one.vars), 'Confirm buy 3 VOO');
  const sell = model.confirmLabel(ticket({ decisions: [decision(0, { side: 'sell', symbol: 'BND', qty: '1200.50' })] }));
  assert.equal(t(sell.key, sell.vars), 'Confirm sell 1,200.5 BND');
  const notional = model.confirmLabel(ticket({ decisions: [decision(0, { qty: null, notional: '500', type: 'market', limit_price: null })] }));
  assert.equal(t(notional.key, notional.vars), 'Confirm buy $500.00 of VOO');
  const many = model.confirmLabel(ticket({ decisions: [decision(0), decision(1, { symbol: 'BND' })] }));
  assert.equal(t(many.key, many.vars), 'Confirm 2 orders');
  // A rejected leg is not counted: `legs` holds only what can be placed.
  const mixed = ticket({ decisions: [decision(0), decision(1, { symbol: 'XYZ' }, { verdict: 'reject' })] });
  assert.equal(t(model.confirmLabel(mixed).key, model.confirmLabel(mixed).vars), 'Confirm buy 3 VOO');
  assert.equal(model.confirmLabel({ legs: [] }).key, 'ticket_confirm_none');
  for (const side of ['buy', 'sell']) {
    for (const kind of ['qty', 'notional']) assert.ok(dict.en[`ticket_confirm_${side}_${kind}`] && dict.ko[`ticket_confirm_${side}_${kind}`]);
  }
  for (const option of ['limit', 'market']) assert.ok(dict.en[`order_type_${option}`] && dict.ko[`order_type_${option}`]);
});

await test('totals, buying power and position share use exact decimals', () => {
  const tk = ticket({ decisions: [
    decision(0, {}, { est_cost_usd: '0.10' }), decision(1, { symbol: 'VTI' }, { est_cost_usd: '0.20' }),
    decision(2, { symbol: 'BND', side: 'sell' }, { est_cost_usd: '740.00' }),
    decision(3, { symbol: 'XYZ' }, { verdict: 'reject', est_cost_usd: '999.00' }),
  ] });
  const totals = model.ticketTotals(tk);
  assert.equal(totals.buys, 2);
  assert.equal(totals.sells, 1);
  assert.equal(totals.complete, true);
  assert.deepEqual(totals.buy, { int: 30n, scale: 2 }); // 0.10 + 0.20 is exactly 0.30
  assert.deepEqual(model.buyingPowerAfter('100000.00', totals), { int: 9999970n, scale: 2 });
  assert.equal(model.buyingPowerAfter(undefined, totals), null);
  const partial = model.ticketTotals(ticket({ decisions: [decision(0, {}, { est_cost_usd: null })] }));
  assert.equal(partial.complete, false);
  // 12,000 held + 1,830 bought, of 132,100 equity = 10.469%
  const share = model.positionPercentAfter(decision(0), [{ symbol: 'VOO', market_value: '12000.00' }], '132100.00');
  assert.deepEqual(share, { int: 10469300n, scale: 6 }); // 10.469300 (percent)
  const sold = model.positionPercentAfter(decision(0, { side: 'sell' }, { est_cost_usd: '99999' }), [{ symbol: 'VOO', market_value: '12000' }], '132100');
  assert.equal(sold.int, 0n); // never negative
  assert.equal(model.positionPercentAfter(decision(0), [], '0'), null);
  assert.equal(model.positionPercentAfter(decision(0), [], undefined), null);
});

await test('clamp and requote changes', () => {
  const clamped = decision(0, { qty: '6' }, { verdict: 'clamp', original: leg({ qty: '10' }) });
  assert.deepEqual(model.clampChanges(clamped), [{ field: 'qty', from: '10', to: '6' }]);
  assert.deepEqual(model.clampChanges(decision(0)), []);
  const old = ticket();
  const next = ticket({ decisions: [decision(0, {}, { est_price: '612.40' })] });
  assert.deepEqual(model.requoteChanges(old, next), [{ symbol: 'VOO', side: 'buy', from: '610.00', to: '612.40' }]);
  assert.deepEqual(model.requoteChanges(old, { decisions: [] }), [{ symbol: 'VOO', side: 'buy', from: '610.00', to: null }]);
});

await test('ticket state follows the stored status and the clock', () => {
  const tk = ticket();
  assert.equal(model.ticketState(tk, clock.time), 'proposed');
  assert.equal(model.ticketState(tk, Date.parse(tk.expires_at)), 'expired');
  assert.equal(model.ticketState({ ...tk, status: 'submitted' }, Date.parse(tk.expires_at) + 1), 'submitted');
  assert.equal(model.ticketState({ ...tk, status: 'superseded' }, clock.time), 'superseded');
});

await test('every answer of the ticket endpoints maps to one outcome', () => {
  const body = (over) => ({ status: 'x', message: 'm', ticket: null, orders: [], new_ticket: null, ...over });
  const kind = (action, status, b = {}, error = '') => model.interpretResult(action, { ok: status >= 200 && status < 300, status, error, body: body(b) }).kind;
  assert.equal(kind('confirm', 200, { status: 'submitted' }), 'submitted');
  assert.equal(kind('confirm', 409, { status: 'requote', new_ticket: { id: 'N' } }), 'requote');
  assert.equal(kind('confirm', 409, { status: 'requote' }), 'rejected'); // a requote without a ticket confirms nothing
  assert.equal(kind('confirm', 409, { status: 'rejected' }), 'rejected');
  assert.equal(kind('confirm', 410, { status: 'expired' }), 'expired');
  assert.equal(kind('confirm', 423, { status: 'disabled' }), 'disabled');
  assert.equal(kind('confirm', 503, {}, 'broker_not_connected'), 'no_broker');
  assert.equal(kind('confirm', 503, {}, 'chat_unavailable'), 'failed');
  assert.equal(kind('confirm', 502, { status: 'error' }), 'uncertain');
  assert.equal(kind('confirm', 500, {}, 'confirm_failed'), 'uncertain');
  assert.equal(kind('confirm', 0, {}, 'network_error'), 'uncertain');
  assert.equal(kind('confirm', 404, { status: 'not_found' }), 'not_found');
  assert.equal(kind('reject', 200, { status: 'rejected' }), 'dismissed');
  assert.equal(kind('reject', 410), 'expired');
  assert.equal(kind('reject', 500, {}, 'reject_failed'), 'failed');
  assert.equal(kind('refresh', 200, { status: 'requote', new_ticket: { id: 'N' } }), 'requote');
  assert.equal(kind('refresh', 410, { status: 'rejected' }), 'expired');
  assert.equal(kind('refresh', 502, {}, 'refresh_failed'), 'failed');
  assert.equal(kind('refresh', 0, {}, 'network_error'), 'failed');
});

await test('the Modify message and its field checks', () => {
  const legs = [leg(), leg({ symbol: 'BND', side: 'sell', qty: '10.000', limit_price: '73.50' }),
    leg({ symbol: 'QQQ', qty: null, notional: '500', type: 'market', limit_price: null })];
  const { edits, errors } = model.readModify(legs, [{ qty: ' 2 ', limit_price: '1,612.00' }, { qty: '10', limit_price: '73.5' }, { notional: '250' }]);
  assert.deepEqual(errors, {});
  const message = model.modifyMessage(edits);
  assert.equal(message, 'Modify: buy 2 VOO limit 1612.00; sell 10 BND limit 73.5; buy $250 of QQQ at market');
  assert.ok(model.modifyMessage(edits, 'ko').startsWith('Modify: buy 2 VOO'));
  assert.ok(/[가-힣]/.test(model.modifyMessage(edits, 'ko')));
  // the shape the scripted demo model (scripts/_demo_llm.py) and a real model both read
  const demo = /\b(buy|sell)\s+(\d+(?:\.\d+)?)\s+(?:shares?\s+(?:of\s+)?)?([A-Za-z][A-Za-z0-9.-]{0,9})(?:[^;,\n]*?(?:limit|@|at)\s*\$?(\d+(?:\.\d+)?))?/gi;
  assert.deepEqual([...message.matchAll(demo)].map((m) => m.slice(1, 5)),
    [['buy', '2', 'VOO', '1612.00'], ['sell', '10', 'BND', '73.5']]);
  const bad = model.readModify(legs, [{ qty: '0', limit_price: 'abc' }, { qty: '-1', limit_price: '' }, { notional: '1e3' }]);
  assert.deepEqual(Object.keys(bad.errors).sort(), ['0.limit_price', '0.qty', '1.limit_price', '1.qty', '2.notional']);
  // the expired-ticket fallback sends the legs as they are
  assert.equal(model.modifyMessage([legs[0]]), 'Modify: buy 3 VOO limit 612.00');
});

// --- pure: tool cards, mandate, orders, popover -----------------------------------------------------------

await test('tool families, untrusted blocks and durations', () => {
  assert.equal(toolcard.familyOf('get_quotes'), 'market');
  assert.equal(toolcard.familyOf('propose_trades'), 'trading');
  assert.equal(toolcard.familyOf('draft_mandate'), 'mandate');
  assert.equal(toolcard.familyOf('unknown_tool'), 'other');
  assert.equal(toolcard.familyOf('constructor'), 'other');
  for (const spec of Object.values(toolcard.FAMILIES)) assert.ok(dict.en[spec.key] && dict.ko[spec.key], spec.key);
  const parts = toolcard.splitUntrusted('[E3] <untrusted source="E3">Title — <b>body</b></untrusted>\nSources:\n<untrusted source="E4,E5">two</untrusted>');
  assert.deepEqual(parts, [
    { kind: 'text', text: '[E3] ' }, { kind: 'external', source: 'E3', text: 'Title — <b>body</b>' },
    { kind: 'text', text: '\nSources:\n' }, { kind: 'external', source: 'E4,E5', text: 'two' },
  ]);
  assert.deepEqual(toolcard.splitUntrusted('<untrusted source="E1">never closed'), [{ kind: 'text', text: '<untrusted source="E1">never closed' }]);
  assert.deepEqual(toolcard.splitUntrusted(undefined), []);
  assert.equal(toolcard.durationText(420), '0.4 s');
  assert.equal(toolcard.durationText(12300), '12 s');
});

await test('mandate reader: decimals stay strings, symbols are upper-cased, shapes are checked', () => {
  const raw = { direction_text: ' Core ETFs ', budget_usd: '5,000', max_order_usd: '', max_position_pct: '20',
    cash_buffer_pct: '0', price_band_pct: '2.5', max_positions: '8', max_orders_per_day: '10', horizon: 'years',
    allow_symbols: 'voo, vti  bnd;voo', deny_symbols: '', order_types: ['market', 'limit'], allow_fractional: true };
  const { payload, errors } = mandateForm.readMandate(raw, { universe: 'us_equity', version: 4 });
  assert.deepEqual(errors, {});
  assert.deepEqual(payload, { universe: 'us_equity', direction_text: 'Core ETFs', budget_usd: '5000', max_order_usd: null,
    max_position_pct: '20', cash_buffer_pct: '0', price_band_pct: '2.5', max_positions: '8', max_orders_per_day: '10',
    horizon: 'years', allow_symbols: ['VOO', 'VTI', 'BND'], deny_symbols: [], order_types: ['limit', 'market'],
    allow_fractional: true });
  assert.ok(!('version' in payload)); // the server assigns it
  for (const value of Object.values(payload)) assert.notEqual(typeof value, 'number');
  const bad = mandateForm.readMandate({ ...raw, budget_usd: '5k', max_position_pct: '', max_positions: '2.5', order_types: [] });
  assert.deepEqual(bad.errors, { budget_usd: 'mandate_err_number', max_position_pct: 'mandate_err_required',
    max_positions: 'mandate_err_count', order_types: 'mandate_err_choice' });
  for (const field of mandateForm.MANDATE_FIELDS) assert.ok(dict.en[field.key] && dict.ko[field.key], field.key);
});

await test('order statuses: the final ones match the server, every label exists', () => {
  assert.deepEqual([...orderView.FINAL_ORDER_STATUSES].sort(),
    ['canceled', 'done_for_day', 'expired', 'filled', 'rejected', 'replaced']);
  for (const status of ['new', 'accepted', 'pending_new', 'partially_filled', 'filled', 'canceled', 'rejected']) {
    const info = orderView.orderStatusInfo(status);
    assert.ok(info.key && dict.en[info.key] && dict.ko[info.key], status);
  }
  assert.equal(orderView.orderStatusInfo('FILLED').final, true);
  assert.equal(orderView.orderStatusInfo('weird_new_status').key, null);
  assert.equal(orderView.orderStatusInfo('__proto__').key, null);
  assert.equal(timeline.anyPending(timeline.normalizeOrders([order({ status: 'new' })])), true);
  assert.equal(timeline.anyPending(timeline.normalizeOrders([order()])), false);
  // an uncertain leg (no order) does not keep the polling alive
  assert.equal(timeline.anyPending([{ client_order_id: 'c1', broker_order_id: null, order: null }]), false);
  assert.deepEqual(timeline.normalizeOrders([order()])[0].broker_order_id, 'o1');
});

await test('poller: every 3 s, at most 60 s, stops when settled or when the card is gone', async () => {
  const run = async ({ pendingFor = Infinity, aliveFor = Infinity, failAt = -1 }) => {
    let now = 0;
    const queue = [];
    const reads = [];
    let stopped = null;
    timeline.createPoller({
      read: async () => {
        reads.push(now);
        if (reads.length === failAt) throw new Error('offline');
        return { n: reads.length };
      },
      pending: (result) => result.n < pendingFor,
      alive: () => reads.length < aliveFor,
      onResult() {},
      onStop: (reason) => { stopped = reason; },
      now: () => now,
      delay: (fn, ms) => { queue.push({ fn, at: now + ms }); return fn; },
      cancel: (fn) => { const i = queue.findIndex((q) => q.fn === fn); if (i >= 0) queue.splice(i, 1); },
    });
    while (queue.length && stopped === null) {
      const next = queue.shift();
      now = next.at;
      await next.fn();
    }
    return { reads, stopped, left: queue.length };
  };
  const timeout = await run({});
  assert.equal(timeout.stopped, 'timeout');
  assert.equal(timeout.reads.length, 20);
  assert.equal(timeout.reads[0], 3000);
  assert.equal(timeout.reads.at(-1), 60000);
  assert.equal(timeout.left, 0);
  const settled = await run({ pendingFor: 3 });
  assert.deepEqual([settled.stopped, settled.reads.length], ['settled', 3]);
  const gone = await run({ aliveFor: 2 });
  assert.deepEqual([gone.stopped, gone.reads.length], ['gone', 2]);
  const flaky = await run({ pendingFor: 4, failAt: 2 });
  assert.deepEqual([flaky.stopped, flaky.reads.length], ['settled', 4]); // a failed read is skipped
});

await test('popover placement stays inside the viewport', () => {
  const viewport = { width: 400, height: 800 };
  const size = { width: 340, height: 200 };
  assert.deepEqual(placePopover({ left: 20, top: 100, bottom: 120 }, size, viewport), { left: 20, top: 126 });
  assert.deepEqual(placePopover({ left: 380, top: 100, bottom: 120 }, size, viewport), { left: 52, top: 126 });
  assert.deepEqual(placePopover({ left: 20, top: 700, bottom: 720 }, size, viewport), { left: 20, top: 494 });
  const tall = placePopover({ left: 0, top: 300, bottom: 320 }, { width: 500, height: 900 }, viewport);
  assert.deepEqual(tall, { left: 8, top: 8 });
});

await test('gains and losses carry a sign and a mark, never colour alone', () => {
  assert.deepEqual(changeParts('12.5'), { mark: '▲', tone: 'up', text: '+$12.50' });
  assert.deepEqual(changeParts('-3.1'), { mark: '▼', tone: 'down', text: '−$3.10' });
  assert.deepEqual(changeParts('0.00'), { mark: '', tone: 'flat', text: '$0.00' });
  assert.deepEqual(changeParts(null), { mark: '', tone: 'muted', text: '–' });
});

await test('settings errors are mapped by code and never echo a value', () => {
  assert.equal(settingsErrorText({ error: 'alpaca_invalid', message: 'x' }), dict.en.settings_err_alpaca_invalid);
  assert.equal(settingsErrorText({ error: 'alpaca_unreachable' }), dict.en.settings_err_alpaca_unreachable);
  assert.equal(settingsErrorText({ error: 'alpaca_pair_required' }), dict.en.settings_err_alpaca_pair);
  assert.equal(settingsErrorText({ error: 'alpaca_not_paper' }), dict.en.settings_err_alpaca_not_paper);
  const invalid = settingsErrorText({ error: 'invalid_value', body: { field: 'tavily_key' }, message: 'server text' });
  assert.equal(invalid, `${dict.en.settings_tavily_key}: ${dict.en.settings_err_invalid_value}`);
  assert.ok(invalid.includes('ASCII'));
  assert.equal(settingsErrorText({ error: 'settings_not_saved', message: 'The settings could not be saved.' }), 'The settings could not be saved.');
});

// --- the shared ticker ---------------------------------------------------------------------------------------

await test('one ticker for every subscriber; detached nodes are dropped and the timer stops', () => {
  const a = document.createElement('div');
  const b = document.createElement('div');
  document.body.append(a, b);
  const seen = [];
  const before = clock.timers;
  ticker.onTick(a, (now) => seen.push(['a', now]));
  ticker.onTick(b, (now) => seen.push(['b', now]));
  assert.equal(clock.timers - before, 1, 'one interval for two subscribers');
  assert.equal(seen.length, 2, 'each subscriber is called once at once');
  advance(1000);
  assert.equal(seen.length, 4);
  b.remove();
  advance(1000);
  assert.equal(seen.length, 5);
  assert.equal(ticker.tickerStats().subscribers, 1);
  a.remove();
  advance(1000);
  assert.deepEqual(ticker.tickerStats(), { subscribers: 0, running: false });
});

// --- components on the fake DOM ---------------------------------------------------------------------------------

await test('tool card: status, sources, external content as text; history is a compact row', () => {
  const turn = liveTurn();
  dispatch({ type: 'tool-input-available', toolCallId: 'c1', toolName: 'search_news', input: { symbols: ['VOO'], q: '<img src=x onerror=alert(1)>' } }, turn);
  const card = turn.slots.activity.querySelector('.toolcard');
  assert.equal(card.getAttribute('data-status'), 'running');
  assert.equal(card.getAttribute('data-family'), 'research');
  assert.equal(card.querySelector('.toolcard-status').textContent, 'Running');
  assert.equal(turn.slots.activity.querySelectorAll('.tool-row').length, 0, 'the shell fallback row is replaced');
  dispatch({ type: 'tool-output-available', toolCallId: 'c1', isError: false,
    output: '[E1] <untrusted source="E1"><script>alert(1)</script> headline</untrusted>',
    sources: [{ id: 'E1', title: 'Headline', url: 'https://news.example.com/1', publisher: 'Demo', tier: 2 }] }, turn);
  assert.equal(card.getAttribute('data-status'), 'done');
  assert.equal(card.querySelector('.toolcard-status').textContent, 'Completed');
  assert.equal(card.querySelector('.toolcard-chip').textContent, '1 source');
  assert.match(card.querySelector('.toolcard-time').textContent, /^\d+(\.\d)? s$/);
  const external = card.querySelector('.toolcard-external');
  assert.equal(external.querySelector('.toolcard-external-label').textContent, 'External contentE1');
  assert.equal(external.querySelector('pre').textContent, '<script>alert(1)</script> headline');
  assert.ok(card.querySelector('.toolcard-pre').textContent.includes('<img src=x onerror=alert(1)>'), 'arguments are text');
  assert.ok(!document.created.includes('script') && !document.created.includes('img'));
  assert.equal(sources.getSource('s1', 'E1').title, 'Headline', 'the turn seeds the source cache');

  dispatch({ type: 'tool-input-available', toolCallId: 'c2', toolName: 'get_quotes', input: {} }, turn);
  emit('turn:end', { turn, outcome: 'stopped', sessionId: 's1' });
  const second = turn.slots.activity.querySelectorAll('.toolcard')[1];
  assert.equal(second.querySelector('.toolcard-status').textContent, 'Stopped');

  const replay = replayTurn();
  dispatch({ type: 'tool-input-available', toolCallId: 'h1', toolName: 'propose_trades' }, replay);
  dispatch({ type: 'tool-output-available', toolCallId: 'h1', toolName: 'propose_trades', isError: true }, replay);
  const compact = replay.slots.activity.querySelector('.toolcard');
  assert.ok(compact.classList.contains('is-compact'));
  assert.equal(compact.querySelectorAll('details').length, 0);
  assert.equal(compact.querySelector('.toolcard-status').textContent, 'Error');
});

await test('thinking block: collapsed, "Thinking…" then "Thought for N s"; text only', () => {
  const turn = liveTurn();
  dispatch({ type: 'start', messageId: 'm1' }, turn);
  dispatch({ type: 'reasoning-delta', delta: 'Checking <b>the</b> account.' }, turn);
  const details = turn.slots.thinking.querySelector('details');
  assert.equal(details.hasAttribute('open'), false);
  assert.equal(details.querySelector('.thinking-label').textContent, 'Thinking…');
  assert.equal(details.querySelector('.thinking-body').textContent, 'Checking <b>the</b> account.');
  dispatch({ type: 'tool-input-available', toolCallId: 'c1', toolName: 'get_account', input: {} }, turn);
  dispatch({ type: 'reasoning-delta', delta: 'Now quotes.' }, turn);
  assert.equal(details.querySelector('.thinking-body').textContent, 'Checking <b>the</b> account.\n\nNow quotes.');
  dispatch({ type: 'text-delta', delta: 'Answer' }, turn);
  assert.match(details.querySelector('.thinking-label').textContent, /^Thought for \d+ s$/);
  assert.equal(turn.slots.thinking.querySelectorAll('details').length, 1);
});

await test('grounding flags: a collapsed note with the numbers; history reads them from the message', () => {
  const turn = liveTurn();
  dispatch({ type: 'data-grounding-flags', numbers: ['12%', '$1.2B'] }, turn);
  const note = turn.slots.extras.querySelector('.flagnote');
  assert.equal(note.hasAttribute('open'), false);
  assert.equal(note.querySelector('.flagnote-title').textContent, '2 numbers not found in the sources');
  assert.equal(note.querySelector('.flagnote-peek').textContent, '12%, $1.2B');
  const replay = replayTurn('s1', { grounding_flags: ['7.5%'] });
  dispatch({ type: 'data-grounding-flags' }, replay);
  assert.equal(replay.slots.extras.querySelector('.flagnote-title').textContent, '1 number not found in the sources');
});

await test('search suggestions go into a sandboxed iframe, untouched, never into the page', () => {
  const turn = liveTurn();
  const html = '<style>.c{color:red}</style><div class="c"><a href="https://www.google.com/search?q=voo" target="_blank">VOO</a></div><script>alert(1)</script>';
  dispatch({ type: 'source-grounding', search_suggestions_html: html }, turn);
  const frame = turn.slots.extras.querySelector('iframe');
  assert.equal(frame.getAttribute('sandbox'), 'allow-popups allow-popups-to-escape-sandbox');
  assert.equal(frame.getAttribute('referrerpolicy'), 'no-referrer');
  assert.equal(frame.srcdoc, `<base target="_blank">${html}`); // Google's HTML itself is untouched
  assert.ok(frame.srcdoc.endsWith(html));
  assert.equal(frame.childNodes.length, 0);
  assert.ok(!document.created.includes('script') && !document.created.includes('style'));
  assert.equal(turn.slots.extras.querySelector('.extra-h').textContent, 'Google Search suggestions');
});

await test('citation chips: a button for a known source, a plain muted chip for an unknown one', () => {
  sources.seedSources('chips', [{ id: 'E1', title: 'Fund page', url: 'https://example.com/a', publisher: 'Issuer', tier: 1 }]);
  const known = sources.sourceChip('chips', 'E1');
  assert.equal(known.localName, 'button');
  assert.equal(known.getAttribute('data-source'), 'E1');
  assert.equal(known.textContent, 'E1');
  const unknown = sources.sourceChip('chips', 'E77');
  assert.equal(unknown.localName, 'span');
  assert.ok(unknown.classList.contains('is-unknown'));
  assert.equal(unknown.hasAttribute('data-source'), false);
  // a later, fuller record fills the gaps without losing what was known
  sources.seedSources('chips', [{ id: 'E1', title: 'Other title', snippet: 'A snippet', published_at: '2026-09-01' }]);
  assert.deepEqual(sources.getSource('chips', 'E1'), { id: 'E1', title: 'Fund page', url: 'https://example.com/a',
    publisher: 'Issuer', tier: 1, snippet: 'A snippet', published_at: '2026-09-01' });
  sources.seedSources('chips', [{ id: '../x', title: 'bad id' }, null, { title: 'no id' }]);
  assert.equal(sources.getSource('chips', '../x'), null);
  assert.equal(sources.getSource('other', 'E1'), null);
});

// --- panels ---------------------------------------------------------------------------------------------

await test('portfolio panel: PAPER (verified), plain PAPER, a NOT-paper warning; not connected; broker error', async () => {
  const view = document.createElement('section');
  document.body.appendChild(view);
  panel('portfolio').mount(view);
  const body = (paper) => {
    const account = { equity: '132100.00', cash: '100000.00', buying_power: '100000.00', last_equity: '131000.00' };
    if (paper !== undefined) account.paper = paper;
    return { connected: true, paper: true, account, open_orders: [],
      positions: [{ symbol: '<b>VTI</b>', qty: '40', avg_entry_price: '300.00', current_price: '330.00',
        market_value: '13200.00', unrealized_pl: '-1200.00', unrealized_plpc: '-0.0909' }] };
  };
  const show = async (status, payload) => {
    net.routes = [{ method: 'GET', match: '/api/portfolio', status, body: payload }];
    await refreshPortfolio();
    return view;
  };
  const badges = () => view.querySelectorAll('.badge').map((b) => b.textContent);

  await show(200, body(true));
  assert.deepEqual(badges(), ['PAPER (verified)']);
  const stats = view.querySelector('.stats').textContent;
  assert.ok(stats.includes('Equity$132,100.00') && stats.includes('Buying power$100,000.00'));
  assert.ok(stats.includes('Day change▲+$1,100.00(+0.84%)'), stats); // sign, mark and percentage, exact
  const row = view.querySelector('tbody').querySelector('tr');
  assert.equal(row.querySelector('th').textContent, '<b>VTI</b>'); // a symbol is text
  assert.equal(row.querySelectorAll('td').at(-1).textContent, '▼−$1,200.00(−9.09%)');
  assert.ok(!document.created.includes('b'));

  await show(200, body(null));
  assert.deepEqual(badges(), ['PAPER']);
  await show(200, body(undefined)); // the field is absent on an older server
  assert.deepEqual(badges(), ['PAPER']);

  await show(200, body(false));
  assert.deepEqual(badges(), []);
  const warning = view.querySelector('.notice');
  assert.equal(warning.getAttribute('data-tone'), 'bad');
  assert.ok(warning.textContent.includes('NOT a paper account'));

  // a live account's keys: the server shows no data, the Settings say why
  setState({ settings: { alpaca: 'not_paper', trading_enabled: true } });
  await show(200, { connected: false });
  assert.equal(view.querySelector('.notice').getAttribute('data-tone'), 'bad');
  assert.ok(view.textContent.includes('NOT a paper account'));
  setState({ settings: { alpaca: 'missing', trading_enabled: true } });
  await show(200, { connected: false });
  assert.ok(view.textContent.includes('not connected'));
  assert.equal(view.querySelector('.notice').querySelector('button').textContent, 'Open Settings');
  assert.equal(view.querySelector('.stats'), null);

  await show(502, { connected: true, error: 'broker_unavailable', message: 'The broker could not be reached. Try again shortly.' });
  assert.ok(view.textContent.includes('The broker could not be reached.'));
  assert.equal(view.querySelector('.notice').querySelector('button').textContent, 'Try again');
});

await test('mandate form: fills from a mandate, reads back strings, shows server field errors as text', async () => {
  const saved = [];
  const form = mandateForm.createMandateForm({
    mandate: { direction_text: 'Core ETFs', budget_usd: '10000', universe: 'us_equity', allow_symbols: ['VOO', 'BND'],
      deny_symbols: [], max_position_pct: '15', max_positions: 8, cash_buffer_pct: '5', horizon: 'years',
      order_types: ['limit'], max_orders_per_day: 10, max_order_usd: null, price_band_pct: '3', allow_fractional: true,
      version: 3 },
    onSave: (payload) => saved.push(payload),
  });
  document.body.appendChild(form.el);
  const input = (name) => form.el.querySelectorAll('.field').find((f) => f.getAttribute('data-field') === name).querySelector('.field-input');
  assert.equal(input('budget_usd').value, '10000');
  assert.equal(input('allow_symbols').value, 'VOO, BND');
  assert.equal(input('max_order_usd').value, '');
  const save = form.el.querySelector('.btn-primary');
  input('budget_usd').value = '12,500.50';
  input('allow_symbols').value = 'voo qqq';
  save.click();
  assert.equal(saved.length, 1);
  assert.equal(saved[0].budget_usd, '12500.50');
  assert.deepEqual(saved[0].allow_symbols, ['VOO', 'QQQ']);
  assert.deepEqual(saved[0].order_types, ['limit']);
  assert.equal(saved[0].max_positions, '8');
  assert.equal(saved[0].max_order_usd, null);
  assert.ok(!('version' in saved[0]));
  // a shape error stops the save and names the field
  input('max_position_pct').value = 'lots';
  save.click();
  assert.equal(saved.length, 1);
  const errors = () => form.el.querySelectorAll('.field-error').filter((e) => !e.hidden).map((e) => e.textContent);
  assert.deepEqual(errors(), ['Enter a number, such as 5000 or 2.5.']);
  // the server's 422: its messages go next to the inputs, as text
  input('max_position_pct').value = '250';
  form.setErrors([{ loc: 'max_position_pct', msg: 'Input should be <= 100 <img src=x>' }, { loc: 'allow_symbols.1', msg: 'invalid symbol' }],
    'Some mandate fields are invalid.');
  assert.deepEqual(errors(), ['Input should be <= 100 <img src=x>', 'invalid symbol']);
  assert.equal(input('max_position_pct').getAttribute('aria-invalid'), 'true');
  assert.equal(form.el.querySelector('.mandate-status').textContent, 'Some mandate fields are invalid.');
  assert.ok(!document.created.includes('img'));
});

finish('component checks');
