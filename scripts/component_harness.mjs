// Shared setup for the component tests (scripts/test_components.mjs, scripts/test_ticket_card.mjs).
//
// Importing this module installs the fake DOM (with the `#thread` element the components look up),
// replaces `fetch` with a scripted stand-in, loads the real components, and gives the shared ticker
// a manual clock, so nothing here waits, reaches a network or needs a browser.
import { installFakeDom } from './fake_dom.mjs';

export const document = installFakeDom();
export const thread = document.createElement('div');
thread.id = 'thread';
document.body.appendChild(thread);

// --- a scripted fetch ---------------------------------------------------------------------------------
// `net.routes` is [{method, match, status, body, once?, gate?}]: the first route whose method matches
// and whose `match` is part of the path answers (a `once` route is then removed, and a `gate`
// promise is awaited before answering); no route means a network failure. `net.calls` records every
// request as {method, path, body, headers}.
export const net = { calls: [], routes: [] };
globalThis.fetch = async (path, init = {}) => {
  const call = { method: init.method || 'GET', path, body: init.body ? JSON.parse(init.body) : null,
    headers: init.headers || {} };
  net.calls.push(call);
  const route = net.routes.find((r) => r.method === call.method && path.includes(r.match));
  if (!route) throw new TypeError(`no route for ${call.method} ${path}`);
  if (route.once) net.routes.splice(net.routes.indexOf(route), 1);
  if (route.gate) await route.gate;
  const text = JSON.stringify(route.body);
  return { ok: route.status >= 200 && route.status < 300, status: route.status,
    json: async () => JSON.parse(text), text: async () => text };
};

/* Let pending promises and their continuations run. */
export const flush = async () => {
  for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

export const { dict, t } = await import('../web/app/js/i18n.js');
export const { dispatch, panel } = await import('../web/app/js/registry.js');
export const { emit, on, setState } = await import('../web/app/js/state.js');
export const { registerDefaultHandlers, Turn } = await import('../web/app/js/turn.js');
export const ticker = await import('../web/app/js/components/ticker.js');
export const model = await import('../web/app/js/components/ticket-model.js');
export const toolcard = await import('../web/app/js/components/toolcard.js');
await import('../web/app/js/components/thinking.js');
export const sources = await import('../web/app/js/components/sources.js');
await import('../web/app/js/components/grounding.js');
export const mandateForm = await import('../web/app/js/components/mandate-form.js');
export const timeline = await import('../web/app/js/components/ticket-timeline.js');
export const orderView = await import('../web/app/js/components/order-view.js');
export const { placePopover } = await import('../web/app/js/components/popover.js');
export const { changeParts } = await import('../web/app/js/components/portfolio.js');
export const { refreshPortfolio } = await import('../web/app/js/components/portfolio-store.js');
export const { settingsErrorText } = await import('../web/app/js/components/settings.js');
await import('../web/app/js/components/ticket-thread.js');
await import('../web/app/js/components/orders.js');
registerDefaultHandlers();

// A manual clock for the shared ticker: no real interval is ever started.
export const clock = { time: Date.parse('2026-10-02T12:00:00Z'), ticks: [], timers: 0 };
ticker.setTickerClock({
  now: () => clock.time,
  every(fn) { clock.ticks.push(fn); clock.timers += 1; return fn; },
  cancel(fn) { clock.ticks = clock.ticks.filter((f) => f !== fn); },
});
export const advance = (ms) => {
  clock.time += ms;
  for (const fn of [...clock.ticks]) fn();
};

let checks = 0;
export async function test(name, fn) {
  try {
    await fn();
    checks += 1;
    console.log(`ok   ${name}`);
  } catch (err) {
    console.error(`FAIL ${name}`);
    throw err;
  }
}
export const finish = (what) => console.log(`\n${checks} ${what} passed`);

// --- fixtures -------------------------------------------------------------------------------------------

export const leg = (over = {}) => ({ symbol: 'VOO', side: 'buy', qty: '3', notional: null, type: 'limit',
  time_in_force: 'day', limit_price: '612.00', ...over });
export const decision = (index, legOver = {}, over = {}) => {
  const l = leg(legOver);
  return { index, verdict: 'accept', leg: l, original: l, reasons: [], est_price: '610.00', est_cost_usd: '1830.00', ...over };
};
let serial = 0;
/* A proposed ticket quoted 12 s ago that expires in 4:48 (by the manual clock). Every ticket has
 * its own hash, so a test can tell which card a confirm came from. */
export function ticket(over = {}) {
  serial += 1;
  const decisions = over.decisions || [decision(0)];
  return {
    id: `T${String(serial).padStart(4, '0')}`, session_id: 's1', status: 'proposed',
    legs: decisions.filter((d) => d.verdict !== 'reject').map((d) => d.leg), decisions,
    thesis: 'Low-cost core holding [E1]', risks: 'Equity drawdowns', invalidation: 'Expense ratio rises',
    evidence_ids: ['E1', 'E9'], mandate_version: 1,
    quote_as_of: new Date(clock.time - 12000).toISOString(), created_at: new Date(clock.time - 12000).toISOString(),
    expires_at: new Date(clock.time + 288000).toISOString(), hash: serial.toString(16).padStart(64, '0'),
    client_order_ids: decisions.map((d, i) => `c${i}`), broker_order_ids: [], ...over,
  };
}
export const order = (over = {}) => ({ id: 'o1', client_order_id: 'c0', symbol: 'VOO', side: 'buy', qty: '3', notional: null,
  type: 'limit', time_in_force: 'day', limit_price: '612.00', status: 'filled', filled_qty: '3',
  filled_avg_price: '610.10', submitted_at: '2026-10-02T12:00:01Z', updated_at: '2026-10-02T12:00:02Z', ...over });

/* A live assistant turn of chat `sessionId`, attached to the thread. */
export function liveTurn(sessionId = 's1') {
  const turn = new Turn();
  turn.sessionId = sessionId;
  thread.appendChild(turn.el);
  return turn;
}
export function replayTurn(sessionId = 's1', meta) {
  const turn = new Turn({ replay: true });
  turn.sessionId = sessionId;
  if (meta) turn.data.meta = meta;
  thread.appendChild(turn.el);
  return turn;
}
