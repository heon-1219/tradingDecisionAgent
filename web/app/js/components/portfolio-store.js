/* The paper account as last read from `GET /api/portfolio`, shared by the Portfolio and Orders
 * panels and by the ticket card (for "buying power after the trade").
 *
 * `snapshot()` is one of:
 *   {status: 'idle'}                         nothing loaded yet
 *   {status: 'loading', body?}               a request is in flight (the previous body is kept)
 *   {status: 'disconnected'}                 no broker: {connected: false}
 *   {status: 'ready', body, at}              {connected: true, account, positions, open_orders}
 *   {status: 'error', message, body?}        502, or the server could not be reached
 * Every change is announced on the bus as `portfolio:data`. `portfolio:refresh` (emitted after a
 * confirm, a settings change or by a refresh button) reloads it.
 */
import { request } from '../api.js';
import { t } from '../i18n.js';
import { emit, on } from '../state.js';

let current = { status: 'idle' };
let inFlight = null;

export function snapshot() {
  return current;
}

function set(next) {
  current = next;
  emit('portfolio:data', current);
}

export function refreshPortfolio() {
  if (inFlight) return inFlight;
  set({ status: 'loading', body: current.body });
  inFlight = (async () => {
    try {
      const body = await request('GET', '/api/portfolio');
      if (!body || body.connected !== true) set({ status: 'disconnected' });
      else set({ status: 'ready', body, at: Date.now() });
    } catch (err) {
      set({ status: 'error', message: err?.message || t('err_generic'), body: current.body });
    } finally {
      inFlight = null;
    }
    return current;
  })();
  return inFlight;
}

/* Load once if nothing has been loaded yet (the ticket card calls this). */
export function ensurePortfolio() {
  if (current.status === 'idle') refreshPortfolio();
}

on('portfolio:refresh', () => refreshPortfolio());
let lastAlpaca;
on('settings', (settings) => {
  // Keys were added, removed or replaced: what was loaded may no longer be true.
  const alpaca = settings ? settings.alpaca : undefined;
  if (lastAlpaca !== undefined && alpaca !== lastAlpaca && current.status !== 'idle') refreshPortfolio();
  lastAlpaca = alpaca;
});
