/* One shared one-second ticker for everything that counts time on screen (ticket countdowns, quote
 * ages), instead of a timer per card.
 *
 * `onTick(node, fn)` calls `fn(nowMs)` now and then once a second for as long as `node` is in the
 * document. A subscriber whose node has left the document (the chat was switched, the card was
 * replaced) is dropped, and the interval stops when nobody is left. The clock is injectable for
 * `scripts/test_components.mjs`.
 */

const subscribers = new Set();
let timer = null;
let clock = {
  now: () => Date.now(),
  every: (fn, ms) => setInterval(fn, ms),
  cancel: (id) => clearInterval(id),
};

function tick() {
  const now = clock.now();
  for (const entry of [...subscribers]) {
    if (!entry.node.isConnected) {
      subscribers.delete(entry);
      continue;
    }
    try {
      entry.fn(now);
    } catch (err) {
      console.error('tick subscriber failed', err);
    }
  }
  if (subscribers.size === 0 && timer !== null) {
    clock.cancel(timer);
    timer = null;
  }
}

/* Subscribe `fn` for the lifetime of `node`. Returns an unsubscribe function. */
export function onTick(node, fn) {
  const entry = { node, fn };
  subscribers.add(entry);
  if (timer === null) timer = clock.every(tick, 1000);
  try {
    fn(clock.now());
  } catch (err) {
    console.error('tick subscriber failed', err);
  }
  return () => subscribers.delete(entry);
}

export function nowMs() {
  return clock.now();
}

/* Tests: swap the clock, and read how many subscribers and timers are alive. */
export function setTickerClock(next) {
  if (timer !== null) clock.cancel(timer);
  timer = null;
  subscribers.clear();
  clock = next;
}

export function tickerStats() {
  return { subscribers: subscribers.size, running: timer !== null };
}
