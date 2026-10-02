// Unit tests for the web app's shell logic that needs no browser. Run: node scripts/test_shell.mjs
//
// - registry.js: several listeners per event type, explicit replacement of the shell's fallback
//   views, and "the first finish wins" (later events for that turn are ignored);
// - state.js: when a message may be sent, and the gate that discards a stale session load;
// - i18n.js: text and attribute bindings keep separate variables;
// - turn.js on the fake DOM: the shell's handlers through the real dispatch.
import assert from 'node:assert/strict';
import { installFakeDom } from './fake_dom.mjs';
import { dict, bindAttr, bindText, setLang, t } from '../web/app/js/i18n.js';
import { addPostProcessor } from '../web/app/js/markdown.js';
import { dispatch, registerEvent, replaceEvent } from '../web/app/js/registry.js';
import { canSend, createLoadGate } from '../web/app/js/state.js';
import { createThread } from '../web/app/js/thread.js';
import { registerDefaultHandlers, Turn } from '../web/app/js/turn.js';

let checks = 0;
function test(name, fn) {
  try {
    fn();
    checks += 1;
    console.log(`ok   ${name}`);
  } catch (err) {
    console.error(`FAIL ${name}`);
    throw err;
  }
}

const quiet = (fn) => {
  const original = console.error;
  console.error = () => {};
  try {
    return fn();
  } finally {
    console.error = original;
  }
};

// --- registry ------------------------------------------------------------------------------------

test('registerEvent adds listeners; all run in order, and one failing does not stop the rest', () => {
  const seen = [];
  registerEvent('t-multi', () => seen.push('first'));
  registerEvent('t-multi', () => { throw new Error('broken component'); });
  registerEvent('t-multi', (event, turn) => seen.push(`third:${event.value}:${turn.name}`));
  const handled = quiet(() => dispatch({ type: 't-multi', value: 7 }, { name: 'turn' }));
  assert.equal(handled, true);
  assert.deepEqual(seen, ['first', 'third:7:turn']);
});

test('a component listening for finish or error does not displace the shell', () => {
  const seen = [];
  registerEvent('t-core', () => seen.push('shell'));
  registerEvent('t-core', () => seen.push('component'));
  dispatch({ type: 't-core' }, {});
  assert.deepEqual(seen, ['shell', 'component']);
});

test('replaceEvent swaps only the fallback view, in either registration order', () => {
  const seen = [];
  registerEvent('t-tool', () => seen.push('core'));
  registerEvent('t-tool', () => seen.push('fallback row'), { fallback: true });
  replaceEvent('t-tool', () => seen.push('ToolCard'));
  dispatch({ type: 't-tool' }, {});
  assert.deepEqual(seen, ['core', 'ToolCard']);

  // The component is imported before the shell registers its fallback.
  seen.length = 0;
  replaceEvent('t-tool-early', () => seen.push('ToolCard'));
  registerEvent('t-tool-early', () => seen.push('fallback row'), { fallback: true });
  dispatch({ type: 't-tool-early' }, {});
  assert.deepEqual(seen, ['ToolCard']);

  // A plain listener added first does not suppress the fallback.
  seen.length = 0;
  registerEvent('t-tool-extra', () => seen.push('extra'));
  registerEvent('t-tool-extra', () => seen.push('fallback row'), { fallback: true });
  dispatch({ type: 't-tool-extra' }, {});
  assert.deepEqual(seen, ['extra', 'fallback row']);
});

test('unknown and malformed events are ignored', () => {
  assert.equal(dispatch({ type: 't-nobody-listens' }, {}), false);
  for (const bad of [null, undefined, 'finish', 42, {}, { type: 7 }, []]) assert.equal(dispatch(bad, {}), false);
});

test('the first finish wins; later events for that turn are ignored', () => {
  const seen = [];
  registerEvent('finish', (event) => seen.push(`finish:${event.elapsed_ms}`));
  registerEvent('t-after', () => seen.push('after'));
  const turn = {};
  assert.equal(dispatch({ type: 't-after' }, turn), true);
  assert.equal(dispatch({ type: 'finish', usage: { input: 1 }, elapsed_ms: 10 }, turn), true);
  assert.equal(turn.finish.elapsed_ms, 10);
  assert.equal(dispatch({ type: 'finish', usage: {}, elapsed_ms: 99 }, turn), false);
  assert.equal(dispatch({ type: 't-after' }, turn), false);
  assert.equal(turn.finish.elapsed_ms, 10);
  assert.deepEqual(seen, ['after', 'finish:10']);
  // Another turn is unaffected.
  assert.equal(dispatch({ type: 't-after' }, {}), true);
});

// --- state ---------------------------------------------------------------------------------------

test('canSend: not while a response streams, not while a chat is loading', () => {
  assert.equal(canSend({ streaming: false, loadingSession: null }), true);
  assert.equal(canSend({ streaming: true, loadingSession: null }), false);
  assert.equal(canSend({ streaming: false, loadingSession: 'sess-2' }), false);
  assert.equal(canSend({ streaming: true, loadingSession: 'sess-2' }), false);
});

test('the load gate: only the newest session load may render', () => {
  const gate = createLoadGate();
  const a = gate.begin();
  assert.equal(gate.isCurrent(a), true);
  const b = gate.begin(); // the user clicked another chat before the first one arrived
  assert.equal(gate.isCurrent(a), false);
  assert.equal(gate.isCurrent(b), true);
  gate.cancel(); // New chat: nothing that is still loading may render
  assert.equal(gate.isCurrent(b), false);
  const c = gate.begin();
  assert.equal(gate.isCurrent(c), true);
  assert.equal(gate.isCurrent(undefined), false);
});

// --- i18n ----------------------------------------------------------------------------------------

test('text and attribute bindings on one element keep separate variables', () => {
  const doc = installFakeDom();
  const el = doc.createElement('button');
  doc.body.appendChild(el);
  bindText(el, 'tool_running', { name: 'get_quotes' });
  bindAttr(el, 'aria-label', 'rename_chat', { title: 'ETF plan' });
  bindAttr(el, 'title', 'stats_tokens', { n: '12' });
  assert.equal(el.textContent, 'Running get_quotes…');
  assert.equal(el.getAttribute('aria-label'), 'Rename “ETF plan”');
  assert.equal(el.getAttribute('title'), '12 tokens');
  setLang('ko');
  assert.equal(el.textContent, 'get_quotes 실행 중…');
  assert.equal(el.getAttribute('aria-label'), '“ETF plan” 이름 바꾸기');
  assert.equal(el.getAttribute('title'), '토큰 12개');
  assert.equal(doc.documentElement.lang, 'ko');
  // Re-binding without variables clears the old ones.
  bindText(el, 'stop');
  setLang('en');
  assert.equal(el.textContent, 'Stop');
  assert.equal(el.getAttribute('aria-label'), 'Rename “ETF plan”');
});

test('copy fixes from the review are in the dictionary', () => {
  assert.ok(dict.ko.empty_lede.includes('승인하기 전에는') && !dict.ko.empty_lede.includes('직접 확인하기'));
  assert.equal(dict.ko.suggest_3, '배당 ETF로 1,500달러 규모의 배당 포트폴리오를 추가해 줘');
  assert.equal(dict.en.tool_unnamed, 'tool');
  assert.equal(dict.ko.tool_unnamed, '도구');
});

// --- the shell's handlers on a turn -------------------------------------------------------------

test('a turn through the real dispatch: text, tools, error, then nothing after finish', () => {
  const doc = installFakeDom();
  globalThis.DOMPurify = { removed: [], sanitize(text) { this.removed = []; return text; } };
  registerDefaultHandlers();
  const turn = new Turn();
  turn.sessionId = 'sess-1';
  doc.body.appendChild(turn.el);

  dispatch({ type: 'start', messageId: 'turn-1' }, turn);
  dispatch({ type: 'reasoning-delta', delta: 'Checking quotes.' }, turn);
  dispatch({ type: 'tool-input-available', toolCallId: 'c1', toolName: 'get_quotes', input: {} }, turn);
  dispatch({ type: 'tool-output-available', toolCallId: 'c1', output: 'ok', isError: false }, turn);
  dispatch({ type: 'tool-input-available', toolCallId: 'c2', toolName: '', input: {} }, turn);
  dispatch({ type: 'text-delta', delta: 'Buy **$VOO** [E1].' }, turn);
  dispatch({ type: 'error', message: 'Something failed.' }, turn);
  dispatch({ type: 'finish', usage: { input: 5, output: 7 }, elapsed_ms: 1200 }, turn);

  assert.equal(turn.id, 'turn-1');
  assert.equal(turn.finish.elapsed_ms, 1200);
  assert.equal(turn.hadError, true);
  // (The parser holds the last character back until the turn is closed.)
  assert.equal(turn.slots.body.textContent, 'Buy $VOO [E1]');
  assert.equal(turn.slots.body.querySelectorAll('strong').length, 1);
  const rows = turn.slots.activity.children.map((row) => `${row.getAttribute('data-status')}:${row.textContent}`);
  assert.deepEqual(rows, ['done:✓ get_quotes', `running:Running ${t('tool_unnamed')}…`]);
  assert.equal(turn.slots.status.querySelector('.turn-error-text').textContent, 'Something failed.');

  // A duplicate finish and anything after it change nothing.
  dispatch({ type: 'finish', usage: {}, elapsed_ms: 1 }, turn);
  dispatch({ type: 'text-delta', delta: ' Late text.' }, turn);
  dispatch({ type: 'tool-input-available', toolCallId: 'c3', toolName: 'late_tool', input: {} }, turn);
  dispatch({ type: 'error', message: 'late error' }, turn);
  assert.equal(turn.finish.elapsed_ms, 1200);
  assert.equal(turn.slots.activity.children.length, 2);
  assert.equal(turn.slots.status.querySelectorAll('.turn-error').length, 1);

  turn.close('error');
  assert.equal(turn.slots.body.textContent, 'Buy $VOO [E1].');
  assert.equal(turn.slots.activity.children[1].getAttribute('data-status'), 'stopped');
  delete globalThis.DOMPurify;
});

test('a component can replace the tool rows without touching text, error or finish', () => {
  const doc = installFakeDom();
  globalThis.DOMPurify = { removed: [], sanitize(text) { this.removed = []; return text; } };
  registerDefaultHandlers();
  const cards = [];
  replaceEvent('tool-input-available', (event) => cards.push(event.toolName));
  registerEvent('finish', (event) => cards.push(`finish:${event.elapsed_ms}`));
  const turn = new Turn();
  doc.body.appendChild(turn.el);
  dispatch({ type: 'tool-input-available', toolCallId: 'c1', toolName: 'get_quotes' }, turn);
  dispatch({ type: 'text-delta', delta: 'Answer.' }, turn);
  dispatch({ type: 'finish', usage: {}, elapsed_ms: 5 }, turn);
  assert.deepEqual(cards, ['get_quotes', 'finish:5']);
  assert.equal(turn.slots.activity.children.length, 0); // the fallback row is gone
  assert.equal(turn.finish.elapsed_ms, 5); // the shell still saw finish
  turn.close('finish');
  assert.equal(turn.slots.body.textContent, 'Answer.');
  delete globalThis.DOMPurify;
});

test('history replay: stored events reach their turn, and post-processors run again afterwards', () => {
  const doc = installFakeDom();
  globalThis.DOMPurify = { removed: [], sanitize(text) { this.removed = []; return text; } };
  registerDefaultHandlers();
  registerEvent('t-history-source', (event, turn) => {
    turn.data.sources = [...(turn.data.sources || []), event.id];
  });
  const passes = [];
  addPostProcessor((root, info) => {
    if (info.turn) passes.push({ id: info.turn.id, session: info.turn.sessionId, final: info.final, sources: info.turn.data.sources });
  });

  const scroller = doc.createElement('div');
  const list = doc.createElement('div');
  scroller.appendChild(list);
  doc.body.appendChild(scroller);
  const view = createThread({ scroller, thread: list, onSuggestion() {} });
  view.renderHistory([
    { role: 'user', content: 'Plan it\nplease', meta: { turn_id: 'h1' } },
    { role: 'assistant', content: 'Low fees [E1].', meta: { turn_id: 'h1' } },
    { role: 'event', content: 'turn summary', meta: { turn_id: 'h1', events: [
      { type: 'tool-input-available', toolName: 'get_quotes', toolCallId: 'c1' },
      { type: 'tool-output-available', toolName: 'get_quotes', toolCallId: 'c1', isError: false },
      { type: 't-history-source', id: 'E1' },
      { type: 'error' }, { type: 'finish' }, { type: 'text-delta', delta: 'never replayed' },
    ] } },
    { role: 'user', content: 'again', meta: { turn_id: 'h2' } },
    { role: 'assistant', content: 'The request failed.', meta: { turn_id: 'h2', error: true } },
  ], { sessionId: 'sess-9' });

  const [user, first, , failed] = list.children;
  assert.equal(user.querySelector('.bubble').textContent, 'Plan it\nplease');
  assert.equal(first.querySelector('.turn-body').textContent, 'Low fees [E1].');
  assert.deepEqual(first.querySelectorAll('.tool-row').map((r) => r.textContent), ['✓ get_quotes']);
  assert.equal(first.querySelectorAll('.turn-error').length, 0);
  assert.equal(first.getAttribute('data-outcome'), 'history');
  assert.equal(failed.querySelector('.turn-error-text').textContent, 'The request failed.');
  assert.equal(failed.querySelector('.turn-body').textContent, '');

  const mine = passes.filter((p) => p.id === 'h1');
  assert.ok(mine.length >= 2);
  assert.deepEqual(mine[0], { id: 'h1', session: 'sess-9', final: false, sources: undefined });
  // The pass after the replay sees the sources the replayed events brought.
  assert.deepEqual(mine[mine.length - 1], { id: 'h1', session: 'sess-9', final: true, sources: ['E1'] });
  delete globalThis.DOMPurify;
});

console.log(`\n${checks} shell checks passed`);
