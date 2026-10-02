// Unit tests for the web app's SSE client (web/app/js/sse.js). Run: node scripts/test_sse_parser.mjs
//
// Feeds a recorded /api/chat stream (Korean text, emoji, keep-alive comments, multi-line data
// fields, CRLF line endings) to the parser in thousands of random splits, at the string level and
// at the byte level (so splits land inside multi-byte UTF-8 characters), and checks that the same
// events come out every time. Then drives `streamChat` against a stubbed fetch: a normal stream, a
// 404 JSON error, a stream cut before `finish`, and an abort.
import assert from 'node:assert/strict';
import { createSseParser, createSseDecoder, streamChat } from '../web/app/js/sse.js';

// --- a recorded stream -------------------------------------------------------------------------

const EVENTS = [
  { type: 'start', messageId: 'a1b2c3' },
  { type: 'reasoning-delta', delta: 'Checking quotes for VOO and VTI…' },
  { type: 'tool-input-available', toolCallId: 'call_1', toolName: 'get_quotes', input: { symbols: ['VOO', 'VTI'] } },
  { type: 'tool-output-available', toolCallId: 'call_1', output: 'VOO 512.30\nVTI 281.04', isError: false, sources: [] },
  { type: 'source-url', sourceId: 'E1', url: 'https://www.sec.gov/x', title: 'SEC 10-K' },
  { type: 'text-delta', delta: '## 핵심 포트폴리오\n\n저비용 ETF로 구성합니다 [E1]. ' },
  { type: 'text-delta', delta: '| 종목 | 비중 |\n|---|---|\n| VOO | 60% |\n' },
  { type: 'text-delta', delta: '이모지도 괜찮습니다 📈👍, 한글 조합형: 한국어 — “따옴표”.' },
  { type: 'data-grounding-flags', numbers: ['60%'] },
  { type: 'error', message: '일시적인 오류가 발생했습니다.' },
  { type: 'finish', usage: { input: 1200, output: 340 }, elapsed_ms: 4210 },
];

const frame = (event) => `data: ${JSON.stringify(event)}\n\n`;

// LF framing with keep-alive comments between events (the server's `: keep-alive\n\n`).
let LF = '';
EVENTS.forEach((event, i) => {
  if (i % 3 === 1) LF += ': keep-alive\n\n';
  LF += frame(event);
});
LF += ': keep-alive\n\n';

// The same stream with CRLF line endings, a bare-CR event, and a multi-line data field.
const pretty = JSON.stringify(EVENTS[5], null, 2).split('\n').map((l) => `data: ${l}`).join('\r\n');
let MIXED = '';
EVENTS.forEach((event, i) => {
  if (i === 5) MIXED += `${pretty}\r\n\r\n`;
  else if (i === 7) MIXED += `data: ${JSON.stringify(event)}\r\r`;
  else MIXED += `:comment line\r\ndata:${JSON.stringify(event)}\r\n\r\n`;
});

// --- helpers -----------------------------------------------------------------------------------

let seed = 0x9e3779b9;
function rand() { // xorshift32, deterministic
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >>> 17;
  seed ^= seed << 5; seed >>>= 0;
  return seed / 0x100000000;
}

function randomCuts(length, maxPiece) {
  const cuts = [];
  let at = 0;
  while (at < length) {
    at += 1 + Math.floor(rand() * maxPiece);
    if (at < length) cuts.push(at);
  }
  return cuts;
}

function splitAt(seq, cuts) {
  const out = [];
  let prev = 0;
  for (const c of cuts) { out.push(seq.slice(prev, c)); prev = c; }
  out.push(seq.slice(prev));
  return out;
}

function parseString(pieces) {
  const got = [];
  const parser = createSseParser((e) => got.push(e));
  pieces.forEach((p) => parser.push(p));
  parser.end();
  return got;
}

function parseBytes(pieces) {
  const got = [];
  const decoder = createSseDecoder((e) => got.push(e));
  pieces.forEach((p) => decoder.push(p));
  decoder.end();
  return got;
}

let checks = 0;
function test(name, fn) {
  return Promise.resolve().then(fn).then(
    () => { checks += 1; console.log(`ok   ${name}`); },
    (err) => { console.error(`FAIL ${name}`); throw err; },
  );
}

// --- parser tests ------------------------------------------------------------------------------

await test('whole stream in one piece (LF and mixed CRLF/CR)', () => {
  assert.deepEqual(parseString([LF]), EVENTS);
  assert.deepEqual(parseString([MIXED]), EVENTS);
});

await test('one character at a time', () => {
  assert.deepEqual(parseString([...LF]), EVENTS);
  assert.deepEqual(parseString(MIXED.split('')), EVENTS);
});

await test('2,000 random string splits, including between \\r and \\n', () => {
  let crlfSplits = 0;
  for (let i = 0; i < 1000; i += 1) {
    for (const source of [LF, MIXED]) {
      const cuts = randomCuts(source.length, 1 + (i % 40));
      crlfSplits += cuts.filter((c) => source[c - 1] === '\r' && source[c] === '\n').length;
      assert.deepEqual(parseString(splitAt(source, cuts)), EVENTS);
    }
  }
  assert.ok(crlfSplits > 50, `expected many \\r|\\n splits, got ${crlfSplits}`);
});

await test('2,000 random byte splits, including inside multi-byte UTF-8 characters', () => {
  const encoder = new TextEncoder();
  let midCharSplits = 0;
  for (let i = 0; i < 1000; i += 1) {
    for (const source of [LF, MIXED]) {
      const bytes = encoder.encode(source);
      const cuts = randomCuts(bytes.length, 1 + (i % 24));
      // A continuation byte (10xxxxxx) right after a cut means the cut split a character.
      midCharSplits += cuts.filter((c) => (bytes[c] & 0xc0) === 0x80).length;
      assert.deepEqual(parseBytes(splitAt(bytes, cuts)), EVENTS);
    }
  }
  assert.ok(midCharSplits > 500, `expected many mid-character splits, got ${midCharSplits}`);
});

await test('every single-byte position as a two-piece split', () => {
  const bytes = new TextEncoder().encode(MIXED);
  for (let c = 1; c < bytes.length; c += 1) {
    assert.deepEqual(parseBytes([bytes.slice(0, c), bytes.slice(c)]), EVENTS);
  }
});

await test('comments only, blank frames and bad JSON produce nothing', () => {
  assert.deepEqual(parseString([': keep-alive\n\n: keep-alive\n\n\n\n']), []);
  assert.deepEqual(parseString(['data: {not json}\n\n', 'data: [1,2]\n\n', 'data: "str"\n\n']), []);
  assert.deepEqual(parseString(['event: ping\nid: 7\nretry: 100\n\n']), []);
});

await test('a last event without its blank line is still delivered at end()', () => {
  assert.deepEqual(parseString(['data: {"type":"finish"}']), [{ type: 'finish' }]);
  assert.deepEqual(parseString(['data: {"type":"finish"}\r']), [{ type: 'finish' }]);
});

await test('"data:" without a space, and a bare "data" line (an empty data line)', () => {
  // The bare line adds an empty data line; the joined payload "\n{...}" is still valid JSON.
  assert.deepEqual(parseString(['data:{"type":"a"}\n\ndata\ndata: {"type":"b"}\n\n']),
    [{ type: 'a' }, { type: 'b' }]);
});

// --- streamChat against a stubbed fetch --------------------------------------------------------

function streamResponse(chunks, { hang = false, signal } = {}) {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
      for (const c of chunks) controller.enqueue(typeof c === 'string' ? encoder.encode(c) : c);
      if (!hang) controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream; charset=utf-8' } });
}

const realFetch = globalThis.fetch;

await test('streamChat: events in order, headers, finished=true', async () => {
  let seen;
  globalThis.fetch = async (url, init) => {
    seen = { url, init };
    const bytes = new TextEncoder().encode(LF);
    return streamResponse(splitAt(bytes, randomCuts(bytes.length, 7)));
  };
  const got = [];
  const result = await streamChat('sess1', '안녕하세요', { onEvent: (e) => got.push(e) });
  assert.deepEqual(got, EVENTS);
  assert.equal(result.finished, true);
  assert.equal(seen.url, '/api/chat');
  assert.equal(seen.init.method, 'POST');
  assert.equal(seen.init.headers['X-FinAgent-Client'], '1');
  assert.equal(seen.init.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(seen.init.body), { session_id: 'sess1', message: '안녕하세요' });
});

await test('streamChat: an HTTP 404 JSON body rejects with {status, error, message}', async () => {
  globalThis.fetch = async () => new Response(
    JSON.stringify({ error: 'session_not_found', message: 'This chat session was not found.' }),
    { status: 404, headers: { 'content-type': 'application/json' } },
  );
  await assert.rejects(streamChat('gone', 'hi'), (err) => {
    assert.equal(err.status, 404);
    assert.equal(err.error, 'session_not_found');
    assert.equal(err.message, 'This chat session was not found.');
    return true;
  });
});

await test('streamChat: a stream cut before finish resolves finished=false', async () => {
  globalThis.fetch = async () => streamResponse([frame(EVENTS[0]), frame(EVENTS[5])]);
  const got = [];
  const result = await streamChat('s', 'm', { onEvent: (e) => got.push(e) });
  assert.equal(result.finished, false);
  assert.deepEqual(got, [EVENTS[0], EVENTS[5]]);
});

await test('streamChat: abort mid-stream rejects with AbortError (a clean stop)', async () => {
  const controller = new AbortController();
  globalThis.fetch = async (url, init) => streamResponse([frame(EVENTS[0])], { hang: true, signal: init.signal });
  const got = [];
  const run = streamChat('s', 'm', {
    signal: controller.signal,
    onEvent: (e) => { got.push(e); setTimeout(() => controller.abort(), 0); },
  });
  await assert.rejects(run, (err) => err.name === 'AbortError');
  assert.deepEqual(got, [EVENTS[0]]);
});

await test('streamChat: a network failure rejects with status 0', async () => {
  globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
  await assert.rejects(streamChat('s', 'm'), (err) => err.status === 0 && err.error === 'network_error');
});

globalThis.fetch = realFetch;
console.log(`\n${checks} SSE checks passed`);
