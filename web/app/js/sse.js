/* The chat stream: POST /api/chat answered with `text/event-stream`, read through fetch.
 *
 * (EventSource cannot POST.) The parser follows the SSE framing rules the server uses: events are
 * separated by a blank line, `\r\n` and `\r` count as line breaks, lines starting with `:` are
 * comments (the server's `: keep-alive`), and several `data:` lines in one event are joined with
 * `\n`. Each event's data is one JSON object. Chunks may split anywhere: inside a line, between
 * `\r` and `\n`, or inside a multi-byte UTF-8 character (the decoder runs in stream mode).
 *
 * `createSseParser` and `createSseDecoder` are pure (no DOM, no fetch) so
 * `scripts/test_sse_parser.mjs` can feed them random splits under Node.
 */
import { readError, requestHeaders, isAbort, networkError, ApiError } from './api.js';
import { t } from './i18n.js';

/* Text in, events out: `push(text)` any number of times, then `end()`. `onEvent(obj)` is called
 * for every complete event whose data parses as a JSON object; anything else is skipped. */
export function createSseParser(onEvent) {
  let buffer = '';
  let carriageReturn = false; // the previous chunk ended with '\r'; the next may start with '\n'

  function dispatch(block) {
    const data = [];
    for (const line of block.split('\n')) {
      if (line === '' || line[0] === ':') continue;
      const colon = line.indexOf(':');
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? '' : line.slice(colon + 1);
      if (value[0] === ' ') value = value.slice(1);
      if (field === 'data') data.push(value);
      // `event:`, `id:` and `retry:` are not used by this server and are ignored.
    }
    if (data.length === 0) return;
    let event;
    try {
      event = JSON.parse(data.join('\n'));
    } catch {
      return;
    }
    if (event && typeof event === 'object' && !Array.isArray(event)) onEvent(event);
  }

  function drain() {
    let cut;
    while ((cut = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      dispatch(block);
    }
  }

  return {
    push(text) {
      if (!text) return;
      let chunk = carriageReturn ? `\r${text}` : text;
      carriageReturn = chunk.endsWith('\r');
      if (carriageReturn) chunk = chunk.slice(0, -1);
      buffer += chunk.replace(/\r\n?/g, '\n');
      drain();
    },
    end() {
      if (carriageReturn) {
        buffer += '\n';
        carriageReturn = false;
      }
      drain();
      // A final event without its blank line: dispatch it if it is complete JSON.
      if (buffer.trim()) dispatch(buffer);
      buffer = '';
    },
  };
}

/* Bytes in, events out: a streaming UTF-8 decoder in front of the parser. */
export function createSseDecoder(onEvent) {
  const decoder = new TextDecoder('utf-8');
  const parser = createSseParser(onEvent);
  return {
    push(bytes) {
      parser.push(decoder.decode(bytes, { stream: true }));
    },
    end() {
      parser.push(decoder.decode());
      parser.end();
    },
  };
}

/* Stream one chat turn. Resolves `{finished}` (whether a `finish` event arrived) when the stream
 * ends. `finish` is the last event delivered: a duplicate, or anything after it, is dropped. Rejects with an ApiError for an HTTP error (e.g. a 404 JSON body for an unknown session),
 * a network failure or a non-stream response, and with the AbortError when `signal` aborts: the
 * caller treats that as a clean Stop. */
export async function streamChat(sessionId, message, { onEvent, signal } = {}) {
  const handler = typeof onEvent === 'function' ? onEvent : () => {};
  let res;
  try {
    res = await fetch('/api/chat', {
      method: 'POST',
      headers: requestHeaders('POST', { Accept: 'text/event-stream' }),
      body: JSON.stringify({ session_id: sessionId, message }),
      credentials: 'same-origin',
      cache: 'no-store',
      signal,
    });
  } catch (err) {
    if (isAbort(err)) throw err;
    throw networkError();
  }
  if (!res.ok) throw await readError(res);
  const type = res.headers.get('content-type') || '';
  if (!type.includes('text/event-stream') || !res.body) {
    throw new ApiError({ status: res.status, error: 'not_a_stream', message: t('err_not_stream') });
  }

  let finished = false;
  const decoder = createSseDecoder((event) => {
    if (finished) return; // the first finish ends the turn: nothing after it is delivered
    if (event.type === 'finish') finished = true;
    handler(event);
  });
  const reader = res.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      decoder.push(value);
    }
    decoder.end();
  } catch (err) {
    if (isAbort(err) || signal?.aborted) throw err;
    throw networkError();
  } finally {
    // cancel() rejects on an aborted body; swallow that so it never surfaces as unhandled.
    try {
      const pending = reader.cancel();
      if (pending && typeof pending.catch === 'function') pending.catch(() => {});
    } catch {
      /* already closed */
    }
  }
  return { finished };
}
