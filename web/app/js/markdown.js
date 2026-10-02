/* Safe markdown for model output: streaming-markdown builds DOM nodes, never HTML strings.
 *
 * Model text can carry content copied from web pages (prompt injection), so every message goes
 * through the same pipeline, streamed or loaded from history:
 *  1. Before each write, the chunk is appended to the message's accumulated text and DOMPurify
 *     sanitizes that text. If DOMPurify removed anything, markdown rendering stops for this message
 *     and the rest is shown as plain text (textContent). This is Chrome's "render LLM responses"
 *     pattern: https://developer.chrome.com/docs/ai/render-llm-responses
 *  2. The renderer never creates <img> (an auto-loaded image URL can exfiltrate data) and only
 *     keeps absolute http(s) URLs.
 *  3. After every write and at the end, the rendered subtree is walked: unknown elements become
 *     text, `on*`/`style` attributes go, links keep only http(s) hrefs and get
 *     rel="noopener noreferrer nofollow", target="_blank", referrerpolicy="no-referrer" and a
 *     title naming the destination host (link text can claim another site), any <img> is replaced
 *     by a text link, and tables/code blocks get a copy button.
 * No DOM is touched at import time, so Node can import the module: scripts/test_markdown.mjs covers
 * the pure helpers and scripts/test_markdown_dom.mjs runs the renderer and the walk on a fake DOM.
 */
import * as smd from '../vendor/streaming-markdown.min.js';
import { bindText, bindAttr } from './i18n.js';
import { icon } from './icons.js';

const LINK_REL = 'noopener noreferrer nofollow';

/* Elements the renderer may create, plus the wrappers and buttons added here. Anything else is
 * replaced by its text. */
const ALLOWED = new Set([
  'p', 'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'em', 'strong', 's', 'code',
  'pre', 'a', 'ul', 'ol', 'li', 'input', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
  'equation-block', 'equation-inline', 'div', 'span', 'button', 'svg', 'path',
]);

/* An absolute http(s) URL without embedded credentials, normalised; otherwise null. */
export function safeHttpUrl(value) {
  if (typeof value !== 'string') return null;
  const text = value.trim();
  if (!text || text.length > 2048) return null;
  let url;
  try {
    url = new URL(text);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  if (url.username || url.password) return null;
  return url.href;
}

// --- writing: keep "$" literal outside code ------------------------------------------------------
//
// streaming-markdown reads "$…$" as inline math, so "Buy $VOO and $VTI" would lose its dollar
// signs and swallow the text between them (even across paragraphs). Finance text is full of "$",
// so a "$" that would open math is written as "\$", which the parser renders as a literal "$".
// Everywhere else (inline, fenced and indented code, link URLs, after a backslash) it is written
// as it is, because there a backslash would show.

/* Would "$" open math in the parser's current state? Asked of a throwaway copy of the parser with
 * a renderer that only watches for an equation token, so the real parser and the DOM are not
 * touched and no guess about its internal states is needed. If the copy cannot be made or driven
 * (a streaming-markdown whose internals differ), the answer is "no": the "$" is written raw and
 * the worst case is the library's own math rendering, never lost text. */
function dollarOpensMath(parser) {
  try {
    let opened = false;
    const probe = { ...parser };
    for (const [key, value] of Object.entries(probe)) {
      if (ArrayBuffer.isView(value)) probe[key] = value.slice();
    }
    probe.renderer = {
      data: null,
      add_token(_, type) {
        if (type === smd.EQUATION_INLINE || type === smd.EQUATION_BLOCK) opened = true;
      },
      end_token() {},
      add_text() {},
      set_attr() {},
    };
    smd.parser_write(probe, '$a');
    return opened;
  } catch {
    return false;
  }
}

/* Write `chunk` into a streaming-markdown `parser`, keeping every "$" a literal dollar sign. */
export function writeMarkdownChunk(parser, chunk) {
  let start = 0;
  for (let at = chunk.indexOf('$'); at !== -1; at = chunk.indexOf('$', start)) {
    if (at > start) smd.parser_write(parser, chunk.slice(start, at));
    smd.parser_write(parser, dollarOpensMath(parser) ? '\\$' : '$');
    start = at + 1;
  }
  if (start < chunk.length) smd.parser_write(parser, chunk.slice(start));
}

// --- the renderer --------------------------------------------------------------------------------

function createRenderer(root) {
  const base = smd.default_renderer(root);
  return {
    data: base.data,
    add_token(data, type) {
      if (type === smd.IMAGE) {
        // A placeholder link instead of <img>: its text is the alt text; finished in sanitizeTree.
        const parent = data.nodes[data.index];
        const el = document.createElement('a');
        el.className = 'md-image';
        el.setAttribute('data-md-image', '');
        data.nodes[++data.index] = parent.appendChild(el);
        return;
      }
      base.add_token(data, type);
    },
    end_token(data) {
      const node = data.nodes[data.index];
      if (node && node.nodeName === 'A') node.setAttribute('data-md-done', '');
      base.end_token(data);
    },
    add_text: base.add_text,
    set_attr(data, type, value) {
      const node = data.nodes[data.index];
      if (!node || !node.setAttribute) return;
      if (type === smd.HREF || type === smd.SRC) {
        const url = safeHttpUrl(value);
        if (url) node.setAttribute(node.hasAttribute('data-md-image') ? 'data-md-url' : 'href', url);
        return;
      }
      if (type === smd.LANG) {
        const lang = String(value).trim().toLowerCase();
        if (/^[a-z0-9+#._-]{1,24}$/.test(lang)) node.setAttribute('data-lang', lang);
        return;
      }
      base.set_attr(data, type, value); // CHECKED and START (list numbering)
    },
  };
}

// --- the walk ------------------------------------------------------------------------------------

const postProcessors = [];
const streams = new WeakMap(); // rendered root -> {context, ended}

/* Task 4.2 hook: `fn(root, info)` runs after every sanitize pass (e.g. turning `.md-cite` spans
 * into source chips). `info` is `{final, ...context}`, where `context` is what the caller of
 * `createMarkdownStream` passed: for a chat turn that is `{turn}`, and `turn.id` / `turn.sessionId`
 * say whose sources to look up. `final` is true once the message is complete. It runs many times
 * per message, so it must be idempotent, and it must build DOM with textContent only. */
export function addPostProcessor(fn) {
  if (typeof fn === 'function') postProcessors.push(fn);
}

function runPostProcessors(root, final) {
  const stream = streams.get(root);
  const info = { ...(stream ? stream.context : null), final };
  for (const fn of postProcessors) {
    try {
      fn(root, info);
    } catch (err) {
      console.error('markdown post-processor failed', err);
    }
  }
}

/* Run the post-processors again for a message that is already on screen: call it when what they
 * depend on arrives later (a turn's sources after its text, as in history replay). `root` is the
 * `.md` element (`stream.root`); a root this module did not render is ignored. */
export function refreshMarkdown(root) {
  const stream = root ? streams.get(root) : null;
  if (stream) runPostProcessors(root, stream.ended);
}

function imageLink(alt, url) {
  const link = document.createElement('a');
  link.className = 'md-image';
  link.setAttribute('data-md-image', '');
  link.setAttribute('data-md-final', '');
  const text = (alt || '').trim();
  if (text) bindText(link, 'md_image', { alt: text });
  else bindText(link, 'md_image_bare');
  if (url) link.setAttribute('href', url);
  finishLink(link);
  return link;
}

function scrubAttributes(el) {
  for (const attr of [...el.attributes]) {
    const name = attr.name.toLowerCase();
    if (name.startsWith('on') || name === 'style' || name === 'srcset' || name === 'src'
      || name === 'formaction' || name === 'action' || name === 'xlink:href'
      || (name === 'href' && el.localName !== 'a')) {
      el.removeAttribute(attr.name);
    }
  }
}

function finishLink(a) {
  const href = a.getAttribute('href');
  if (href !== null) {
    const url = safeHttpUrl(href);
    if (url) {
      if (url !== href) a.setAttribute('href', url);
    } else {
      a.removeAttribute('href');
    }
  }
  if (a.hasAttribute('href')) {
    a.setAttribute('rel', LINK_REL);
    a.setAttribute('target', '_blank');
    a.setAttribute('referrerpolicy', 'no-referrer');
    // The text of a link can name another site: the tooltip always shows the real destination.
    a.setAttribute('title', new URL(a.getAttribute('href')).host);
  } else {
    a.removeAttribute('target');
    a.removeAttribute('rel');
    a.removeAttribute('title');
  }
}

const CITATION = /^E\d{1,4}(?:\s*,\s*E\d{1,4})*$/;

/* A finished "[text]" without a usable URL (e.g. the [E1] evidence markers, or a link whose URL
 * was refused): show the brackets again, as a citation span or as plain text. */
function unlink(a) {
  const text = a.textContent;
  if (CITATION.test(text.trim())) {
    const span = document.createElement('span');
    span.className = 'md-cite';
    span.setAttribute('data-cite', text.replace(/\s+/g, ''));
    span.textContent = `[${text.trim()}]`;
    a.replaceWith(span);
  } else {
    a.replaceWith(document.createTextNode(`[${text}]`));
  }
}

function copyButton(kind) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'md-copy';
  btn.setAttribute('data-copy', kind);
  bindAttr(btn, 'aria-label', kind === 'table' ? 'copy_table' : 'copy_code');
  btn.appendChild(icon('copy', { size: 16 }));
  const label = document.createElement('span');
  label.className = 'md-copy-label';
  bindText(label, 'copy');
  btn.appendChild(label);
  return btn;
}

function wrapBlock(el, kind, labelText, labelKey) {
  const box = document.createElement('div');
  box.className = kind === 'table' ? 'md-block md-table' : 'md-block md-code';
  const head = document.createElement('div');
  head.className = 'md-block-head';
  const label = document.createElement('span');
  label.className = 'md-block-label';
  if (labelText) label.textContent = labelText;
  else bindText(label, labelKey);
  head.append(label, copyButton(kind));
  const body = document.createElement('div');
  body.className = 'md-block-body';
  el.replaceWith(box);
  body.appendChild(el);
  box.append(head, body);
  if (kind === 'table') {
    body.tabIndex = 0; // scrollable region: reachable by keyboard
    body.setAttribute('role', 'region');
    bindAttr(body, 'aria-label', 'table_label');
  }
}

/* Make the rendered subtree safe; see the module comment. `final` also finishes elements whose
 * token never closed (the stream ended). */
export function sanitizeTree(root, { final = false } = {}) {
  for (const el of [...root.querySelectorAll('*')]) {
    if (!root.contains(el)) continue; // already removed with a replaced ancestor
    if (el.localName === 'img') {
      el.replaceWith(imageLink(el.getAttribute('alt'), safeHttpUrl(el.getAttribute('src') || '')));
      continue;
    }
    if (!ALLOWED.has(el.localName)) {
      el.replaceWith(document.createTextNode(el.textContent));
      continue;
    }
    scrubAttributes(el);
    if (el.localName === 'input') {
      el.setAttribute('type', 'checkbox');
      el.setAttribute('disabled', '');
    }
  }
  for (const a of [...root.querySelectorAll('a')]) {
    const done = final || a.hasAttribute('data-md-done');
    if (a.hasAttribute('data-md-image')) {
      if (a.hasAttribute('data-md-final')) {
        finishLink(a);
      } else if (done) {
        a.replaceWith(imageLink(a.textContent, safeHttpUrl(a.getAttribute('data-md-url') || '')));
      }
      continue;
    }
    finishLink(a);
    if (done && !a.hasAttribute('href')) unlink(a);
  }
  for (const pre of root.querySelectorAll('pre')) {
    if (pre.parentElement?.classList.contains('md-block-body')) continue;
    const lang = pre.querySelector('code')?.getAttribute('data-lang');
    wrapBlock(pre, 'code', lang || '', 'code_label');
  }
  for (const table of root.querySelectorAll('table')) {
    if (table.parentElement?.classList.contains('md-block-body')) continue;
    wrapBlock(table, 'table', '', 'table_label');
  }
  if (final) root.normalize();
  runPostProcessors(root, final);
}

/* The purify step's verdict on the accumulated text: true when nothing would be removed. Fails
 * closed: without a sanitizer every message is shown as plain text. */
function purifyAccepts(purify, text) {
  if (!purify || typeof purify.sanitize !== 'function') return false;
  purify.sanitize(text);
  return Array.isArray(purify.removed) && purify.removed.length === 0;
}

/* A streaming markdown view appended to `container`. `write(chunk)` any number of times, then
 * `end()`. Once HTML is detected, the rest of the message goes into a plain-text block.
 * Options: `purify` is the sanitizer used for the gate (DOMPurify by default; tests inject their
 * own); `context` is handed to the post-processors (a chat turn passes `{turn}`). */
export function createMarkdownStream(container, { purify = globalThis.DOMPurify, context = null } = {}) {
  const root = document.createElement('div');
  root.className = 'md';
  container.appendChild(root);
  const parser = smd.parser(createRenderer(root));
  const stream = { context, ended: false };
  streams.set(root, stream);
  let source = '';
  let plain = null;

  function switchToPlain(rest) {
    try {
      smd.parser_end(parser);
    } catch {
      /* the parser is abandoned either way */
    }
    sanitizeTree(root, { final: true });
    const note = document.createElement('p');
    note.className = 'md-note';
    bindText(note, 'md_unsafe_note');
    const block = document.createElement('div');
    block.className = 'md-plain';
    plain = document.createTextNode(rest);
    block.appendChild(plain);
    container.append(note, block);
  }

  return {
    root,
    get text() { return source; },
    get isPlain() { return plain !== null; },
    write(chunk) {
      if (stream.ended || typeof chunk !== 'string' || chunk === '') return;
      source += chunk;
      if (plain) {
        plain.appendData(chunk);
        return;
      }
      if (!purifyAccepts(purify, source)) {
        switchToPlain(chunk);
        return;
      }
      writeMarkdownChunk(parser, chunk);
      sanitizeTree(root);
    },
    end() {
      if (stream.ended) return;
      stream.ended = true;
      if (!plain) {
        smd.parser_end(parser);
        sanitizeTree(root, { final: true });
      }
    },
  };
}

/* History: the whole text in one write, through the same pipeline. */
export function renderMarkdown(container, text, options) {
  const view = createMarkdownStream(container, options);
  view.write(String(text ?? ''));
  view.end();
  return view;
}
