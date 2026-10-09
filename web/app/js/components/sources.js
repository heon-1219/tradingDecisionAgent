/* Sources: citation chips in the answer, the "Sources" row under it, and the source popover.
 *
 * The answer's `[E1]` markers arrive from the markdown pipeline as
 * `<span class="md-cite" data-cite="E1,E2">`. A post-processor turns each id into a chip: a button
 * that opens the source popover when the source is known, a plain muted chip when it is not. The
 * same chips are used for a ticket's evidence.
 *
 * Source details come from `GET /api/sources/{session_id}` and are cached per session. They are
 * fetched when a chat is opened and after every turn; during a turn the chips already resolve from
 * what the tool results carried (title, publisher, tier and URL, without the snippet). Everything
 * in a source record is untrusted text and is rendered with textContent; its URL becomes a link
 * only if it is an absolute http(s) URL.
 */
import { request } from '../api.js';
import { bindAttr } from '../i18n.js';
import { addPostProcessor } from '../markdown.js';
import { registerEvent } from '../registry.js';
import { emit, on, state } from '../state.js';
import { badge, el, externalLink, extraSection, hostOf, iconSpan, label, str } from './dom.js';
import { initPopovers } from './popover.js';

const SOURCE_ID = /^E\d{1,4}$/;
const TIERS = {
  1: ['tier_1', 'ok'],
  2: ['tier_2', 'accent'],
  3: ['tier_3', 'muted'],
  4: ['tier_4', 'warn'],
};

const FIELDS = ['title', 'url', 'publisher', 'tier', 'published_at', 'connector', 'untrusted', 'snippet'];

const cache = new Map(); // session id -> Map(source id -> record)
const turns = new Set(); // turns on screen whose chips may need another pass

function sessionMap(sessionId) {
  if (!cache.has(sessionId)) cache.set(sessionId, new Map());
  return cache.get(sessionId);
}

export function getSource(sessionId, id) {
  return (sessionId && cache.get(sessionId)?.get(id)) || null;
}

/* Merge partial records (from a live tool result) without losing fields already known. */
export function seedSources(sessionId, records) {
  if (!sessionId || !Array.isArray(records)) return;
  const map = sessionMap(sessionId);
  for (const record of records) {
    const id = record && str(record.id || record.sourceId);
    if (!SOURCE_ID.test(id)) continue;
    const known = map.get(id) || { id };
    for (const key of FIELDS) {
      const value = record[key];
      if (value === null || value === undefined || value === '') continue;
      if (known[key] === undefined || known[key] === null || known[key] === '') known[key] = value;
    }
    map.set(id, known);
  }
}

/* Fetch a session's sources and replace its cache entry. Failures are quiet: the chips simply stay
 * as they are. Resolves true when the cache was updated. */
export async function loadSources(sessionId) {
  if (!sessionId) return false;
  let body;
  try {
    body = await request('GET', `/api/sources/${encodeURIComponent(sessionId)}`);
  } catch {
    return false;
  }
  const map = new Map();
  for (const record of Array.isArray(body?.sources) ? body.sources : []) {
    if (record && SOURCE_ID.test(str(record.id))) map.set(record.id, record);
  }
  // Keep anything seeded from a turn that the server has not listed (it should list everything).
  for (const [id, record] of cache.get(sessionId) || []) if (!map.has(id)) map.set(id, record);
  cache.set(sessionId, map);
  return true;
}

function refreshTurns(sessionId) {
  for (const turn of [...turns]) {
    if (!turn.el.isConnected) turns.delete(turn);
    else if (turn.sessionId === sessionId) turn.refreshMarkdown();
  }
}

// --- chips --------------------------------------------------------------------------------------

/* A chip for one source id: a button when the source is known, a plain muted chip otherwise. */
export function sourceChip(sessionId, id, className = 'cite-chip') {
  const known = Boolean(getSource(sessionId, id));
  const chip = el(known ? 'button' : 'span', className, id);
  if (known) {
    chip.type = 'button';
    chip.setAttribute('data-source', id);
    chip.setAttribute('aria-expanded', 'false');
    bindAttr(chip, 'aria-label', 'source_chip_label', { id });
  } else {
    chip.classList.add('is-unknown');
    bindAttr(chip, 'title', 'source_unknown');
  }
  return chip;
}

function citedIds(root) {
  const ids = [];
  for (const span of root.querySelectorAll('.md-cite[data-cite]')) {
    for (const id of span.getAttribute('data-cite').split(',')) {
      if (SOURCE_ID.test(id) && !ids.includes(id)) ids.push(id);
    }
  }
  return ids;
}

function chipCitations(root, sessionId) {
  for (const span of root.querySelectorAll('.md-cite[data-cite]')) {
    const ids = span.getAttribute('data-cite').split(',').filter((id) => SOURCE_ID.test(id));
    if (ids.length === 0) continue;
    const signature = ids.map((id) => `${id}${getSource(sessionId, id) ? '+' : '-'}`).join(',');
    if (span.getAttribute('data-chips') === signature) continue;
    span.setAttribute('data-chips', signature);
    span.replaceChildren(...ids.map((id) => sourceChip(sessionId, id)));
  }
}

function sourcesRow(turn, ids) {
  if (ids.length === 0) return;
  const box = extraSection(turn, 'sources');
  const signature = ids.map((id) => `${id}${getSource(turn.sessionId, id) ? '+' : '-'}`).join(',');
  if (box.getAttribute('data-sig') === signature) return;
  box.setAttribute('data-sig', signature);
  const list = el('ul', 'source-list');
  for (const id of ids) {
    const source = getSource(turn.sessionId, id);
    const item = el('li');
    if (!source) {
      const row = el('span', 'source-item is-unknown');
      row.append(el('span', 'source-item-id', id), label('span', 'source-item-title', 'source_unknown'));
      item.appendChild(row);
    } else {
      const row = el('button', 'source-item');
      row.type = 'button';
      row.setAttribute('data-source', id);
      row.setAttribute('aria-expanded', 'false');
      row.append(el('span', 'source-item-id', id), el('span', 'source-item-title', str(source.title) || id));
      const from = str(source.publisher) || hostOf(source.url);
      if (from) row.appendChild(el('span', 'source-item-from', from));
      item.appendChild(row);
    }
    list.appendChild(item);
  }
  box.replaceChildren(label('h3', 'extra-h', 'sources_heading'), list);
}

addPostProcessor((root, info) => {
  const turn = info && info.turn;
  if (!turn) return;
  turns.add(turn);
  chipCitations(root, turn.sessionId);
  if (info.final) sourcesRow(turn, citedIds(root));
});

// --- the popover ----------------------------------------------------------------------------------

function dateText(value) {
  const text = str(value);
  return /^\d{4}-\d{2}-\d{2}/.test(text) ? text.slice(0, 10) : text.slice(0, 40);
}

function buildPopover(anchor) {
  const id = anchor.getAttribute('data-source');
  const source = getSource(state.sessionId, id);
  if (!source) return null;
  const box = el('div', 'source-pop');
  const head = el('div', 'source-pop-head');
  head.appendChild(el('span', 'source-pop-id', id));
  const [tierKey, tone] = TIERS[source.tier] || ['tier_unknown', 'muted'];
  head.appendChild(badge(tone, tierKey));
  box.append(head, el('p', 'source-pop-title', str(source.title) || id));
  const meta = [str(source.publisher), dateText(source.published_at)].filter(Boolean);
  if (meta.length) box.appendChild(el('p', 'source-pop-meta', meta.join(' · ')));
  const snippet = str(source.snippet);
  if (snippet) {
    if (source.untrusted) box.appendChild(label('p', 'source-pop-note', 'tool_external'));
    box.appendChild(el('p', 'source-pop-snippet', snippet));
  }
  const link = externalLink(source.url, 'source-pop-link');
  if (link) {
    link.append(label('span', '', 'source_open'), el('span', 'source-pop-host', hostOf(source.url)),
      iconSpan('external', 14));
    box.appendChild(link);
  }
  return box;
}

initPopovers(document.getElementById('thread'), { selector: '[data-source]', build: buildPopover });

// --- keeping the cache fresh ------------------------------------------------------------------------

registerEvent('tool-output-available', (event, turn) => seedSources(turn.sessionId, event.sources));
registerEvent('source-url', (event, turn) => seedSources(turn.sessionId, [event]));

on('turn:end', async ({ turn, sessionId }) => {
  if (await loadSources(sessionId)) {
    turns.add(turn);
    refreshTurns(sessionId);
    emit('sources:loaded', sessionId);
  }
});

on('session:loaded', async (detail) => {
  const sessionId = detail?.session?.id;
  if (await loadSources(sessionId)) {
    refreshTurns(sessionId);
    emit('sources:loaded', sessionId);
  }
});
