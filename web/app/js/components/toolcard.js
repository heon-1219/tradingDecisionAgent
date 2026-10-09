/* Tool cards: one per tool call, in the turn's `activity` slot (replacing the shell's one-line rows).
 *
 * A card shows the tool's family (icon and label), its name, a status badge with text (Running,
 * Completed, Error, Stopped), the duration measured here, and a chip with the number of sources it
 * returned. Arguments and output are collapsed by default and are always rendered as text: tool
 * output is untrusted. `<untrusted source="E3">…</untrusted>` blocks in an output are shown as text
 * under an "External content" label.
 *
 * In history (`turn.replay`) the stored events carry only the tool's name, id and error flag, so a
 * replayed card is the compact row without a body.
 */
import { bindText } from '../i18n.js';
import { replaceEvent } from '../registry.js';
import { on } from '../state.js';
import { badge, el, iconSpan, label, str } from './dom.js';
import { icon } from '../icons.js';

export const FAMILIES = {
  account: { icon: 'wallet', key: 'family_account' },
  market: { icon: 'chart', key: 'family_market' },
  research: { icon: 'search', key: 'family_research' },
  knowledge: { icon: 'book', key: 'family_knowledge' },
  mandate: { icon: 'shield', key: 'family_mandate' },
  trading: { icon: 'ticket', key: 'family_trading' },
  other: { icon: 'tool', key: 'family_other' },
};

const TOOL_FAMILY = {
  get_account: 'account', get_positions: 'account', list_orders: 'account',
  get_market_status: 'market', lookup_asset: 'market', get_quotes: 'market', price_metrics: 'market',
  get_movers: 'market', screen_us_etfs: 'market',
  search_news: 'research', web_search: 'research', fetch_source: 'research', sec_filings: 'research',
  sec_financial_facts: 'research', sec_filing_passages: 'research', grounded_web_answer: 'research',
  ask_product_database: 'knowledge', search_products: 'knowledge', graph_neighbors: 'knowledge',
  draft_mandate: 'mandate',
  propose_trades: 'trading',
};

export function familyOf(toolName) {
  return Object.prototype.hasOwnProperty.call(TOOL_FAMILY, toolName) ? TOOL_FAMILY[toolName] : 'other';
}

const STATUS = {
  running: ['tool_status_running', 'accent'],
  done: ['tool_status_done', 'ok'],
  error: ['tool_status_error', 'bad'],
  stopped: ['tool_status_stopped', 'muted'],
};

const UNTRUSTED_RE = /<untrusted source="([^"]*)">([\s\S]*?)<\/untrusted>/g;

/* Split a tool output into plain text and external (untrusted) blocks, in order. */
export function splitUntrusted(output) {
  const text = str(output);
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(UNTRUSTED_RE)) {
    if (match.index > last) parts.push({ kind: 'text', text: text.slice(last, match.index) });
    parts.push({ kind: 'external', source: match[1], text: match[2] });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push({ kind: 'text', text: text.slice(last) });
  return parts.filter((part) => part.kind === 'external' || part.text.trim() !== '');
}

/* 420 -> "0.4 s", 12300 -> "12 s". A duration, not money. */
export function durationText(ms) {
  const seconds = Math.max(0, ms) / 1000;
  return `${seconds < 10 ? seconds.toFixed(1) : String(Math.round(seconds))} s`; // not money: seconds
}

function prettyInput(input) {
  if (input === undefined || input === null) return '';
  try {
    const text = JSON.stringify(input, null, 2);
    return text === '{}' ? '' : text;
  } catch {
    return '';
  }
}

function setStatus(card, status) {
  const [key, tone] = STATUS[status] || STATUS.done;
  card.status = status;
  card.root.setAttribute('data-status', status);
  card.badge.setAttribute('data-tone', tone);
  bindText(card.badge, key);
}

function head(card, name, family) {
  const spec = FAMILIES[family];
  const mark = iconSpan(spec.icon, 16, 'icon toolcard-icon');
  const title = el('span', 'toolcard-name', name);
  const kind = label('span', 'toolcard-family', spec.key);
  const meta = el('span', 'toolcard-meta');
  card.meta = meta;
  card.badge = badge('accent', 'tool_status_running');
  card.badge.classList.add('toolcard-status');
  const spinner = el('span', 'tool-spinner');
  spinner.setAttribute('aria-hidden', 'true');
  return [mark, title, kind, meta, spinner, card.badge];
}

function createCard(turn, id, name, input) {
  const family = familyOf(name);
  const root = el('li', 'toolcard');
  root.setAttribute('data-family', family);
  const card = { root, name, started: performance.now(), status: 'running' };
  if (turn.replay) {
    // History: no arguments or output were stored.
    root.classList.add('is-compact');
    const row = el('div', 'toolcard-row');
    row.append(...head(card, name, family));
    root.appendChild(row);
  } else {
    const details = el('details', 'toolcard-details');
    const summary = el('summary', 'toolcard-row');
    summary.append(...head(card, name, family), icon('chevron', { size: 16, className: 'chev' }));
    const body = el('div', 'toolcard-body');
    const args = prettyInput(input);
    if (args) {
      body.append(label('h4', 'toolcard-h', 'tool_arguments'), el('pre', 'toolcard-pre', args));
    }
    card.output = el('div', 'toolcard-output');
    body.appendChild(card.output);
    details.append(summary, body);
    root.appendChild(details);
  }
  setStatus(card, turn.replay ? 'done' : 'running');
  turn.slots.activity.appendChild(root);
  turn.data.toolcards.set(id, card);
  return card;
}

function cardsOf(turn) {
  if (!turn.data.toolcards) turn.data.toolcards = new Map();
  return turn.data.toolcards;
}

function fillOutput(card, event) {
  if (!card.output) return;
  const parts = splitUntrusted(event.output);
  if (parts.length === 0) return;
  const nodes = [label('h4', 'toolcard-h', 'tool_output')];
  for (const part of parts) {
    if (part.kind === 'text') {
      nodes.push(el('pre', 'toolcard-pre', part.text.trim()));
      continue;
    }
    const box = el('div', 'toolcard-external');
    const tag = el('p', 'toolcard-external-label');
    tag.appendChild(label('span', '', 'tool_external'));
    if (part.source) tag.appendChild(el('span', 'toolcard-external-id', part.source));
    box.append(tag, el('pre', 'toolcard-pre', part.text.trim()));
    nodes.push(box);
  }
  card.output.replaceChildren(...nodes);
}

function onInput(event, turn) {
  const cards = cardsOf(turn);
  const id = str(event.toolCallId) || `tool-${cards.size + 1}`;
  if (cards.has(id)) return;
  createCard(turn, id, str(event.toolName) || 'tool', event.input);
}

function onOutput(event, turn) {
  const cards = cardsOf(turn);
  const id = str(event.toolCallId);
  const card = cards.get(id) || createCard(turn, id || `tool-${cards.size + 1}`, str(event.toolName) || 'tool');
  setStatus(card, event.isError ? 'error' : 'done');
  if (turn.replay) return;
  const sources = Array.isArray(event.sources) ? event.sources.length : 0;
  if (sources > 0) {
    card.meta.appendChild(label('span', 'toolcard-chip', sources === 1 ? 'tool_sources_one' : 'tool_sources',
      { n: sources }));
  }
  card.meta.appendChild(el('span', 'toolcard-time', durationText(performance.now() - card.started)));
  fillOutput(card, event);
}

replaceEvent('tool-input-available', onInput);
replaceEvent('tool-output-available', onOutput);

// A turn that was stopped or failed leaves its unfinished calls as "Stopped".
on('turn:end', ({ turn }) => {
  for (const card of cardsOf(turn).values()) {
    if (card.status === 'running') setStatus(card, 'stopped');
  }
});
