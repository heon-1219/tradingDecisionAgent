/* One conversation turn as DOM, plus the shell's minimal stream-event handlers.
 *
 * An assistant turn has named slots, top to bottom, so Task 4.2's components know where to render:
 *   thinking  the collapsible "Thoughts" block (reasoning-delta)
 *   activity  one row per tool call (4.2: ToolCards)
 *   body      the answer, streamed markdown (text-delta)
 *   extras    empty in 4.1 (4.2: sources, mandate card, trade ticket)
 *   status    the "Working…" indicator, inline errors, the "Stopped" note
 * Empty slots take no space. Model and tool text only ever reaches the DOM through textContent or
 * the markdown pipeline; the element receiving streamed text is never an aria-live region.
 */
import { bindText } from './i18n.js';
import { icon } from './icons.js';
import { createMarkdownStream } from './markdown.js';
import { registerEvent } from './registry.js';

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

const text = (value) => (typeof value === 'string' ? value : '');

/* The user's message: plain text with its line breaks (CSS pre-wrap), never markdown or HTML. */
export function createUserMessage(content) {
  const article = el('article', 'turn turn-user');
  const heading = el('h2', 'sr-only');
  bindText(heading, 'you');
  const bubble = el('div', 'bubble');
  bubble.textContent = text(content);
  article.append(heading, bubble);
  return article;
}

export class Turn {
  /* `replay` is true for a turn rebuilt from history: no live indicators, no animation state. */
  constructor({ replay = false } = {}) {
    this.replay = replay;
    this.id = null;          // the server's messageId / turn_id
    this.finish = null;      // the `finish` event, once seen
    this.hadError = false;
    this.closed = false;
    this.outcome = null;     // 'finish' | 'error' | 'stopped' | 'cut' | 'history'
    this.tools = new Map();  // toolCallId -> {row, label, name}
    this.data = {};          // scratch space for components (4.2)
    this.md = null;
    this.thinking = null;
    this.pending = null;
    this.newRound = false;   // a tool ran since the last reasoning text

    this.el = el('article', 'turn turn-assistant');
    const mark = el('div', 'turn-mark');
    mark.setAttribute('aria-hidden', 'true');
    mark.appendChild(icon('spark', { size: 14 }));
    const main = el('div', 'turn-main');
    const heading = el('h2', 'sr-only');
    bindText(heading, 'assistant');
    this.slots = {
      thinking: el('div', 'turn-slot turn-thinking'),
      activity: el('ul', 'turn-slot turn-activity'),
      body: el('div', 'turn-slot turn-body'),
      extras: el('div', 'turn-slot turn-extras'),
      status: el('div', 'turn-slot turn-status'),
    };
    main.append(heading, ...Object.values(this.slots));
    this.el.append(mark, main);
  }

  /* The "Working…" line shown until the answer starts. */
  setPending(on) {
    if (on && !this.pending && !this.closed) {
      this.pending = el('div', 'turn-pending');
      const label = el('span', 'shimmer');
      bindText(label, 'pending');
      this.pending.appendChild(label);
      this.slots.status.prepend(this.pending);
    } else if (!on && this.pending) {
      this.pending.remove();
      this.pending = null;
    }
  }

  markdown() {
    if (!this.md) this.md = createMarkdownStream(this.slots.body);
    return this.md;
  }

  appendText(delta) {
    if (this.closed || !delta) return;
    this.setPending(false);
    this.settleThinking();
    this.markdown().write(delta);
  }

  appendReasoning(delta) {
    if (this.closed || !delta) return;
    if (!this.thinking) {
      const details = el('details', 'thinking');
      const summary = el('summary');
      const label = el('span', this.replay ? 'thinking-label' : 'thinking-label shimmer');
      bindText(label, this.replay ? 'thinking_done' : 'thinking_live');
      summary.append(label, icon('chevron', { size: 16, className: 'chev' }));
      const body = el('div', 'thinking-body');
      const node = document.createTextNode('');
      body.appendChild(node);
      details.append(summary, body);
      this.slots.thinking.appendChild(details);
      this.thinking = { details, label, node };
    }
    // The agent sends one reasoning text per tool round: separate rounds with a blank line.
    if (this.newRound && this.thinking.node.length > 0) this.thinking.node.appendData('\n\n');
    this.newRound = false;
    this.thinking.node.appendData(delta);
  }

  settleThinking() {
    if (!this.thinking) return;
    this.thinking.label.classList.remove('shimmer');
    bindText(this.thinking.label, 'thinking_done');
  }

  toolStart(id, name) {
    this.newRound = true;
    const key = id || `tool-${this.tools.size + 1}`;
    if (this.tools.has(key)) return this.tools.get(key);
    const row = el('li', 'tool-row');
    row.setAttribute('data-status', 'running');
    const spinner = el('span', 'tool-spinner');
    spinner.setAttribute('aria-hidden', 'true');
    const label = el('span', 'tool-label');
    const entry = { row, label, name: name || 'tool' };
    bindText(label, 'tool_running', { name: entry.name });
    row.append(spinner, label);
    this.slots.activity.appendChild(row);
    this.tools.set(key, entry);
    return entry;
  }

  toolEnd(id, name, isError) {
    const entry = this.tools.get(id) || this.toolStart(id, name);
    entry.row.setAttribute('data-status', isError ? 'error' : 'done');
    bindText(entry.label, isError ? 'tool_error' : 'tool_done', { name: entry.name });
  }

  showError(message) {
    this.hadError = true;
    this.setPending(false);
    const box = el('div', 'turn-error');
    const body = el('span', 'turn-error-text');
    if (message) body.textContent = message;
    else bindText(body, 'err_generic');
    const holder = el('span', 'icon');
    holder.appendChild(icon('alert', { size: 18 }));
    box.append(holder, body);
    this.slots.status.appendChild(box);
  }

  /* End the turn (idempotent): flush the markdown, settle every live indicator. */
  close(outcome = 'finish') {
    if (this.closed) return;
    this.setPending(false);
    this.closed = true;
    this.outcome = outcome;
    if (this.md) this.md.end();
    this.settleThinking();
    for (const entry of this.tools.values()) {
      if (entry.row.getAttribute('data-status') === 'running') {
        entry.row.setAttribute('data-status', 'stopped');
        bindText(entry.label, 'tool_stopped', { name: entry.name });
      }
    }
    if (outcome === 'stopped') {
      const note = el('p', 'turn-note');
      bindText(note, 'stopped_note');
      this.slots.status.appendChild(note);
    }
    this.el.setAttribute('data-outcome', outcome);
  }
}

/* The shell's handlers. Task 4.2 replaces the tool and thinking ones through `registerEvent`. */
export function registerDefaultHandlers() {
  const fallback = { fallback: true };
  registerEvent('start', (event, turn) => {
    if (typeof event.messageId === 'string') turn.id = event.messageId;
  }, fallback);
  registerEvent('reasoning-delta', (event, turn) => turn.appendReasoning(text(event.delta)), fallback);
  registerEvent('text-delta', (event, turn) => turn.appendText(text(event.delta)), fallback);
  registerEvent('tool-input-available', (event, turn) => {
    turn.toolStart(text(event.toolCallId), text(event.toolName));
  }, fallback);
  registerEvent('tool-output-available', (event, turn) => {
    turn.toolEnd(text(event.toolCallId), text(event.toolName), Boolean(event.isError));
  }, fallback);
  registerEvent('error', (event, turn) => turn.showError(text(event.message)), fallback);
  registerEvent('finish', (event, turn) => {
    turn.finish = event;
  }, fallback);
}
