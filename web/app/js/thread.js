/* The conversation view: the empty state, the list of turns, history, and follow-the-stream
 * scrolling (only while the reader is at the bottom; scrolling up stops the following). */
import { bindText, bindAttr } from './i18n.js';
import { icon } from './icons.js';
import { dispatch } from './registry.js';
import { createUserMessage, Turn } from './turn.js';

const SUGGESTIONS = [
  ['suggest_1', 'pie'],
  ['suggest_2', 'compare'],
  ['suggest_3', 'dollar'],
  ['suggest_4', 'scale'],
];

// Stored events that must not be replayed: the assistant row already carries the error text.
const NO_REPLAY = new Set(['error', 'start', 'finish', 'text-delta']);

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function buildEmpty(onSuggestion) {
  const section = el('section', 'empty');
  const greeting = el('h2', 'empty-greeting');
  bindText(greeting, 'empty_greeting');
  const lede = el('p', 'empty-lede');
  bindText(lede, 'empty_lede');
  const list = el('ul', 'suggestions');
  bindAttr(list, 'aria-label', 'suggestions_label');
  for (const [key, iconName] of SUGGESTIONS) {
    const item = el('li');
    const button = el('button', 'suggestion');
    button.type = 'button';
    const holder = el('span', 'icon');
    holder.appendChild(icon(iconName, { size: 18 }));
    const label = el('span', 'suggestion-text');
    bindText(label, key);
    button.append(holder, label);
    button.addEventListener('click', () => onSuggestion(label.textContent));
    item.appendChild(button);
    list.appendChild(item);
  }
  section.append(greeting, lede, list);
  return section;
}

export function createThread({ scroller, thread, onSuggestion }) {
  let pinned = true;

  const toBottom = () => {
    scroller.scrollTop = scroller.scrollHeight;
  };

  scroller.addEventListener('scroll', () => {
    pinned = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 96;
  }, { passive: true });

  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => {
      // The empty state reads from the top; only a conversation follows its newest content.
      if (pinned && !thread.classList.contains('is-empty')) toBottom();
    }).observe(thread);
  }

  function reset() {
    thread.replaceChildren();
    thread.classList.remove('is-empty');
  }

  function begin() {
    if (thread.classList.contains('is-empty') || thread.querySelector('.thread-loading')) reset();
  }

  return {
    showEmpty() {
      reset();
      thread.classList.add('is-empty');
      thread.appendChild(buildEmpty(onSuggestion));
      scroller.scrollTop = 0;
    },

    showLoading() {
      reset();
      const note = el('p', 'thread-loading');
      bindText(note, 'loading_session');
      thread.appendChild(note);
    },

    appendUser(content) {
      begin();
      const node = createUserMessage(content);
      thread.appendChild(node);
      return node;
    },

    appendTurn(turn) {
      begin();
      thread.appendChild(turn.el);
      return turn;
    },

    /* Keep the newest content in view if the reader has not scrolled away. */
    follow() {
      if (pinned) toBottom();
    },

    scrollToBottom() {
      pinned = true;
      toBottom();
    },

    /* Rebuild a stored conversation (`GET /api/sessions/{id}` -> messages). Assistant text goes
     * through the same markdown pipeline in one write; each stored turn summary (role `event`)
     * is replayed through the registry into its turn, so 4.2's components also draw history.
     * Every turn is closed after the replay, which runs the markdown post-processors once more:
     * by then the turn's replayed events (its sources) are known. */
    renderHistory(messages, { sessionId = null } = {}) {
      reset();
      const rows = Array.isArray(messages) ? messages : [];
      const turns = new Map();
      const all = [];
      let last = null;
      for (const row of rows) {
        if (!row || typeof row !== 'object') continue;
        const meta = row.meta && typeof row.meta === 'object' ? row.meta : {};
        const content = typeof row.content === 'string' ? row.content : '';
        if (row.role === 'user') {
          const node = createUserMessage(content);
          node.classList.add('is-static');
          thread.appendChild(node);
        } else if (row.role === 'assistant') {
          const turn = new Turn({ replay: true });
          turn.id = typeof meta.turn_id === 'string' ? meta.turn_id : null;
          turn.sessionId = sessionId;
          turn.data.meta = meta;
          turn.el.classList.add('is-static');
          if (meta.error) turn.showError(content);
          else turn.appendText(content);
          thread.appendChild(turn.el);
          if (turn.id) turns.set(turn.id, turn);
          all.push(turn);
          last = turn;
        } else if (row.role === 'event' && Array.isArray(meta.events)) {
          const turn = turns.get(meta.turn_id) || last;
          if (!turn) continue;
          for (const event of meta.events) {
            if (event && typeof event === 'object' && !NO_REPLAY.has(event.type)) dispatch(event, turn);
          }
        }
      }
      for (const turn of all) turn.close('history');
      if (!thread.firstChild) {
        this.showEmpty();
        return;
      }
      this.scrollToBottom();
    },
  };
}
