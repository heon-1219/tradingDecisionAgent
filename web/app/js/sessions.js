/* The rail's chat list: open a chat, rename it in place.
 * Titles are user text and server data: they only reach the DOM through textContent / value. */
import { api, isAbort } from './api.js';
import { bindText, bindAttr } from './i18n.js';
import { icon } from './icons.js';
import { isImeEvent } from './composer.js';
import { notify } from './notify.js';
import { state, setState } from './state.js';

const TITLE_MAX = 80;

export function initSessions({ list, status, onOpen, onRenamed }) {
  let renaming = false; // a rename box is open: do not rebuild the list under it

  function render() {
    if (renaming) return;
    list.replaceChildren();
    for (const session of state.sessions) list.appendChild(row(session));
    if (state.sessions.length === 0) {
      bindText(status, 'sessions_empty');
      status.hidden = false;
    } else {
      status.hidden = true;
    }
  }

  function row(session) {
    const item = document.createElement('li');
    item.className = 'session';
    item.setAttribute('data-session-id', session.id);

    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'session-open';
    open.textContent = session.title;
    open.title = session.title;
    if (session.id === state.sessionId) open.setAttribute('aria-current', 'page');
    open.addEventListener('click', () => onOpen(session.id));
    open.addEventListener('keydown', (event) => {
      if (event.key === 'F2') {
        event.preventDefault();
        startRename(item, session);
      }
    });

    const rename = document.createElement('button');
    rename.type = 'button';
    rename.className = 'icon-btn session-rename';
    bindAttr(rename, 'aria-label', 'rename_chat', { title: session.title });
    rename.appendChild(icon('pencil', { size: 15 }));
    rename.addEventListener('click', () => startRename(item, session));

    item.append(open, rename);
    return item;
  }

  function focusRow(id) {
    for (const item of list.children) {
      if (item.getAttribute('data-session-id') === id) {
        item.querySelector('.session-open')?.focus();
        return;
      }
    }
  }

  function startRename(item, session) {
    if (renaming) return;
    renaming = true;
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'session-input';
    input.maxLength = TITLE_MAX;
    input.value = session.title;
    input.autocomplete = 'off';
    input.spellcheck = false;
    bindAttr(input, 'aria-label', 'rename_input');
    for (const child of item.children) child.hidden = true;
    item.appendChild(input);
    input.focus();
    input.select();

    let settled = false;
    const finish = async (save) => {
      if (settled) return;
      settled = true;
      const title = input.value.trim();
      const hadFocus = document.activeElement === input;
      if (save && title && title !== session.title) {
        input.disabled = true;
        try {
          const updated = await api.renameSession(session.id, title);
          const next = { ...session, ...updated };
          setState({ sessions: state.sessions.map((s) => (s.id === session.id ? { ...s, title: next.title } : s)) });
          onRenamed(next);
        } catch (err) {
          if (!isAbort(err)) notify(err.message);
        }
      }
      renaming = false;
      render();
      if (hadFocus) focusRow(session.id);
    };

    input.addEventListener('keydown', (event) => {
      if (isImeEvent(event)) return;
      if (event.key === 'Enter') {
        event.preventDefault();
        finish(true);
      } else if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation(); // Esc here cancels the rename; it must not also stop a response
        finish(false);
      }
    });
    input.addEventListener('blur', () => finish(true));
  }

  return {
    render,
    /* Mark the open chat without rebuilding the list. */
    setCurrent(id) {
      for (const item of list.children) {
        const button = item.querySelector('.session-open');
        if (!button) continue;
        if (item.getAttribute('data-session-id') === id) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
      }
    },
  };
}
