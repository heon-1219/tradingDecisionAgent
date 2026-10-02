/* Entry point: wires the shell together and runs the chat flow.
 *
 * One turn: the user's message is shown, a Turn is created, POST /api/chat streams events, and
 * every event goes through `dispatch(event, turn)` (registry.js), the single place where events
 * meet their views. When the stream ends the turn is closed, the top bar shows the tokens and
 * latency of `finish`, and one polite announcement is made. Stop aborts the fetch; that is a clean
 * end, not an error.
 *
 * Task 4.2: add `import './components/index.js';` to the imports below. Components register
 * themselves with `registerEvent` / `registerPanel` (registry.js) and `addPostProcessor`
 * (markdown.js); the shell's own handlers and Settings view are fallbacks and step aside.
 */
import { api, isAbort } from './api.js';
import { initCopyButtons } from './clipboard.js';
import { initComposer } from './composer.js';
import { applyI18n, bindText, getLang, t } from './i18n.js';
import { hydrateIcons } from './icons.js';
import { initLayout } from './layout.js';
import { announce, notify } from './notify.js';
import { initPanels } from './panels.js';
import { dispatch } from './registry.js';
import { initSessions } from './sessions.js';
import { streamChat } from './sse.js';
import { emit, on, setState, state, storage, STORAGE_KEYS } from './state.js';
import { createThread } from './thread.js';
import { initTopbar } from './topbar.js';
import { registerDefaultHandlers, Turn } from './turn.js';

const MAX_MESSAGE_CHARS = 16000; // the server's limit for POST /api/chat
const TITLE_CHARS = 60;

const $ = (id) => document.getElementById(id);
let active = null; // {controller, turn} while a response streams
let openSeq = 0;   // guards against a slow session load landing after a newer one

/* A chat title from its first message: the first line, cut at a word near 60 characters. */
function titleFrom(message) {
  const line = message.trim().split(/\r?\n/)[0].replace(/\s+/g, ' ').trim();
  if (!line) return t('new_chat');
  if (line.length <= TITLE_CHARS) return line;
  const cut = line.slice(0, TITLE_CHARS);
  const space = cut.lastIndexOf(' ');
  return `${(space > TITLE_CHARS / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

registerDefaultHandlers();
hydrateIcons(document);
applyI18n(document);

const thread = createThread({
  scroller: $('scroller'),
  thread: $('thread'),
  onSuggestion: (text) => composer.fill(text),
});
const composer = initComposer({
  form: $('composer'),
  input: $('composer-input'),
  sendButton: $('send-button'),
  stopButton: $('stop-button'),
  onSend: (text) => send(text),
  onStop: () => stop(),
});
const sessions = initSessions({
  list: $('session-list'),
  status: $('session-status'),
  onOpen: (id) => openSession(id),
  onRenamed: (session) => {
    if (session.id === state.sessionId) setState({ sessionTitle: session.title });
  },
});
initPanels({ refreshSettings });
const layout = initLayout({ onEscape: () => stop() });
initTopbar();
initCopyButtons($('thread'));
$('new-chat').addEventListener('click', () => newChat());

// Hooks for components (Task 4.2).
on('chat:send', (text) => send(String(text ?? '')));
on('panel:open', (name) => layout.openPanel(name));
on('settings:refresh', () => refreshSettings());
on('sessions:refresh', () => refreshSessions({ quiet: true }));

function setStreaming(value) {
  setState({ streaming: value });
  composer.setStreaming(value);
}

function stop() {
  if (active) active.controller.abort();
}

async function refreshSettings() {
  try {
    const settings = await api.getSettings();
    setState({ settings, settingsError: null });
    emit('settings', settings);
  } catch (err) {
    if (isAbort(err)) return;
    setState({ settings: null, settingsError: err.message || t('settings_error') });
  }
}

async function refreshSessions({ quiet = false } = {}) {
  try {
    const body = await api.listSessions();
    setState({ sessions: Array.isArray(body?.sessions) ? body.sessions : [] });
    sessions.render();
    return true;
  } catch (err) {
    if (isAbort(err)) return false;
    if (!quiet) {
      const status = $('session-status');
      bindText(status, 'err_sessions');
      status.hidden = false;
      notify(err.message || t('err_sessions'));
    }
    return false;
  }
}

function newChat() {
  stop();
  openSeq += 1;
  storage.remove(STORAGE_KEYS.session);
  setState({ sessionId: null, sessionTitle: '', lastFinish: null });
  sessions.setCurrent(null);
  thread.showEmpty();
  layout.closeDrawer();
  if (!layout.isPhone()) composer.focus();
  emit('session:new', null);
}

async function openSession(id) {
  if (!id) return;
  stop();
  const seq = ++openSeq;
  layout.closeDrawer();
  thread.showLoading();
  try {
    const detail = await api.getSession(id);
    if (seq !== openSeq) return;
    const session = detail?.session || { id, title: '' };
    storage.set(STORAGE_KEYS.session, id);
    setState({ sessionId: id, sessionTitle: session.title || '', lastFinish: null });
    sessions.setCurrent(id);
    thread.renderHistory(detail?.messages);
    emit('session:loaded', detail);
  } catch (err) {
    if (seq !== openSeq || isAbort(err)) return;
    notify(err.message || t('err_generic'));
    newChat();
    if (err.status === 404) refreshSessions({ quiet: true });
  }
}

async function send(raw) {
  const message = String(raw ?? '').trim();
  if (!message || state.streaming) return;
  if (message.length > MAX_MESSAGE_CHARS) {
    notify(t('err_too_long'));
    return;
  }

  composer.clear();
  thread.appendUser(message);
  const turn = new Turn();
  thread.appendTurn(turn);
  turn.setPending(true);
  thread.scrollToBottom();

  const controller = new AbortController();
  const run = { controller, turn };
  active = run;
  setStreaming(true);

  let outcome = 'finish';
  let sessionId = state.sessionId;
  try {
    if (!sessionId) {
      const session = await api.createSession({ title: titleFrom(message), lang: getLang() }, { signal: controller.signal });
      sessionId = session.id;
      storage.set(STORAGE_KEYS.session, sessionId);
      setState({
        sessionId,
        sessionTitle: session.title || '',
        sessions: [{ ...session, updated_at: session.created_at }, ...state.sessions],
      });
      sessions.render();
    }
    emit('turn:start', { turn, sessionId });
    const { finished } = await streamChat(sessionId, message, {
      signal: controller.signal,
      onEvent(event) {
        dispatch(event, turn);
        thread.follow();
      },
    });
    if (!finished) {
      outcome = 'cut';
      turn.showError(t('err_stream_cut'));
    } else if (turn.hadError) {
      outcome = 'error';
    }
  } catch (err) {
    if (controller.signal.aborted || isAbort(err)) {
      outcome = 'stopped';
    } else {
      outcome = 'error';
      turn.showError(err?.message || t('err_generic'));
      if (err?.status === 404 && state.sessionId === sessionId) {
        // The chat no longer exists on the server: the next message starts a new one.
        storage.remove(STORAGE_KEYS.session);
        setState({ sessionId: null, sessionTitle: '' });
        refreshSessions({ quiet: true });
      }
    }
  } finally {
    turn.close(outcome);
    thread.follow();
    if (active === run) {
      active = null;
      setStreaming(false);
    }
    if (turn.finish && state.sessionId === sessionId) setState({ lastFinish: turn.finish });
    const key = { finish: 'announce_done', stopped: 'announce_stopped' }[outcome] || 'announce_error';
    announce(t(key));
    emit('turn:end', { turn, outcome, finish: turn.finish, sessionId });
  }
}

async function start() {
  refreshSettings();
  const last = storage.get(STORAGE_KEYS.session);
  thread.showEmpty();
  const loaded = await refreshSessions();
  if (loaded && last && state.sessions.some((s) => s.id === last) && !state.streaming && !state.sessionId) {
    await openSession(last);
  } else if (!layout.isPhone() && document.activeElement === document.body) {
    composer.focus();
  }
}

start();
