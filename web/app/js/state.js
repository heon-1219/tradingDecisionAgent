/* App state, a tiny event bus and safe localStorage access.
 *
 * Nothing here touches the DOM, so the module also loads under Node (the i18n and SSE checks).
 * Every storage access is wrapped in try/catch: storage can be blocked (private mode, policy)
 * or full, and the app must keep working without it.
 */

export const STORAGE_KEYS = Object.freeze({
  theme: 'finagent-app.theme',
  lang: 'finagent-app.lang',
  rail: 'finagent-app.rail',
  session: 'finagent-app.session',
});

export const storage = {
  get(key) {
    try {
      return globalThis.localStorage ? globalThis.localStorage.getItem(key) : null;
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      if (globalThis.localStorage) globalThis.localStorage.setItem(key, String(value));
    } catch {
      /* storage blocked or full: the setting simply does not persist */
    }
  },
  remove(key) {
    try {
      if (globalThis.localStorage) globalThis.localStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  },
};

/* The shared state. Modules read it directly and change it through `setState`, which emits
 * `change` with the patch, so views can react without knowing who changed what. */
export const state = {
  sessionId: null,     // the open chat session, or null for a new (not yet created) chat
  sessionTitle: '',
  sessions: [],        // GET /api/sessions -> sessions
  streaming: false,    // a /api/chat stream is open
  settings: null,      // GET /api/settings body, or null
  settingsError: null, // the message to show when settings could not be loaded
  lastFinish: null,    // the last `finish` event {usage, elapsed_ms}
};

const listeners = new Map();

/* Subscribe to a bus event; returns an unsubscribe function.
 * Events the shell emits: `change` (state patch), `session:loaded` (GET /api/sessions/{id} body),
 * `session:new`, `turn:start` ({turn, sessionId}), `turn:end` ({turn, outcome, finish, sessionId}),
 * `settings` (settings body), `lang` (new language).
 * Events the shell listens to (for components): `chat:send` (text: send it as a user message),
 * `panel:open` (tab name), `settings:refresh`, `sessions:refresh`. */
export function on(name, fn) {
  if (!listeners.has(name)) listeners.set(name, new Set());
  listeners.get(name).add(fn);
  return () => listeners.get(name)?.delete(fn);
}

export function emit(name, payload) {
  const set = listeners.get(name);
  if (!set) return;
  for (const fn of [...set]) {
    try {
      fn(payload);
    } catch (err) {
      console.error(`listener for "${name}" failed`, err);
    }
  }
}

export function setState(patch) {
  Object.assign(state, patch);
  emit('change', patch);
}
