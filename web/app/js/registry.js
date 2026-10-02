/* The component registry: the one place where stream events and side-panel tabs meet their views.
 *
 * Stream events. `dispatch(event, turn)` is the single entry point for every event of a chat turn,
 * live (from the SSE stream) or replayed from history (`turn.replay === true`). Every listener
 * registered for `event.type` is called with `(event, turn)`, in registration order; one that
 * throws is logged and the others still run. Unknown types are ignored. The first `finish` of a
 * turn wins: it is stored as `turn.finish`, and every later event for that turn is ignored.
 *
 *   registerEvent(type, listener)   adds a listener. Nothing is displaced, so a component may
 *                                   listen for `finish`, `error` or `text-delta` next to the shell.
 *   replaceEvent(type, listener)    takes over the shell's fallback view for a type: the one-line
 *                                   tool rows (`tool-input-available`, `tool-output-available`) and
 *                                   the Thoughts block (`reasoning-delta`). Other listeners stay.
 *
 * The shell registers its fallback views with `{fallback: true}`. A fallback is dropped by
 * `replaceEvent`, and is not registered at all if `replaceEvent` already ran for that type, so the
 * import order of components and shell does not matter.
 *
 * Panels. `registerPanel(name, {mount, show})` attaches a view to a right-panel tab
 * (`portfolio`, `orders`, `mandate`, `settings`): `mount(container)` runs once, replacing the
 * placeholder; `show(container)` runs every time the tab becomes visible (refresh there).
 */

const listeners = new Map(); // type -> [{listener, fallback}]
const replaced = new Set();  // types whose fallback view a component took over
const panels = new Map();

export function registerEvent(type, listener, { fallback = false } = {}) {
  if (typeof type !== 'string' || typeof listener !== 'function') return;
  if (fallback && replaced.has(type)) return;
  if (!listeners.has(type)) listeners.set(type, []);
  listeners.get(type).push({ listener, fallback });
}

export function replaceEvent(type, listener) {
  if (typeof type !== 'string' || typeof listener !== 'function') return;
  replaced.add(type);
  const kept = (listeners.get(type) || []).filter((entry) => !entry.fallback);
  kept.push({ listener, fallback: false });
  listeners.set(type, kept);
}

/* Deliver one event to the listeners of its type. Returns whether anything handled it. */
export function dispatch(event, turn) {
  if (!event || typeof event !== 'object' || typeof event.type !== 'string') return false;
  if (turn && turn.finish) return false; // the turn is over: a duplicate finish or a late event
  const entries = listeners.get(event.type) || [];
  for (const { listener } of [...entries]) {
    try {
      listener(event, turn);
    } catch (err) {
      // One broken component must not take the stream down with it.
      console.error(`listener for "${event.type}" failed`, err);
    }
  }
  if (event.type === 'finish' && turn && !turn.finish) turn.finish = event;
  return entries.length > 0 || event.type === 'finish';
}

export function registerPanel(name, panel) {
  if (typeof name !== 'string' || !panel || typeof panel !== 'object') return;
  panels.set(name, { mounted: false, ...panel });
}

export function panel(name) {
  return panels.get(name) || null;
}
