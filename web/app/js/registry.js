/* The component registry: the one place where stream events and side-panel tabs meet their views.
 *
 * Stream events. `dispatch(event, turn)` is the single entry point for every event of a chat turn,
 * live (from the SSE stream) or replayed from history (`turn.replay === true`). It looks up the
 * handler registered for `event.type` and calls `handler(event, turn)`. Unknown types are ignored.
 * `turn.js` registers the shell's minimal handlers as fallbacks (`{fallback: true}`: kept only if
 * nothing else claimed the type, whatever the import order). Task 4.2's components call
 * `registerEvent(type, handler)` to take over a type (ToolCard, Thinking, …) or to add one
 * (`source-url`, `data-mandate`, `data-trade-proposal`, `data-grounding-flags`).
 *
 * Panels. `registerPanel(name, {mount, show})` attaches a view to a right-panel tab
 * (`portfolio`, `orders`, `mandate`, `settings`): `mount(container)` runs once, replacing the
 * placeholder; `show(container)` runs every time the tab becomes visible (refresh there).
 */

const eventHandlers = new Map();
const panels = new Map();

export function registerEvent(type, handler, { fallback = false } = {}) {
  if (typeof type !== 'string' || typeof handler !== 'function') return;
  if (fallback && eventHandlers.has(type)) return;
  eventHandlers.set(type, handler);
}

export function eventHandler(type) {
  return eventHandlers.get(type) || null;
}

export function dispatch(event, turn) {
  if (!event || typeof event !== 'object' || typeof event.type !== 'string') return false;
  const handler = eventHandlers.get(event.type);
  if (!handler) return false;
  try {
    handler(event, turn);
  } catch (err) {
    // One broken component must not take the stream down with it.
    console.error(`handler for "${event.type}" failed`, err);
  }
  return true;
}

export function registerPanel(name, panel) {
  if (typeof name !== 'string' || !panel || typeof panel !== 'object') return;
  panels.set(name, { mounted: false, ...panel });
}

export function panel(name) {
  return panels.get(name) || null;
}
