/* The Thinking block: the model's interim text (`reasoning-delta`), collapsed by default.
 *
 * While the turn works it says "Thinking…"; once the answer starts (or the turn ends) it says
 * "Thought for N s", counted from the start of the turn. The text is plain text (one text node),
 * never markdown. History has no block: the server does not store reasoning.
 */
import { bindText } from '../i18n.js';
import { icon } from '../icons.js';
import { registerEvent, replaceEvent } from '../registry.js';
import { on } from '../state.js';
import { el, str } from './dom.js';

function settle(turn) {
  const view = turn.data.thinking;
  if (!view || view.settled) return;
  view.settled = true;
  view.label.classList.remove('shimmer');
  const seconds = Math.max(1, Math.round((performance.now() - view.started) / 1000)); // not money: seconds
  bindText(view.label, 'thinking_for', { s: seconds });
}

registerEvent('start', (event, turn) => {
  turn.data.startedAt = performance.now();
});

replaceEvent('reasoning-delta', (event, turn) => {
  const delta = str(event.delta);
  if (turn.closed || !delta) return;
  let view = turn.data.thinking;
  if (!view) {
    const details = el('details', 'thinking');
    const summary = el('summary');
    const label = el('span', 'thinking-label shimmer');
    bindText(label, 'thinking_live');
    summary.append(label, icon('chevron', { size: 16, className: 'chev' }));
    const body = el('div', 'thinking-body');
    const node = document.createTextNode('');
    body.appendChild(node);
    details.append(summary, body);
    turn.slots.thinking.appendChild(details);
    view = { label, node, settled: false, started: turn.data.startedAt ?? performance.now() };
    turn.data.thinking = view;
  }
  // The agent sends one interim text per tool round: separate rounds with a blank line.
  if (turn.newRound && view.node.length > 0) view.node.appendData('\n\n');
  turn.newRound = false;
  view.node.appendData(delta);
});

registerEvent('text-delta', (event, turn) => settle(turn));
on('turn:end', ({ turn }) => settle(turn));
