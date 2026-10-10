/* The DEMO badge in the top bar, next to PAPER: shown when `GET /api/settings` says `demo: true`
 * (the keyless demo server, scripts/demo_server.py), hidden otherwise, also when the field is absent.
 * Its title and accessible name say what that means: a fake broker and a scripted model, nothing
 * leaves the computer. On a phone the PAPER badge shrinks to its dot beside it (components.css). */
import { bindAttr } from '../i18n.js';
import { on, state } from '../state.js';
import { iconSpan, label } from './dom.js';

function render() {
  const badge = typeof document !== 'undefined' ? document.getElementById('demo-badge') : null;
  if (!badge) return;
  const demo = Boolean(state.settings && state.settings.demo === true);
  badge.hidden = !demo;
  if (!demo) return;
  if (!badge.querySelector('.badge-demo-label')) {
    badge.replaceChildren(iconSpan('info', 12, 'icon badge-demo-icon'), label('span', 'badge-demo-label', 'demo_badge'));
  }
  bindAttr(badge, 'title', 'demo_title');
  bindAttr(badge, 'aria-label', 'demo_title');
}

on('change', (patch) => {
  if ('settings' in patch) render();
});
