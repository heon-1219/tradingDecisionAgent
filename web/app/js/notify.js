/* Messages outside the conversation: screen-reader announcements and error toasts.
 *
 * The app has exactly one live region, `#announcer` (polite, visually hidden). It gets one short
 * announcement at the end of each turn ("Response complete"), and the text of a toast when one
 * appears. Streamed text is never inside a live region. There is no alert(): a failed request
 * shows the server's `message` as a toast above the composer.
 */
import { bindAttr } from './i18n.js';
import { icon } from './icons.js';

const TOAST_MS = 9000;
const MAX_TOASTS = 3;
let announceTimer = null;

export function announce(message) {
  const region = document.getElementById('announcer');
  if (!region || !message) return;
  // Clear first so the same sentence twice in a row is announced twice.
  region.textContent = '';
  clearTimeout(announceTimer);
  announceTimer = setTimeout(() => {
    region.textContent = message;
  }, 60);
}

export function notify(message, { kind = 'error' } = {}) {
  const host = document.getElementById('toasts');
  if (!host || !message) return;
  while (host.children.length >= MAX_TOASTS) host.firstChild.remove();

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.setAttribute('data-kind', kind);
  const mark = document.createElement('span');
  mark.className = 'icon';
  mark.appendChild(icon('alert', { size: 18 }));
  const body = document.createElement('span');
  body.className = 'toast-text';
  body.textContent = message;
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'icon-btn';
  bindAttr(close, 'aria-label', 'dismiss');
  close.appendChild(icon('close', { size: 16 }));
  toast.append(mark, body, close);

  let timer = null;
  const dismiss = () => {
    clearTimeout(timer);
    toast.remove();
  };
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(dismiss, TOAST_MS);
  };
  close.addEventListener('click', dismiss);
  // Reading or focusing a toast keeps it on screen.
  toast.addEventListener('pointerenter', () => clearTimeout(timer));
  toast.addEventListener('pointerleave', arm);
  toast.addEventListener('focusin', () => clearTimeout(timer));
  toast.addEventListener('focusout', arm);

  host.appendChild(toast);
  arm();
  announce(message);
}
