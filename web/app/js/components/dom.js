/* Small DOM helpers shared by the conversation components.
 *
 * Everything a component shows that came from the model, a tool, the broker or the web is untrusted
 * text: it only ever reaches the DOM through `textContent` (the `text` argument of `el`), never as
 * markup. Links are built by `externalLink`, which keeps only absolute http(s) URLs.
 */
import { bindAttr, bindText, getLang } from '../i18n.js';
import { icon } from '../icons.js';
import { safeHttpUrl } from '../markdown.js';

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

export const str = (value) => (typeof value === 'string' ? value : '');

/* An element whose text follows the interface language. */
export function label(tag, className, key, vars) {
  return bindText(el(tag, className), key, vars);
}

export function iconSpan(name, size = 16, className = 'icon') {
  const holder = el('span', className);
  holder.appendChild(icon(name, { size }));
  return holder;
}

/* <button type="button"> with a translated label and an optional leading icon. */
export function actionButton(className, key, { vars, iconName, iconSize = 16 } = {}) {
  const button = el('button', className);
  button.type = 'button';
  if (iconName) button.appendChild(iconSpan(iconName, iconSize));
  button.appendChild(label('span', 'btn-label', key, vars));
  return button;
}

/* An icon-only button; the accessible name is the translated `key`. */
export function iconButton(iconName, key, { size = 16, className = 'icon-btn' } = {}) {
  const button = el('button', className);
  button.type = 'button';
  bindAttr(button, 'aria-label', key);
  button.appendChild(icon(iconName, { size }));
  return button;
}

/* A status word with a tone. The word carries the meaning; the tone is only a colour hint. */
export function badge(tone, key, vars) {
  const node = label('span', 'badge', key, vars);
  node.setAttribute('data-tone', tone);
  return node;
}

/* A link to another site, or null when `url` is not an absolute http(s) URL without credentials. */
export function externalLink(url, className) {
  const safe = safeHttpUrl(url);
  if (!safe) return null;
  const link = el('a', className);
  link.setAttribute('href', safe);
  link.setAttribute('rel', 'noopener noreferrer nofollow');
  link.setAttribute('target', '_blank');
  link.setAttribute('referrerpolicy', 'no-referrer');
  return link;
}

export function hostOf(url) {
  const safe = safeHttpUrl(url);
  if (!safe) return '';
  try {
    return new URL(safe).host.replace(/^www\./, '');
  } catch {
    return '';
  }
}

// Turn extras, top to bottom. A component asks for its section by name; the order is fixed here.
const EXTRAS = ['sources', 'flags', 'suggestions', 'mandate', 'ticket'];

/* The named container inside `turn.slots.extras`, created in its fixed place on first use. */
export function extraSection(turn, name) {
  const host = turn.slots.extras;
  for (const child of host.children) {
    if (child.getAttribute('data-extra') === name) return child;
  }
  const box = el('div', 'extra');
  box.setAttribute('data-extra', name);
  const rank = EXTRAS.indexOf(name);
  const after = [...host.children].find((child) => EXTRAS.indexOf(child.getAttribute('data-extra')) > rank);
  host.insertBefore(box, after || null);
  return box;
}

/* How long ago, as an i18n key and its variables: seconds below a minute, then minutes, hours. */
export function ageParts(ms) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return ['age_seconds', { n: seconds }];
  if (seconds < 3600) return ['age_minutes', { n: Math.floor(seconds / 60) }];
  return ['age_hours', { n: Math.floor(seconds / 3600) }];
}

/* 272 -> "4:32" (a countdown). */
export function clockText(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds)); // not money: seconds
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

/* An ISO timestamp as a short local date and time; the raw text when it cannot be parsed. */
export function dateTimeText(iso) {
  if (typeof iso !== 'string' || !iso) return '';
  const time = Date.parse(iso);
  if (Number.isNaN(time)) return iso.slice(0, 40);
  try {
    return new Intl.DateTimeFormat(getLang() === 'ko' ? 'ko-KR' : 'en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    }).format(new Date(time));
  } catch {
    return iso.slice(0, 16).replace('T', ' ');
  }
}

/* Milliseconds since the epoch for an ISO timestamp, or null. */
export function timeOf(iso) {
  const time = typeof iso === 'string' ? Date.parse(iso) : NaN;
  return Number.isNaN(time) ? null : time;
}

let serial = 0;

/* A document-unique id for label/for and aria-describedby pairs. */
export function uid(prefix) {
  serial += 1;
  return `${prefix}-${serial}`;
}
