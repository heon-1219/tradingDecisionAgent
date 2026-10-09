/* Form pieces for the Settings panel's sections: a labelled field with its error line, and messages
 * that follow a language switch.
 *
 * A message is `{key}` (an i18n key: bound with `bindText`, so `applyI18n` re-translates it) or
 * `{raw}` (a server's own fixed text, for a code the app has no words for: shown as it is). Both are
 * set as text, never as markup.
 */
import { bindText, t } from '../i18n.js';
import { el, label, uid } from './dom.js';

/* A field: caption, the control (or a `holder` around it), an optional help line and an error line
 * (hidden until used). */
export function field(captionKey, control, { help, holder } = {}) {
  const wrap = el('div', 'field');
  const caption = label('label', 'field-label', captionKey);
  control.id = uid('field');
  caption.setAttribute('for', control.id);
  const error = el('p', 'field-error');
  error.id = uid('field-error');
  error.hidden = true;
  wrap.append(caption, holder || control, ...(help ? [help] : []), error);
  return { wrap, caption, control, error };
}

/* Put a message into `node`; null empties it. */
export function setMessage(node, message) {
  if (message && message.key) {
    bindText(node, message.key);
    return;
  }
  node.removeAttribute('data-i18n');
  node.removeAttribute('data-i18n-vars');
  node.textContent = (message && message.raw) || '';
}

/* The message as text in the current language (for an announcement). */
export const textOf = (message) => (message.key ? t(message.key) : message.raw);

/* Show `message` under a field from `field()` and mark its control invalid; null clears both. */
export function setFieldError(spot, message) {
  spot.error.hidden = !message;
  setMessage(spot.error, message);
  if (message) {
    spot.control.setAttribute('aria-invalid', 'true');
    spot.control.setAttribute('aria-describedby', spot.error.id);
  } else {
    spot.control.removeAttribute('aria-invalid');
    spot.control.removeAttribute('aria-describedby');
  }
}
