/* Grounding: Google's search suggestions and the "numbers not found in the sources" note.
 *
 * `source-grounding {search_suggestions_html}`. Google requires its search-suggestions HTML to be
 * shown unmodified, and it is third-party markup, so it never touches this page's DOM: it is given,
 * as it is, to a sandboxed iframe through `srcdoc`. The sandbox allows popups only: no scripts, no
 * same-origin access, no forms, no top navigation. The links in Google's markup carry no `target`,
 * so inside the sandbox they could only navigate the frame itself, which the page's CSP
 * (`frame-src 'self'`) refuses. One constant is therefore put in front of the HTML, `NEW_TAB`
 * (`<base target="_blank">`), so that a suggestion opens Google Search in a new tab, as Google's
 * terms expect; the HTML that follows it is untouched. This file is the only place in the app that
 * sets `srcdoc`, and tests/test_web_app.py pins the sandbox value and that one statement.
 * History has no copy of the HTML (the server does not store it), so a reloaded chat shows nothing.
 *
 * `data-grounding-flags {numbers}`. An advisory, collapsed note under the answer listing the numbers
 * the server could not find in the turn's sources. It blocks nothing. In history the numbers come
 * from the stored assistant message (`meta.grounding_flags`).
 */
import { bindAttr } from '../i18n.js';
import { registerEvent } from '../registry.js';
import { el, extraSection, iconSpan, label, str } from './dom.js';

const SANDBOX = 'allow-popups allow-popups-to-escape-sandbox';
const NEW_TAB = '<base target="_blank">';
const MAX_HTML_CHARS = 200_000;
const MAX_FLAGS = 40;

registerEvent('source-grounding', (event, turn) => {
  const html = str(event.search_suggestions_html);
  if (!html || html.length > MAX_HTML_CHARS) return;
  const box = extraSection(turn, 'suggestions');
  const frame = el('iframe', 'suggestions-frame');
  // The sandbox is set before the content and before the frame joins the document.
  frame.setAttribute('sandbox', SANDBOX);
  frame.setAttribute('referrerpolicy', 'no-referrer');
  frame.setAttribute('loading', 'lazy');
  bindAttr(frame, 'title', 'search_suggestions_label');
  frame.srcdoc = NEW_TAB + html;
  box.replaceChildren(label('h3', 'extra-h', 'search_suggestions_label'), frame);
});

function flagsNote(turn, numbers) {
  const shown = numbers.filter((n) => typeof n === 'string' && n.trim()).slice(0, MAX_FLAGS);
  if (shown.length === 0) return;
  const box = extraSection(turn, 'flags');
  const details = el('details', 'flagnote');
  const summary = el('summary');
  const texts = shown.map((number) => number.trim().slice(0, 60));
  summary.append(iconSpan('info', 16),
    label('span', 'flagnote-title', shown.length === 1 ? 'flags_one' : 'flags_many', { n: shown.length }),
    el('span', 'flagnote-peek', texts.join(', ')));
  const list = el('ul', 'flagnote-list');
  for (const text of texts) list.appendChild(el('li', 'flagnote-number', text));
  details.append(summary, list, label('p', 'flagnote-help', 'flags_help'));
  box.replaceChildren(details);
}

registerEvent('data-grounding-flags', (event, turn) => {
  const stored = turn.replay && turn.data.meta ? turn.data.meta.grounding_flags : null;
  const numbers = Array.isArray(event.numbers) ? event.numbers : stored;
  if (Array.isArray(numbers)) flagsNote(turn, numbers);
});
