/* The inline "Modify" fields of a ticket: quantity (or dollar amount) and limit price per
 * executable leg. Nothing is changed here: Send writes a "Modify: …" message and hands it to the
 * chat (`chat:send`), so the agent re-proposes and the risk engine re-checks; the new ticket
 * replaces this one.
 */
import { toTrimmed } from '../decimal.js';
import { bindAttr, bindText, getLang } from '../i18n.js';
import { actionButton, el, label, str, uid } from './dom.js';
import { sideBadge } from './order-view.js';
import { modifyMessage, readModify } from './ticket-model.js';

function field(index, name, key, value, unit) {
  const wrap = el('div', 'field');
  const id = uid('modify');
  const caption = label('label', 'field-label', key);
  caption.setAttribute('for', id);
  const input = el('input', 'field-input');
  input.id = id;
  input.type = 'text';
  input.inputMode = 'decimal';
  input.autocomplete = 'off';
  input.maxLength = 20;
  input.value = value;
  input.setAttribute('data-leg', String(index));
  input.setAttribute('data-name', name);
  const box = el('div', 'field-box');
  if (unit) {
    const mark = el('span', 'field-unit', unit);
    mark.setAttribute('aria-hidden', 'true');
    box.classList.add('has-prefix');
    box.append(mark, input);
  } else {
    box.appendChild(input);
  }
  wrap.append(caption, box);
  return wrap;
}

/* `legs` are the ticket's executable legs. `onSend(message)` returns false when the message could
 * not be sent (a response is still streaming). */
export function createModifyForm({ legs, onSend, onCancel }) {
  const root = el('div', 'ticket-modify');
  root.appendChild(label('h4', 'ticket-h', 'ticket_modify_heading'));
  const rows = el('div', 'modify-rows');
  legs.forEach((leg, index) => {
    const row = el('div', 'modify-row');
    const who = el('div', 'modify-leg');
    who.append(sideBadge(leg.side), el('span', 'leg-symbol', str(leg.symbol)));
    row.appendChild(who);
    if (leg.qty !== null && leg.qty !== undefined) {
      row.appendChild(field(index, 'qty', 'ticket_modify_qty', toTrimmed(str(leg.qty)) || ''));
    } else {
      row.appendChild(field(index, 'notional', 'ticket_modify_notional', str(leg.notional), '$'));
    }
    if (str(leg.type) === 'limit') {
      row.appendChild(field(index, 'limit_price', 'ticket_modify_limit', str(leg.limit_price), '$'));
    }
    rows.appendChild(row);
  });
  const error = el('p', 'field-error');
  error.hidden = true;
  const actions = el('div', 'modify-actions');
  const send = actionButton('btn btn-primary btn-small', 'ticket_modify_send');
  const cancel = actionButton('btn btn-quiet btn-small', 'cancel');
  actions.append(send, cancel);
  root.append(rows, label('p', 'card-note', 'ticket_modify_note'), error, actions);

  function values() {
    const out = legs.map(() => ({}));
    for (const input of rows.querySelectorAll('input')) {
      out[input.getAttribute('data-leg')][input.getAttribute('data-name')] = input.value;
    }
    return out;
  }

  function submit() {
    const { edits, errors } = readModify(legs, values());
    let first = null;
    for (const input of rows.querySelectorAll('input')) {
      const bad = `${input.getAttribute('data-leg')}.${input.getAttribute('data-name')}` in errors;
      input.setAttribute('aria-invalid', String(bad));
      if (bad && !first) first = input;
    }
    error.hidden = !first;
    if (first) {
      bindText(error, 'ticket_modify_err');
      first.focus();
      return;
    }
    if (onSend(modifyMessage(edits, getLang())) === false) {
      error.hidden = false;
      bindText(error, 'ticket_modify_busy');
    }
  }

  send.addEventListener('click', submit);
  cancel.addEventListener('click', () => onCancel());
  root.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.isComposing && event.target.tagName === 'INPUT') {
      event.preventDefault();
      submit();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      onCancel();
    }
  });
  bindAttr(root, 'aria-label', 'ticket_modify_heading');
  root.setAttribute('role', 'group');
  return { el: root, focus: () => rows.querySelector('input')?.focus() };
}
