/* The kill switch in the Settings panel: stops trading at once (`POST /api/kill-switch`).
 *
 * The first button only opens an inline confirmation (not a browser dialog): the question, a
 * checkbox for the stronger form ("also suspend at the broker and cancel open orders") and a second
 * button that stops trading; Cancel gets the focus. If the server could not save the change to its
 * .env file (`persisted: false`), trading is off now but would be on again after a restart, so a
 * warning stays on screen for as long as the settings report trading as off.
 */
import { request } from '../api.js';
import { bindText, t } from '../i18n.js';
import { announce } from '../notify.js';
import { emit, state } from '../state.js';
import { actionButton, el, iconSpan, label } from './dom.js';

export function createKillSwitch({ adopt }) {
  const root = el('div', 'kill');
  const open = actionButton('btn btn-danger', 'kill_open', { iconName: 'power' });
  open.setAttribute('aria-expanded', 'false');
  const hint = label('p', 'panel-note', 'kill_hint');

  const confirmBox = el('div', 'kill-confirm');
  confirmBox.hidden = true;
  const suspendWrap = el('label', 'check');
  const suspend = el('input');
  suspend.type = 'checkbox';
  suspendWrap.append(suspend, label('span', '', 'kill_suspend'));
  const actions = el('div', 'conn-actions');
  const go = actionButton('btn btn-danger-solid btn-small', 'kill_confirm');
  const cancel = actionButton('btn btn-quiet btn-small', 'cancel');
  actions.append(go, cancel);
  confirmBox.append(label('p', 'kill-question', 'kill_question'), suspendWrap, actions);

  const result = el('p', 'conn-message');
  result.hidden = true;
  const warning = el('div', 'notice');
  warning.setAttribute('data-tone', 'bad');
  warning.hidden = true;
  const warningBody = el('div', 'notice-body');
  warningBody.appendChild(label('p', 'notice-text', 'kill_not_persisted'));
  warning.append(iconSpan('alert', 18), warningBody);

  root.append(open, hint, confirmBox, result, warning);

  function close() {
    confirmBox.hidden = true;
    open.setAttribute('aria-expanded', 'false');
    suspend.checked = false;
  }

  open.addEventListener('click', () => {
    confirmBox.hidden = !confirmBox.hidden;
    open.setAttribute('aria-expanded', String(!confirmBox.hidden));
    if (!confirmBox.hidden) cancel.focus(); // the safe choice gets the focus, not the destructive one
  });
  cancel.addEventListener('click', () => {
    close();
    open.focus();
  });
  confirmBox.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    event.stopPropagation();
    close();
    open.focus();
  });
  go.addEventListener('click', async () => {
    go.disabled = true;
    cancel.disabled = true;
    result.hidden = true;
    try {
      const body = await request('POST', '/api/kill-switch', { suspend_broker: suspend.checked });
      warning.hidden = body.persisted !== false;
      result.hidden = false;
      result.setAttribute('data-tone', 'ok');
      bindText(result, body.suspend_broker && body.broker_connected ? 'kill_done_suspended' : 'kill_done');
      announce(t('kill_done'));
      if (state.settings) adopt({ ...state.settings, trading_enabled: false });
      emit('settings:refresh');
      emit('portfolio:refresh');
      close();
      open.focus();
    } catch (err) {
      result.hidden = false;
      result.setAttribute('data-tone', 'bad');
      result.removeAttribute('data-i18n');
      result.textContent = err?.message || t('err_generic');
    }
    go.disabled = false;
    cancel.disabled = false;
  });

  return {
    el: root,
    /* Called with every settings update: the warning ends once trading is on again. */
    update(settings) {
      if (settings && settings.trading_enabled) {
        warning.hidden = true;
        result.hidden = true;
      }
    },
  };
}
