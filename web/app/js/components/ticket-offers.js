/* What a ticket card offers in its notices: a link to a panel, "Ask for a new ticket", "Show the newer
 * ticket", and the notice for a refused refresh (409), chosen by the server's `reason`
 * (ticket-model.js `refusalKind`): only `not_newest` claims that a newer ticket exists. */
import { getLang, t } from '../i18n.js';
import { notify } from '../notify.js';
import { canSend, emit, state } from '../state.js';
import { actionButton } from './dom.js';
import { confirmLabel, executableLegs, modifyMessage, refusalKind } from './ticket-model.js';
import { newerCard } from './ticket-store.js';

export function panelLink(key, panel, iconName) {
  const button = actionButton('btn btn-quiet btn-small', key, { iconName, iconSize: 14 });
  button.addEventListener('click', () => emit('panel:open', panel));
  return button;
}

/* "Ask for a new ticket": the same legs, sent to the assistant as a Modify message. Disabled, like
 * Confirm, when the card cannot read an order of the ticket. */
function askAgainButton(ticket) {
  const ask = actionButton('btn btn-small', 'ticket_ask_again', { iconName: 'refresh', iconSize: 14 });
  const message = confirmLabel(ticket).safe ? modifyMessage(executableLegs(ticket), getLang()) : null;
  ask.disabled = !message;
  ask.addEventListener('click', () => {
    if (!message) return;
    if (canSend(state)) emit('chat:send', message);
    else notify(t('ticket_modify_busy'));
  });
  return ask;
}

/* "Show the newer ticket": scroll to it and move the focus there. */
function showButton(card) {
  const show = actionButton('btn btn-quiet btn-small', 'ticket_show_newer', { iconName: 'ticket', iconSize: 14 });
  show.addEventListener('click', () => {
    card.el.scrollIntoView({ block: 'nearest' });
    card.el.focus({ preventScroll: true });
  });
  return show;
}


/* The notice for a refused refresh (409), from the server's reason; `detail` is the server's text. */
export function refusedNotice(ticket, sessionId, outcome) {
  const detail = outcome.message;
  switch (refusalKind(outcome.reason, outcome.blockCodes)) {
    case 'newer': {
      const newer = newerCard(sessionId, ticket.id);
      return { tone: 'warn', key: 'ticket_refresh_newer', actions: [...(newer ? [showButton(newer)] : []), askAgainButton(ticket)] };
    }
    case 'prices':
      return { tone: 'warn', key: 'ticket_refresh_prices', detail, actions: [askAgainButton(ticket)] };
    case 'mandate':
      return { tone: 'warn', key: 'ticket_refresh_mandate', detail, actions: [panelLink('open_mandate', 'mandate', 'shield')] };
    case 'not_paper':
      return { tone: 'bad', key: 'ticket_refresh_not_paper', detail, actions: [panelLink('open_settings', 'settings', 'sliders')] };
    default:
      return { tone: 'bad', key: 'ticket_refresh_refused', detail };
  }
}
