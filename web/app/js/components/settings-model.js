/* The Settings panel's model section: provider, API key, model, the Gemini paid switch, effort,
 * and what is in use. The rules for what a save sends live in llm-model.js.
 *
 * Keys follow the panel's rules (settings.js): the key field is a password input that is never
 * pre-filled and is emptied when a save is answered, whatever the answer; a typed key goes from the
 * input straight into a request body (`PUT /api/settings`, or `POST /api/llm/models` for "Load
 * models"), never into storage, onto the bus or into a log line or a message.
 *
 * The model list comes from the app's own `POST /api/llm/models` (the server asks the provider):
 * it is loaded when another provider whose key is saved is chosen, and by "Load models" (with the
 * typed key, if any), and once each time the panel opens when the provider's key is saved. A list
 * is shown only if it is the newest load and its provider is still the one chosen, whether it
 * succeeded or failed. A model id the list does not have can be typed in "Custom model id"; it wins
 * over the list. Effort is off for OpenAI, whose models ignore it.
 *
 * Every message is set from an i18n key, so it follows a language switch; only a server message the
 * app has no words for is shown as the server wrote it. A failed save or list load is announced.
 *
 * The form follows the server's settings until the person changes something; from then on it keeps
 * what they chose (only the status words and the summary follow), until a save succeeds.
 */
import { request } from '../api.js';
import { bindText, formatNumber, t } from '../i18n.js';
import { announce, notify } from '../notify.js';
import { on } from '../state.js';
import { actionButton, el, externalLink, label, uid } from './dom.js';
import { field, setFieldError, setMessage, textOf } from './form-fields.js';
import {
  checksTheKey, depthOf, EFFORTS, errorMessage, errorSpot, isProvider, KEY_LINKS, MODEL_MAX, modelOptions,
  PROVIDER_NAMES, PROVIDERS, saveBody, usesEffort,
} from './llm-model.js';

const MODE_KEYS = { shallow: 'mode_shallow', deep: 'mode_deep' };

export function createModelSection({ adopt }) {
  let settings = null;
  let selected = null; // the provider chosen in the form
  let touched = false; // the person changed the form since it last followed the server
  let busy = false;
  let generation = 0; // only the newest model-list load may write
  let openLoad = false; // the panel was just opened: load the list once, when the settings allow it
  const lists = new Map(); // provider -> [{id, label}] loaded while the panel is open
  const suggested = new Map(); // provider -> the model the server suggests for it
  const picked = new Map(); // provider -> the model the person picked in the list

  const root = el('section', 'panel-section model-section');
  const heading = label('h2', 'panel-h', 'settings_llm_heading');
  heading.tabIndex = -1; // "Open settings" moves the focus here
  const summary = el('p', 'model-summary');

  // Provider: one radio per provider, each with its key status.
  const group = el('fieldset', 'field provider-group');
  group.appendChild(label('legend', 'field-label', 'settings_llm_provider'));
  const segments = el('div', 'segmented');
  const name = uid('provider');
  const radios = PROVIDERS.map((provider) => {
    const option = el('label', 'segment');
    const radio = el('input');
    radio.type = 'radio';
    radio.name = name;
    radio.value = provider;
    const status = el('span', 'segment-status');
    option.append(radio, el('span', 'segment-name', PROVIDER_NAMES[provider]), status);
    segments.appendChild(option);
    radio.addEventListener('change', () => {
      if (radio.checked) choose(provider);
    });
    return { provider, radio, status };
  });
  group.appendChild(segments);

  // API key, for the selected provider.
  const keyInput = el('input', 'field-input');
  keyInput.type = 'password';
  keyInput.autocomplete = 'new-password'; // the browser neither fills a saved password in nor offers to store one
  keyInput.spellcheck = false;
  keyInput.autocapitalize = 'off';
  keyInput.maxLength = 256;
  keyInput.setAttribute('data-field', 'llm_key');
  const keyHelp = el('p', 'field-help key-help');
  const key = field('settings_llm_key', keyInput, { help: keyHelp });

  // Model: the list from the provider, "Load models", and a custom id.
  const select = el('select', 'field-input');
  const load = actionButton('btn btn-quiet btn-small', 'settings_llm_load', { iconName: 'refresh', iconSize: 14 });
  const listStatus = el('p', 'field-help model-list-status');
  listStatus.hidden = true;
  const row = el('div', 'model-row');
  row.append(select, load);
  const model = field('settings_llm_model', select, { holder: row, help: listStatus });
  const customInput = el('input', 'field-input');
  customInput.type = 'text';
  customInput.autocomplete = 'off';
  customInput.spellcheck = false;
  customInput.autocapitalize = 'off';
  customInput.maxLength = MODEL_MAX;
  const custom = field('settings_llm_custom', customInput, { help: label('p', 'field-help', 'settings_llm_custom_help') });

  // Gemini only: the paid switch.
  const paidBox = el('input');
  paidBox.type = 'checkbox';
  const paidLabel = el('label', 'check');
  paidLabel.append(paidBox, label('span', '', 'settings_llm_paid'));
  const paid = el('div', 'field model-paid');
  paid.append(paidLabel, label('p', 'field-help', 'settings_llm_paid_help'));

  // Effort.
  const effortSelect = el('select', 'field-input');
  for (const value of EFFORTS) {
    const option = label('option', '', `settings_effort_${value}`);
    option.value = value;
    effortSelect.appendChild(option);
  }
  const effortHelp = label('p', 'field-help effort-help', 'settings_effort_openai');
  effortHelp.hidden = true;
  const effort = field('settings_effort', effortSelect, { help: effortHelp });

  const save = actionButton('btn btn-primary btn-small', 'settings_llm_save');
  const saveLabel = save.querySelector('.btn-label') || save;
  const actions = el('div', 'conn-actions');
  actions.appendChild(save);
  const message = el('p', 'conn-message');
  message.hidden = true;

  root.append(heading, summary, group, key.wrap, model.wrap, custom.wrap, paid, effort.wrap, actions, message);

  /* The line under Save: a message ({key} or {raw}) with a tone, or nothing. */
  function say(text, tone) {
    message.hidden = !text;
    setMessage(message, text);
    message.setAttribute('data-tone', tone || 'muted');
  }

  function clearErrors() {
    for (const spot of [key, model, custom]) setFieldError(spot, null);
    say(null);
  }

  /* An error next to its field (or under Save), announced once. */
  function showError(code, fieldName, serverMessage) {
    const text = errorMessage(code, serverMessage);
    const spot = errorSpot(code, fieldName);
    if (spot === 'key') setFieldError(key, text);
    else if (spot === 'model') setFieldError(customInput.value.trim() ? custom : model, text);
    else say(text, 'bad');
    announce(textOf(text));
  }

  function setListStatus(text, tone) {
    listStatus.hidden = !text;
    setMessage(listStatus, text);
    listStatus.setAttribute('data-tone', tone || 'muted');
    listStatus.classList.toggle('is-loading', tone === 'loading');
  }

  /* The select's options: the loaded list, plus the model in use when the list does not have it. */
  function renderModels() {
    const options = [...(lists.get(selected) || [])];
    const inUse = settings && selected === settings.provider && typeof settings.model === 'string' ? settings.model : '';
    if (inUse && !options.some((option) => option.id === inUse)) options.unshift({ id: inUse, label: inUse });
    select.replaceChildren(...options.map((option) => {
      const node = el('option', '', option.label);
      node.value = option.id;
      return node;
    }));
    const want = picked.get(selected) || (inUse && selected === settings.provider ? inUse : '') || suggested.get(selected);
    select.value = options.some((option) => option.id === want) ? want : (options[0]?.id || '');
    select.disabled = busy || !settings || options.length === 0;
  }

  /* Every control is off while there are no settings or a save is being answered. */
  function setDisabled() {
    for (const control of [save, load, keyInput, customInput, paidBox, effortSelect]) control.disabled = !settings || busy;
    for (const { radio } of radios) radio.disabled = !settings || busy;
    if (!usesEffort(selected)) effortSelect.disabled = true;
    select.disabled = busy || !settings || select.querySelectorAll('option').length === 0;
  }

  function renderProvider() {
    for (const { provider, radio } of radios) radio.checked = provider === selected;
    bindText(key.caption, 'settings_llm_key', { provider: PROVIDER_NAMES[selected] || '' });
    const link = externalLink(KEY_LINKS[selected], 'key-link');
    keyHelp.replaceChildren();
    if (link) {
      link.textContent = new URL(KEY_LINKS[selected]).host;
      keyHelp.append(label('span', '', 'settings_llm_key_help'), link);
    }
    paid.hidden = selected !== 'gemini';
    effortHelp.hidden = usesEffort(selected);
    if (!usesEffort(selected) && settings && EFFORTS.includes(settings.effort)) effortSelect.value = settings.effort;
    renderModels();
    setDisabled();
  }

  async function loadModels() {
    if (!isProvider(selected)) return;
    const provider = selected;
    const mine = ++generation;
    const typed = keyInput.value.trim();
    setFieldError(key, null);
    setListStatus({ key: 'settings_llm_loading' }, 'loading');
    load.disabled = true;
    try {
      // The typed key goes in the body of this one request and nowhere else.
      const body = await request('POST', '/api/llm/models', typed ? { provider, api_key: typed } : { provider });
      if (mine !== generation || provider !== selected) return;
      lists.set(provider, modelOptions(body?.models));
      if (typeof body?.suggested === 'string') suggested.set(provider, body.suggested);
      setListStatus(lists.get(provider).length ? null : { key: 'settings_llm_none' }, 'muted');
      renderModels();
    } catch (err) {
      // An answer for a provider that is no longer chosen is dropped, like a successful one.
      if (mine !== generation || provider !== selected) return;
      const code = err?.error || '';
      const text = code === 'llm_key_required' ? { key: 'settings_llm_err_list_key' } : errorMessage(code, err?.message);
      if (errorSpot(code, err?.body?.field) === 'key') {
        setListStatus(null);
        setFieldError(key, text);
      } else {
        setListStatus(text, 'bad');
      }
      announce(textOf(text));
    } finally {
      if (mine === generation) load.disabled = busy || !settings;
    }
  }

  function choose(provider) {
    touched = true;
    selected = provider;
    keyInput.value = ''; // a key typed for another provider is never sent for this one
    customInput.value = '';
    clearErrors();
    setListStatus(null);
    renderProvider();
    if (settings?.providers?.[provider] === 'set' && !lists.has(provider)) loadModels();
    else if (!lists.has(provider) && provider !== settings?.provider) setListStatus({ key: 'settings_llm_enter_key' }, 'muted');
  }

  /* The panel opened: load the chosen provider's list once if its key is saved (now, or as soon as
   * the settings are known). */
  function loadOnOpen() {
    if (!openLoad || !settings || busy || !isProvider(selected)) return;
    openLoad = false;
    if (settings.providers?.[selected] === 'set') loadModels();
  }

  async function submit() {
    if (busy || !settings) return;
    clearErrors();
    const form = {
      provider: selected, key: keyInput.value, model: customInput.value.trim() || select.value,
      paid: paidBox.checked, effort: effortSelect.value,
    };
    const { body, error } = saveBody(settings, form);
    if (error) {
      showError(error);
      return;
    }
    if (!Object.keys(body).length) {
      keyInput.value = '';
      say({ key: 'settings_llm_nothing' }, 'muted');
      return;
    }
    busy = true;
    setDisabled();
    bindText(saveLabel, checksTheKey(body) ? 'settings_llm_checking' : 'settings_llm_saving');
    try {
      const answer = await request('PUT', '/api/settings', body);
      const { warning, ...fresh } = answer && typeof answer === 'object' ? answer : {};
      touched = false;
      picked.clear();
      customInput.value = '';
      adopt(fresh);
      const done = warning === 'llm_rate_limited' ? 'settings_llm_saved_rate_limited' : 'settings_llm_saved';
      say({ key: done }, warning ? 'warn' : 'ok');
      if (!warning) notify(t('settings_llm_saved'), { kind: 'success' });
      announce(t(done));
    } catch (err) {
      showError(err?.error || '', err?.body?.field, err?.message);
    } finally {
      keyInput.value = ''; // never keep a typed key, whatever the answer
      busy = false;
      setDisabled();
      bindText(saveLabel, 'settings_llm_save');
    }
  }

  load.addEventListener('click', () => loadModels());
  save.addEventListener('click', () => submit());
  select.addEventListener('change', () => {
    touched = true;
    picked.set(selected, select.value);
  });
  for (const control of [customInput, paidBox, effortSelect, keyInput]) {
    control.addEventListener('input', () => { touched = true; });
    control.addEventListener('change', () => { touched = true; });
  }
  root.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.isComposing && event.target.tagName === 'INPUT' && event.target.type !== 'checkbox'
      && event.target.type !== 'radio') {
      event.preventDefault();
      submit();
    }
  });

  /* Follow a settings body: the status words and the summary always, the form until it is touched. */
  function update(next) {
    settings = next && typeof next === 'object' ? next : null;
    const providers = settings?.providers || {};
    for (const { provider, status } of radios) {
      const saved = providers[provider] === 'set';
      bindText(status, saved ? 'settings_llm_key_saved' : 'settings_llm_no_key');
      status.setAttribute('data-tone', saved ? 'ok' : 'muted');
    }
    const depth = depthOf(settings);
    if (settings && isProvider(settings.provider) && depth) {
      bindText(summary, 'settings_llm_summary', {
        provider: PROVIDER_NAMES[settings.provider], model: String(settings.model || '–'), mode: t(MODE_KEYS[depth.mode]),
        n: depth.rounds ?? '–', s: depth.seconds === null ? '–' : formatNumber(depth.seconds),
      });
    } else {
      bindText(summary, 'settings_llm_summary_none');
    }
    if (settings && !touched) {
      selected = isProvider(settings.provider) ? settings.provider : 'gemini';
      paidBox.checked = Boolean(settings.gemini_paid);
      effortSelect.value = EFFORTS.includes(settings.effort) ? settings.effort : 'medium';
      renderProvider();
    } else if (settings) {
      renderModels();
    }
    setDisabled();
    loadOnOpen();
  }

  // The summary's mode name is a variable: say it again in the new language.
  on('lang', () => {
    if (root.isConnected) update(settings);
  });
  return {
    el: root,
    update,
    opened() {
      openLoad = true;
      loadOnOpen();
    },
    focus() {
      root.scrollIntoView({ block: 'start' });
      heading.focus({ preventScroll: true });
    },
  };
}
