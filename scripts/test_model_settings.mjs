// Tests for the model provider settings and the research-depth notices (Task 4.3) on the fake DOM:
// web/app/js/components/{llm-model,settings-model,mode}.js and the Settings panel around them.
// Run: node scripts/test_model_settings.mjs
//
// The real Settings panel is driven through `POST /api/llm/models` and `PUT /api/settings` answers
// from a scripted fetch (scripts/component_harness.mjs): what each save sends, what the button says
// while the key is checked, where each error is shown, that a typed key is cleared and never stored,
// and how the mode chip, the empty-state notice, the note above the first answer and the Settings
// link under a budget-cut answer follow the settings and the `finish` events.
import assert from 'node:assert/strict';
import {
  dict, document, emit, finish, flush, net, on, panel, replayTurn, setState, test, thread,
} from './component_harness.mjs';
import { setLang } from '../web/app/js/i18n.js';
import { state, STORAGE_KEYS } from '../web/app/js/state.js';
import * as llm from '../web/app/js/components/llm-model.js';

// A localStorage that records every write, to prove no key ever reaches it.
const stored = new Map();
const writes = [];
globalThis.localStorage = {
  getItem: (k) => (stored.has(k) ? stored.get(k) : null),
  setItem: (k, v) => { writes.push([k, String(v)]); stored.set(k, String(v)); },
  removeItem: (k) => stored.delete(k),
};

// The top bar's chip and the toast host, as in index.html.
const topbar = document.createElement('header');
const chip = document.createElement('button');
chip.className = 'chip-mode';
chip.id = 'mode-chip';
chip.hidden = true;
topbar.appendChild(chip);
const toasts = document.createElement('div');
toasts.id = 'toasts';
const announcer = document.createElement('div');
announcer.id = 'announcer';
document.body.append(topbar, toasts, announcer);
const announced = async () => {
  await new Promise((resolve) => setTimeout(resolve, 90)); // notify.js sets the text after 60 ms
  return announcer.textContent;
};
const { modeExplanation } = await import('../web/app/js/components/mode.js');
const { createModelSection } = await import('../web/app/js/components/settings-model.js');

const SHALLOW = {
  alpaca: 'connected', tavily: 'set', sec_user_agent: 'missing', gemini: 'set', trading_enabled: true,
  env_fallbacks: [], provider: 'gemini', model: 'gemini-3.8-flash', effort: 'medium', mode: 'shallow',
  max_rounds: 4, time_budget_sec: 150, gemini_paid: false,
  providers: { gemini: 'set', openai: 'missing', anthropic: 'set' },
};
const DEEP_OPENAI = { ...SHALLOW, provider: 'openai', model: 'gpt-6', mode: 'deep', max_rounds: 12, time_budget_sec: 300,
  providers: { gemini: 'set', openai: 'set', anthropic: 'set' } };
const LISTS = {
  anthropic: { provider: 'anthropic', models: [{ id: 'claude-opus-5-5', label: 'Claude Opus 5.5' }, { id: 'claude-sonnet-5', label: 'Claude Sonnet 5' }], suggested: 'claude-opus-5-5' },
  openai: { provider: 'openai', models: [{ id: 'gpt-6', label: 'gpt-6' }, { id: 'gpt-6-mini', label: 'gpt-6-mini' }], suggested: 'gpt-6' },
};
const models = (provider, over = {}) => ({ method: 'POST', match: '/api/llm/models', status: 200, body: LISTS[provider], ...over });

await test('pure: what a save sends, where an error goes, the key links', () => {
  const s = SHALLOW;
  const save = (form) => llm.saveBody(s, { provider: 'gemini', key: '', model: 'gemini-3.8-flash', paid: false, effort: 'medium', ...form });
  assert.deepEqual(save({}), { body: {} });
  assert.deepEqual(save({ provider: 'openai', model: 'gpt-6', key: ' sk-1 ' }),
    { body: { llm_provider: 'openai', llm_model: 'gpt-6', openai_key: 'sk-1' } });
  assert.deepEqual(save({ provider: 'openai', model: '' }), { error: 'llm_model_required' }, 'a new provider needs its model');
  assert.deepEqual(save({ model: 'gemini-3.8-pro' }), { body: { llm_model: 'gemini-3.8-pro' } });
  assert.deepEqual(save({ key: 'AIza-new' }), { body: { gemini_key: 'AIza-new' } });
  assert.deepEqual(save({ paid: true, effort: 'high' }), { body: { gemini_paid: true, llm_effort: 'high' } });
  assert.deepEqual(save({ provider: 'anthropic', model: 'claude-opus-5-5', paid: true }).body,
    { llm_provider: 'anthropic', llm_model: 'claude-opus-5-5' }, 'the paid switch is Gemini only');
  assert.deepEqual(save({ effort: 'turbo' }), { body: {} });
  // OpenAI models ignore effort: the select is off for OpenAI, and a save there never sends it
  assert.deepEqual(llm.saveBody(DEEP_OPENAI, { provider: 'openai', key: '', model: 'gpt-6', paid: false, effort: 'max' }),
    { body: {} });
  assert.equal(llm.usesEffort('openai'), false);
  assert.equal(llm.usesEffort('gemini') && llm.usesEffort('anthropic'), true);
  assert.deepEqual(save({ model: 'x'.repeat(257) }), { error: 'llm_model_too_long' });
  assert.equal(llm.checksTheKey({ gemini_paid: true, llm_effort: 'low' }), false);
  for (const body of [{ llm_model: 'm' }, { llm_provider: 'openai' }, { anthropic_key: 'k' }]) assert.equal(llm.checksTheKey(body), true);
  assert.equal(llm.errorSpot('llm_invalid_key'), 'key');
  assert.equal(llm.errorSpot('llm_key_required'), 'key');
  assert.equal(llm.errorSpot('llm_model_unavailable'), 'model');
  assert.equal(llm.errorSpot('invalid_value', 'openai_key'), 'key');
  assert.equal(llm.errorSpot('invalid_value', 'api_key'), 'key');
  assert.equal(llm.errorSpot('invalid_value', 'llm_model'), 'model');
  assert.equal(llm.errorSpot('llm_unreachable'), 'save');
  assert.equal(llm.errorSpot('settings_not_saved'), 'save');
  assert.deepEqual(llm.errorMessage('llm_invalid_key', 'server words'), { key: 'settings_llm_err_invalid_key' });
  assert.deepEqual(llm.errorMessage('settings_not_saved', 'The settings could not be saved.'), { raw: 'The settings could not be saved.' });
  assert.deepEqual(llm.errorMessage('network_error'), { key: 'err_generic' });
  assert.deepEqual(llm.modelOptions([{ id: 'a', label: 'A' }, { id: 'a' }, { id: '' }, { id: 7 }, null, { id: 'b' }]),
    [{ id: 'a', label: 'A (a)' }, { id: 'b', label: 'b' }]);
  assert.deepEqual(llm.depthOf({ mode: 'shallow', max_rounds: 4, time_budget_sec: 150 }), { mode: 'shallow', rounds: 4, seconds: 150 });
  assert.equal(llm.depthOf({ usage: {} }), null, 'an emergency finish says nothing about the depth');
  assert.deepEqual(Object.values(llm.KEY_LINKS).map((u) => new URL(u).host),
    ['aistudio.google.com', 'platform.openai.com', 'console.anthropic.com']);
  assert.deepEqual(llm.EFFORTS, ['low', 'medium', 'high', 'xhigh', 'max']);
  for (const lang of ['en', 'ko']) {
    for (const e of llm.EFFORTS) assert.ok(dict[lang][`settings_effort_${e}`], `${lang} ${e}`);
    for (const key of Object.values(llm.LLM_ERRORS)) assert.ok(dict[lang][key], `${lang} ${key}`);
  }
});

// The Settings panel, mounted as the shell does.
const view = document.createElement('section');
document.body.appendChild(view);
panel('settings').mount(view);
const section = () => view.querySelector('.model-section');
const radio = (provider) => section().querySelectorAll('input').find((i) => i.type === 'radio' && i.value === provider);
const keyInput = () => section().querySelectorAll('input').find((i) => i.type === 'password');
const customInput = () => section().querySelectorAll('input').find((i) => i.type === 'text');
const paidBox = () => section().querySelectorAll('input').find((i) => i.type === 'checkbox');
const selects = () => section().querySelectorAll('select');
const saveButton = () => section().querySelector('.btn-primary');
const loadButton = () => section().querySelectorAll('button').find((b) => b.textContent === 'Load models');
const errors = () => section().querySelectorAll('.field-error').filter((e) => !e.hidden).map((e) => e.textContent);
const message = () => section().querySelector('.conn-message');
const choose = (provider) => { radio(provider).checked = true; radio(provider).dispatch('change'); };
const puts = () => net.calls.filter((c) => c.method === 'PUT');

await test('the model section shows what is in use, the key status of each provider and where keys are made', () => {
  setState({ settings: SHALLOW });
  assert.equal(section().querySelector('.model-summary').textContent,
    'Using Gemini · gemini-3.8-flash · Shallow mode (up to 4 steps, 150 s per reply)');
  assert.deepEqual(section().querySelectorAll('.segment').map((s) => s.textContent),
    ['GeminiKey saved', 'OpenAINo key', 'AnthropicKey saved']);
  assert.equal(radio('gemini').checked, true);
  const key = keyInput();
  assert.equal(key.autocomplete, 'new-password');
  assert.equal(key.value ?? '', '', 'never pre-filled');
  const link = section().querySelector('.key-link');
  assert.equal(link.getAttribute('href'), 'https://aistudio.google.com/apikey');
  assert.equal(link.getAttribute('target'), '_blank');
  assert.ok(link.getAttribute('rel').includes('noopener') && link.getAttribute('rel').includes('noreferrer'));
  assert.equal(link.textContent, 'aistudio.google.com');
  assert.equal(section().querySelector('.model-paid').hidden, false);
  assert.equal(paidBox().checked, false);
  assert.equal(selects()[0].value, 'gemini-3.8-flash', 'the model in use, before any list is loaded');
  assert.equal(selects()[1].value, 'medium');
  assert.deepEqual(selects()[1].querySelectorAll('option').map((o) => o.textContent), ['Low', 'Medium', 'High', 'Extra high', 'Maximum']);
  // the Alpaca row says plainly when the keys are not for a paper account
  setState({ settings: { ...SHALLOW, alpaca: 'not_paper' } });
  const warning = view.querySelector('.conn-warning');
  assert.equal(warning.hidden, false);
  assert.equal(warning.getAttribute('data-tone'), 'bad');
  assert.ok(warning.textContent.includes('This is not a paper account, so trading is disabled'));
  setState({ settings: SHALLOW });
  assert.equal(view.querySelector('.conn-warning').hidden, true);
});

await test('choosing a provider with a saved key loads its models; the newest load wins', async () => {
  net.calls.length = 0;
  let releaseOld;
  net.routes = [
    models('anthropic', { once: true, gate: new Promise((r) => { releaseOld = r; }),
      body: { provider: 'anthropic', models: [{ id: 'stale-model' }], suggested: 'stale-model' } }),
    models('anthropic'),
  ];
  choose('anthropic');
  await flush();
  assert.deepEqual(net.calls.map((c) => [c.method, c.path, c.body]), [['POST', '/api/llm/models', { provider: 'anthropic' }]]);
  assert.ok(section().querySelector('.model-list-status').textContent.includes('Loading'));
  assert.equal(loadButton().disabled, true, 'one load at a time from the button');
  choose('openai'); // no saved key: nothing to load
  choose('anthropic'); // a second load overtakes the first
  await flush();
  releaseOld();
  await flush();
  const options = selects()[0].querySelectorAll('option').map((o) => o.value);
  assert.deepEqual(options, ['claude-opus-5-5', 'claude-sonnet-5']);
  assert.equal(selects()[0].value, 'claude-opus-5-5', 'the suggested model is preselected');
  assert.equal(section().querySelector('.model-paid').hidden, true, 'the paid switch is for Gemini only');
  assert.equal(section().querySelector('.key-link').getAttribute('href'), 'https://console.anthropic.com/settings/keys');
  // a settings refresh does not undo what the person chose
  setState({ settings: { ...SHALLOW } });
  assert.equal(radio('anthropic').checked, true);
});

await test('a provider without a key: type it, load its models with it, save; the key is checked and then cleared', async () => {
  net.calls.length = 0;
  net.routes = [];
  choose('openai');
  await flush();
  assert.equal(net.calls.length, 0, 'no saved key: nothing is loaded by itself');
  assert.ok(section().querySelector('.model-list-status').textContent.includes('Enter this provider’s API key'));
  keyInput().value = 'sk-test-typed';
  net.routes = [models('openai')];
  loadButton().click();
  await flush();
  assert.deepEqual(net.calls[0].body, { provider: 'openai', api_key: 'sk-test-typed' }, 'the typed key, in the body only');
  assert.equal(net.calls[0].path, '/api/llm/models');
  assert.equal(keyInput().value, 'sk-test-typed', 'still there for the save');
  selects()[0].value = 'gpt-6-mini';
  selects()[0].dispatch('change');
  let release;
  net.routes = [{ method: 'PUT', match: '/api/settings', status: 200, gate: new Promise((r) => { release = r; }),
    body: { ...DEEP_OPENAI, model: 'gpt-6-mini' } }];
  saveButton().click();
  await flush();
  const button = section().querySelector('.btn-primary');
  assert.equal(button.disabled, true);
  assert.equal(button.textContent, 'Checking the key…');
  assert.deepEqual(puts().at(-1).body, { llm_provider: 'openai', llm_model: 'gpt-6-mini', openai_key: 'sk-test-typed' });
  release();
  await flush();
  assert.equal(keyInput().value, '', 'cleared after the answer');
  assert.equal(button.disabled, false);
  assert.equal(button.textContent, 'Save model settings');
  assert.equal(state.settings.provider, 'openai');
  assert.equal(section().querySelector('.model-summary').textContent,
    'Using OpenAI · gpt-6-mini · Deep mode (up to 12 steps, 300 s per reply)');
  assert.equal(message().textContent, 'Model settings saved.');
  assert.ok(toasts.textContent.includes('Model settings saved.'));
  assert.equal(chip.textContent, 'Deep mode', 'the top bar follows');
  assert.ok(!JSON.stringify(state).includes('sk-test-typed'), 'the key is not in the app state');
  assert.ok(!writes.some(([, v]) => v.includes('sk-test')), 'nor in localStorage');
});

await test('every save result lands next to its field, and the key is cleared each time', async () => {
  setState({ settings: DEEP_OPENAI });
  const attempt = async (status, body, typed = 'sk-bad') => {
    net.routes = [{ method: 'PUT', match: '/api/settings', status, body }];
    keyInput().value = typed;
    saveButton().click();
    await flush();
    assert.equal(keyInput().value, '', `${body.error}: cleared`);
  };
  await attempt(400, { error: 'llm_invalid_key', message: 'The provider rejected this API key.' });
  assert.deepEqual(errors(), [dict.en.settings_llm_err_invalid_key]);
  assert.equal(keyInput().getAttribute('aria-invalid'), 'true');
  assert.equal(state.settings.provider, 'openai', 'nothing changed');
  await attempt(400, { error: 'invalid_value', field: 'openai_key', message: 'bad characters' });
  assert.deepEqual(errors(), [dict.en.settings_err_invalid_value]);
  await attempt(502, { error: 'llm_unreachable', message: 'x' });
  assert.deepEqual(errors(), []);
  assert.equal(message().textContent, dict.en.settings_llm_err_unreachable);
  assert.equal(message().getAttribute('data-tone'), 'bad');
  await attempt(500, { error: 'settings_not_saved', message: 'The settings could not be saved.' });
  assert.equal(message().textContent, 'The settings could not be saved.');
  // the model: picked from the list, refused by the provider
  customInput().value = 'gpt-nope';
  await attempt(400, { error: 'llm_model_unavailable', message: 'x' }, '');
  assert.deepEqual(errors(), [dict.en.settings_llm_err_model_unavailable]);
  assert.equal(customInput().getAttribute('aria-invalid'), 'true');
  assert.deepEqual(puts().at(-1).body, { llm_model: 'gpt-nope' }, 'a custom id wins over the list');
  customInput().value = '';
  // rate limited: saved, with a note
  await attempt(200, { ...DEEP_OPENAI, warning: 'llm_rate_limited' });
  assert.equal(message().getAttribute('data-tone'), 'warn');
  assert.ok(message().textContent.startsWith('Saved. The key works'));
  assert.equal(state.settings.warning, undefined, 'the warning is not kept as a setting');
  // nothing to save
  net.calls.length = 0;
  saveButton().click();
  await flush();
  assert.equal(net.calls.length, 0);
  assert.equal(message().textContent, 'Nothing changed.');
});

await test('the Gemini paid switch and effort save without a key check', async () => {
  setState({ settings: SHALLOW });
  assert.equal(radio('gemini').checked, true, 'the form follows the saved settings again');
  paidBox().checked = true;
  paidBox().dispatch('change');
  selects()[1].value = 'high';
  selects()[1].dispatch('change');
  let release;
  net.routes = [{ method: 'PUT', match: '/api/settings', status: 200, gate: new Promise((r) => { release = r; }),
    body: { ...SHALLOW, gemini_paid: true, effort: 'high', mode: 'deep', max_rounds: 12, time_budget_sec: 300 } }];
  saveButton().click();
  await flush();
  assert.equal(section().querySelector('.btn-primary').textContent, 'Saving…');
  assert.deepEqual(puts().at(-1).body, { gemini_paid: true, llm_effort: 'high' });
  release();
  await flush();
  assert.equal(chip.textContent, 'Deep mode');
  assert.equal(section().querySelector('.model-summary').textContent,
    'Using Gemini · gemini-3.8-flash · Deep mode (up to 12 steps, 300 s per reply)');
});

await test('model list errors: a key problem by the key field, anything else under the list', async () => {
  setState({ settings: SHALLOW });
  choose('anthropic');
  const listWith = async (status, body) => {
    net.routes = [{ method: 'POST', match: '/api/llm/models', status, body }];
    loadButton().click();
    await flush();
  };
  await listWith(400, { error: 'llm_invalid_key', message: 'x' });
  assert.deepEqual(errors(), [dict.en.settings_llm_err_invalid_key]);
  await listWith(400, { error: 'llm_key_required', message: 'x' });
  assert.deepEqual(errors(), [dict.en.settings_llm_err_list_key]);
  for (const [status, code] of [[502, 'llm_list_failed'], [502, 'llm_unreachable'], [429, 'llm_rate_limited']]) {
    await listWith(status, { error: code, message: 'x' });
    const status_ = section().querySelector('.model-list-status');
    assert.equal(status_.textContent, dict.en[llm.LLM_ERRORS[code]], code);
    assert.equal(status_.getAttribute('data-tone'), 'bad');
    assert.deepEqual(errors(), []);
  }
});

await test('the mode chip: the word, the explanation and the way to Settings', () => {
  setState({ settings: SHALLOW });
  assert.equal(chip.hidden, false);
  assert.equal(chip.textContent, 'Shallow mode');
  assert.ok(chip.classList.contains('is-shallow'));
  const box = modeExplanation();
  assert.ok(box.textContent.includes('Research is limited to 4 steps per reply because of the Gemini free tier’s rate limits'));
  assert.ok(box.textContent.includes('add a paid API key, or mark your Gemini key as paid in Settings'));
  const opened = [];
  const stop = on('panel:open', (name) => opened.push(name));
  box.querySelector('button').click();
  stop();
  assert.deepEqual(opened, ['settings']);
  assert.equal(document.activeElement, section().querySelector('.panel-h'), 'the focus lands on the model section');
  setLang('ko');
  assert.equal(chip.textContent, '얕은 리서치 모드');
  assert.equal(section().querySelector('.model-summary').textContent,
    '사용 중: Gemini · gemini-3.8-flash · 얕은 리서치 모드 (답변당 최대 4단계, 150초)', 'the mode name follows the language');
  setLang('en');
  setState({ settings: DEEP_OPENAI });
  assert.equal(chip.textContent, 'Deep mode');
  assert.equal(chip.classList.contains('is-shallow'), false);
  assert.ok(modeExplanation().textContent.includes('up to 12 steps and 300 s per reply'));
});

await test('the empty state shows the shallow notice with a Settings link, and drops it in deep mode', () => {
  const empty = document.createElement('section');
  const slot = document.createElement('div');
  slot.className = 'empty-notice';
  empty.appendChild(slot);
  document.body.appendChild(empty);
  setState({ settings: SHALLOW });
  emit('thread:empty', slot);
  assert.equal(slot.hidden, false);
  assert.ok(slot.textContent.includes('Research is limited to 4 steps per reply'));
  assert.equal(slot.querySelector('button').textContent, 'Open settings');
  setState({ settings: DEEP_OPENAI });
  assert.equal(slot.hidden, true);
  assert.equal(slot.textContent, '');
  empty.remove();
});

await test('shallow note above the first answer: once per chat, dismissible, remembered in this browser', () => {
  thread.replaceChildren();
  setState({ settings: SHALLOW });
  const first = replayTurn('s1');
  emit('turn:start', { turn: first, sessionId: 's1' });
  const notes = () => thread.querySelectorAll('.mode-note');
  assert.equal(notes().length, 1);
  assert.equal(thread.children.indexOf(notes()[0]), thread.children.indexOf(first.el) - 1, 'right above the first answer');
  assert.equal(notes()[0].textContent, 'Shallow mode: up to 4 research steps per reply.');
  const second = replayTurn('s1');
  emit('turn:start', { turn: second, sessionId: 's1' });
  assert.equal(notes().length, 1, 'once per chat');
  setState({ settings: DEEP_OPENAI });
  assert.equal(notes().length, 0, 'gone in deep mode');
  setState({ settings: SHALLOW });
  emit('turn:start', { turn: second, sessionId: 's1' });
  notes()[0].querySelector('button').click();
  assert.equal(notes().length, 0);
  assert.equal(stored.get(STORAGE_KEYS.shallowNote), 'dismissed');
  assert.equal(STORAGE_KEYS.shallowNote, 'finagent-app.shallow-note');
  emit('turn:start', { turn: second, sessionId: 's1' });
  assert.equal(notes().length, 0, 'remembered');
  stored.delete(STORAGE_KEYS.shallowNote);
  thread.replaceChildren();
});

await test('an answer cut by the shallow budget gets the Settings link; the finish event moves the chip', () => {
  thread.replaceChildren();
  setState({ settings: SHALLOW });
  const cut = replayTurn('s1');
  const refreshes = [];
  const stop = on('settings:refresh', () => refreshes.push(1));
  emit('turn:end', { turn: cut, outcome: 'finish', finish: { mode: 'shallow', max_rounds: 4, time_budget_sec: 150, budget_exhausted: true } });
  const hint = cut.slots.status.querySelector('.mode-hint');
  assert.ok(hint);
  assert.equal(hint.textContent, 'Shallow modeOpen settings');
  assert.equal(refreshes.length, 0);
  const fine = replayTurn('s1');
  emit('turn:end', { turn: fine, finish: { mode: 'shallow', max_rounds: 4, time_budget_sec: 150, budget_exhausted: false } });
  assert.equal(fine.slots.status.querySelector('.mode-hint'), null);
  // the provider was switched elsewhere: the reply ran in deep mode
  const deep = replayTurn('s1');
  emit('turn:end', { turn: deep, finish: { mode: 'deep', max_rounds: 12, time_budget_sec: 300, budget_exhausted: true } });
  assert.equal(deep.slots.status.querySelector('.mode-hint'), null, 'deep mode has no shallow hint');
  assert.equal(chip.textContent, 'Deep mode');
  assert.equal(refreshes.length, 1, 'the settings are read again');
  stop();
  // history: the stored answer's meta says it was cut
  thread.replaceChildren();
  setState({ settings: SHALLOW });
  const a = replayTurn('s1');
  const b = replayTurn('s1');
  emit('session:loaded', { session: { id: 's1' }, messages: [
    { role: 'user', content: 'q', meta: null }, { role: 'assistant', content: 'x', meta: { mode: 'shallow', budget_exhausted: true } },
    { role: 'user', content: 'q', meta: null }, { role: 'assistant', content: 'y', meta: { mode: 'shallow', budget_exhausted: false } }] });
  assert.ok(a.slots.status.querySelector('.mode-hint'));
  assert.equal(b.slots.status.querySelector('.mode-hint'), null);
  assert.equal(thread.querySelectorAll('.mode-note').length, 1, 'a reopened chat gets the note too');
  thread.replaceChildren();
});

await test('a model list that answers after the provider changed is dropped, success or error', async () => {
  // A fresh section each time: nothing loaded or cached yet.
  const fresh = () => {
    const made = createModelSection({ adopt: (body) => setState({ settings: body }) });
    document.body.appendChild(made.el);
    made.update(SHALLOW);
    const root = made.el;
    const pick = (provider) => {
      const input = root.querySelectorAll('input').find((i) => i.type === 'radio' && i.value === provider);
      input.checked = true;
      input.dispatch('change');
    };
    const status = () => root.querySelector('.model-list-status');
    const shownErrors = () => root.querySelectorAll('.field-error').filter((e) => !e.hidden).map((e) => e.textContent);
    return { root, pick, status, shownErrors };
  };
  const hint = 'Enter this provider’s API key, then load its models.';

  // success: Anthropic's list arrives after OpenAI (no key, so no load of its own) was chosen
  let release;
  let s = fresh();
  net.routes = [models('anthropic', { gate: new Promise((r) => { release = r; }) })];
  s.pick('anthropic');
  await flush();
  s.pick('openai');
  assert.equal(s.status().textContent, hint);
  release();
  await flush();
  assert.equal(s.status().textContent, hint, 'the late list does not touch OpenAI\'s state');
  assert.deepEqual(s.root.querySelector('select').querySelectorAll('option').map((o) => o.value), []);
  s.root.remove();

  // error: Anthropic's refusal arrives after OpenAI was chosen
  s = fresh();
  net.routes = [{ method: 'POST', match: '/api/llm/models', status: 400, gate: new Promise((r) => { release = r; }),
    body: { error: 'llm_invalid_key', message: 'x' } }];
  s.pick('anthropic');
  await flush();
  s.pick('openai');
  release();
  await flush();
  assert.deepEqual(s.shownErrors(), [], 'no stale error at OpenAI\'s key field');
  assert.equal(s.status().textContent, hint);
  s.root.remove();
  setState({ settings: SHALLOW });
});

await test('opening the panel loads the saved provider\'s models once per opening', async () => {
  // the form follows the server again
  net.routes = [{ method: 'PUT', match: '/api/settings', status: 200, body: SHALLOW }];
  setState({ settings: { ...SHALLOW, effort: 'low' } });
  await flush();
  setState({ settings: SHALLOW });
  section().querySelector('.btn-primary').click(); // nothing to save, but the form is in sync
  await flush();
  net.calls.length = 0;
  net.routes = [{ method: 'POST', match: '/api/llm/models', status: 200,
    body: { provider: 'gemini', models: [{ id: 'gemini-3.8-pro' }, { id: 'gemini-3.8-flash' }], suggested: 'gemini-3.8-flash' } },
  { method: 'GET', match: '/api/settings', status: 200, body: SHALLOW }];
  panel('settings').show(view);
  await flush();
  const lists = () => net.calls.filter((c) => c.path === '/api/llm/models');
  assert.deepEqual(lists().map((c) => c.body), [{ provider: 'gemini' }], 'once, with the saved key');
  assert.deepEqual(selects()[0].querySelectorAll('option').map((o) => o.value), ['gemini-3.8-pro', 'gemini-3.8-flash']);
  assert.equal(selects()[0].value, 'gemini-3.8-flash');
  setState({ settings: { ...SHALLOW } }); // a settings refresh is not an opening
  await flush();
  assert.equal(lists().length, 1);
  panel('settings').show(view);
  await flush();
  assert.equal(lists().length, 2, 'the next opening loads again');
  // no saved key for the provider in use: nothing is loaded
  setState({ settings: { ...SHALLOW, providers: { gemini: 'missing', openai: 'missing', anthropic: 'missing' } } });
  panel('settings').show(view);
  await flush();
  assert.equal(lists().length, 2);
  setState({ settings: SHALLOW });
});

await test('effort is off for OpenAI, with a note; on for the others', () => {
  setState({ settings: SHALLOW });
  const effortHelp = () => section().querySelector('.effort-help');
  assert.equal(selects()[1].disabled, false);
  assert.equal(effortHelp().hidden, true);
  choose('openai');
  assert.equal(selects()[1].disabled, true);
  assert.equal(effortHelp().hidden, false);
  assert.equal(effortHelp().textContent, 'OpenAI models ignore this setting.');
  choose('anthropic');
  assert.equal(selects()[1].disabled, false);
  assert.deepEqual(selects()[1].querySelectorAll('option').map((o) => o.value), ['low', 'medium', 'high', 'xhigh', 'max']);
  setState({ settings: SHALLOW });
});

await test('messages follow a language switch, and failures are announced', async () => {
  setState({ settings: DEEP_OPENAI });
  net.routes = [{ method: 'PUT', match: '/api/settings', status: 400, body: { error: 'llm_invalid_key', message: 'x' } }];
  keyInput().value = 'sk-bad';
  saveButton().click();
  await flush();
  assert.deepEqual(errors(), [dict.en.settings_llm_err_invalid_key]);
  assert.equal(await announced(), dict.en.settings_llm_err_invalid_key, 'a failed save is announced');
  setLang('ko');
  assert.deepEqual(errors(), [dict.ko.settings_llm_err_invalid_key]);
  setLang('en');
  // the message under Save
  net.routes = [{ method: 'PUT', match: '/api/settings', status: 502, body: { error: 'llm_unreachable', message: 'x' } }];
  keyInput().value = 'sk-bad';
  saveButton().click();
  await flush();
  setLang('ko');
  assert.equal(message().textContent, dict.ko.settings_llm_err_unreachable);
  setLang('en');
  // a server message the app has no words for stays as the server wrote it, in any language
  net.routes = [{ method: 'PUT', match: '/api/settings', status: 500, body: { error: 'settings_not_saved', message: 'The settings could not be saved.' } }];
  keyInput().value = 'sk-bad';
  saveButton().click();
  await flush();
  setLang('ko');
  assert.equal(message().textContent, 'The settings could not be saved.');
  setLang('en');
  // the model list's status line, and its announcement
  setState({ settings: SHALLOW });
  choose('anthropic');
  net.routes = [{ method: 'POST', match: '/api/llm/models', status: 502, body: { error: 'llm_list_failed', message: 'x' } }];
  loadButton().click();
  await flush();
  assert.equal(await announced(), dict.en.settings_llm_err_list_failed, 'a failed list load is announced');
  setLang('ko');
  assert.equal(section().querySelector('.model-list-status').textContent, dict.ko.settings_llm_err_list_failed);
  setLang('en');
  setState({ settings: SHALLOW });
});

await test('the mode chip carries its name for every width: an icon, a label and the accessible name', () => {
  setState({ settings: SHALLOW });
  assert.equal(chip.getAttribute('aria-label'), 'Shallow mode');
  assert.equal(chip.querySelector('.chip-mode-label').textContent, 'Shallow mode');
  assert.ok(chip.querySelector('.chip-mode-icon'));
  const shallowIcon = chip.querySelector('.chip-mode-icon').getAttribute('data-icon');
  setState({ settings: DEEP_OPENAI });
  assert.equal(chip.getAttribute('aria-label'), 'Deep mode');
  assert.notEqual(chip.querySelector('.chip-mode-icon').getAttribute('data-icon'), shallowIcon, 'the shape differs, not only the colour');
  setLang('ko');
  assert.equal(chip.getAttribute('aria-label'), '심층 리서치 모드');
  setLang('en');
  setState({ settings: SHALLOW });
});

await test('no typed key is ever written to storage', () => {
  assert.deepEqual(writes.map(([k]) => k).filter((k) => k !== STORAGE_KEYS.lang), [STORAGE_KEYS.shallowNote]);
});

finish('model settings checks');
