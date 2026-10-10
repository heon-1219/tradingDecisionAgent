// Tests for the Settings panel and its neighbours on the fake DOM (final review, part 2a):
// the key rows, the trading switch, the kill switch and a mandate save say what failed, announce it
// once and re-translate it; typed keys are cleared after every answer and never reach any storage;
// an automatic model-list load never sends a typed key; "Model settings" scrolls the panel, never the
// document; the DEMO badge and the demo note follow `GET /api/settings` `demo`.
// Run: node scripts/test_settings_panel.mjs
import assert from 'node:assert/strict';

// Storage spies, in place before any app module loads: every write to any storage API is recorded,
// whichever name it is reached by (localStorage, sessionStorage, Storage.prototype, document.cookie).
const writes = [];
class SpyStorage {
  constructor() { this.map = new Map(); }
  getItem(k) { return this.map.has(k) ? this.map.get(k) : null; }
  setItem(k, v) { writes.push(['storage', String(k), String(v)]); this.map.set(String(k), String(v)); }
  removeItem(k) { this.map.delete(k); }
  clear() { this.map.clear(); }
}
globalThis.Storage = SpyStorage;
globalThis.localStorage = new SpyStorage();
globalThis.sessionStorage = new SpyStorage();

const harness = await import('./component_harness.mjs');
const { dict, document, emit, finish, flush, net, panel, setState, test } = harness;
Object.defineProperty(document, 'cookie', {
  get: () => '',
  set: (value) => writes.push(['cookie', '', String(value)]),
});
const { setLang } = await import('../web/app/js/i18n.js');
const { saveMandate } = await import('../web/app/js/components/mandate.js');
const { modeExplanation } = await import('../web/app/js/components/mode.js');
await import('../web/app/js/components/demo.js');

// The page pieces the panel and the notices use, as in index.html.
const topbar = document.createElement('header');
for (const [tag, id] of [['span', 'demo-badge'], ['button', 'mode-chip']]) {
  const node = document.createElement(tag);
  node.id = id;
  node.hidden = true;
  topbar.appendChild(node);
}
const announcer = document.createElement('div');
announcer.id = 'announcer';
const panelBody = document.createElement('div');
panelBody.className = 'panel-body';
const view = document.createElement('section');
panelBody.appendChild(view);
document.body.append(topbar, announcer, panelBody);
await import('../web/app/js/components/settings.js');
panel('settings').mount(view);

const announced = async () => {
  await new Promise((resolve) => setTimeout(resolve, 90)); // notify.js sets the text after 60 ms
  return announcer.textContent;
};
const SETTINGS = {
  alpaca: 'connected', tavily: 'set', sec_user_agent: 'missing', gemini: 'set', trading_enabled: true, env_fallbacks: [],
  provider: 'gemini', model: 'gemini-3.8-flash', effort: 'medium', mode: 'shallow', max_rounds: 4, time_budget_sec: 150,
  gemini_paid: false, providers: { gemini: 'set', openai: 'missing', anthropic: 'missing' },
};
const TYPED = ['typed-key-id-1', 'typed-secret-2', 'typed-tavily-3', 'typed-model-key-4'];
const rows = () => view.querySelectorAll('.conn');
const rowInputs = (row) => row.querySelectorAll('input');
const rowMessage = (row) => row.querySelector('.conn-message');
const rowButton = (row, cls) => row.querySelector(`.${cls}`);
const put = (status, body) => [{ method: 'PUT', match: '/api/settings', status, body }];
setState({ settings: SETTINGS });

await test('key rows: emptied after the answer, success or failure; a failure is said, announced and re-translated', async () => {
  const [alpaca, tavily] = rows();
  rowButton(alpaca, 'btn-link').click(); // Change
  const [id, secret] = rowInputs(alpaca);
  // success
  id.value = TYPED[0];
  secret.value = TYPED[1];
  net.routes = put(200, SETTINGS);
  rowButton(alpaca, 'btn-primary').click();
  await flush();
  assert.equal(id.value, '');
  assert.equal(secret.value, '');
  assert.equal(rowMessage(alpaca).textContent, 'Saved');
  setLang('ko');
  assert.equal(rowMessage(alpaca).textContent, dict.ko.settings_saved);
  setLang('en');
  // failure
  id.value = TYPED[0];
  secret.value = TYPED[1];
  net.routes = put(400, { error: 'alpaca_invalid', message: 'Alpaca rejected these keys.' });
  rowButton(alpaca, 'btn-primary').click();
  await flush();
  assert.equal(id.value, '', 'cleared after a failure too');
  assert.equal(secret.value, '');
  assert.equal(rowMessage(alpaca).textContent, dict.en.settings_err_alpaca_invalid);
  assert.equal(await announced(), dict.en.settings_err_alpaca_invalid);
  setLang('ko');
  assert.equal(rowMessage(alpaca).textContent, dict.ko.settings_err_alpaca_invalid);
  setLang('en');
  // a value the server cannot save: the field's name and the reason, both re-translated
  rowButton(tavily, 'btn-link').click();
  rowInputs(tavily)[0].value = TYPED[2];
  net.routes = put(400, { error: 'invalid_value', field: 'tavily_key', message: 'x' });
  rowButton(tavily, 'btn-primary').click();
  await flush();
  assert.equal(rowInputs(tavily)[0].value, '');
  assert.equal(rowMessage(tavily).textContent, `${dict.en.settings_tavily_key}: ${dict.en.settings_err_invalid_value}`);
  setLang('ko');
  assert.equal(rowMessage(tavily).textContent, `${dict.ko.settings_tavily_key}: ${dict.ko.settings_err_invalid_value}`);
  setLang('en');
  // an error the app has no words for: the server's text, announced
  rowInputs(tavily)[0].value = TYPED[2];
  net.routes = put(500, { error: 'settings_not_saved', message: 'The settings could not be saved.' });
  rowButton(tavily, 'btn-primary').click();
  await flush();
  assert.equal(rowMessage(tavily).textContent, 'The settings could not be saved.');
  assert.equal(await announced(), 'The settings could not be saved.');
});

await test('the trading switch: a failure is said, announced and re-translated', async () => {
  const toggle = view.querySelector('.switch');
  net.routes = put(500, { error: 'settings_not_saved', message: 'The settings could not be saved.' });
  toggle.click();
  await flush();
  const message = toggle.parentNode.parentNode.querySelectorAll('.conn-message').find((m) => !m.hidden);
  assert.equal(message.textContent, 'The settings could not be saved.');
  assert.equal(await announced(), 'The settings could not be saved.');
  net.routes = put(400, { error: 'invalid_value', field: 'tavily_key', message: 'x' });
  toggle.click();
  await flush();
  setLang('ko');
  assert.equal(message.textContent, `${dict.ko.settings_tavily_key}: ${dict.ko.settings_err_invalid_value}`);
  setLang('en');
});

await test('re-enabling trading: a kill press meanwhile is said and the settings re-read; a broker left suspended is noted', async () => {
  const { on } = harness;
  const { state } = await import('../web/app/js/state.js');
  const toggle = view.querySelector('.switch');
  const section = toggle.parentNode.parentNode;
  const visible = (cls) => section.querySelectorAll(cls).filter((n) => !n.hidden);
  const refreshes = [];
  const stop = on('settings:refresh', () => refreshes.push(1));
  setState({ settings: { ...SETTINGS, trading_enabled: false } });
  net.routes = put(409, { error: 'kill_switch_pressed', message: 'The kill switch was pressed while these settings were being saved.' });
  toggle.click();
  await flush();
  const message = visible('.conn-message')[0];
  assert.equal(message.textContent, dict.en.settings_err_kill_pressed);
  assert.equal(await announced(), dict.en.settings_err_kill_pressed);
  assert.equal(refreshes.length, 1, 'the settings are read again');
  setLang('ko');
  assert.equal(message.textContent, dict.ko.settings_err_kill_pressed);
  setLang('en');
  stop();
  // trading back on, but the broker could not be resumed: a note that stays
  net.routes = put(200, { ...SETTINGS, trading_enabled: true, broker_resumed: false });
  toggle.click();
  await flush();
  assert.equal(state.settings.broker_resumed, undefined, 'not kept as a setting');
  const note = () => visible('.broker-note')[0];
  assert.ok(note(), 'the note is shown');
  assert.equal(note().getAttribute('data-tone'), 'warn');
  assert.ok(note().textContent.includes(dict.en.settings_broker_still_suspended));
  setState({ settings: { ...SETTINGS, trading_enabled: true } }); // a later refresh does not hide it
  assert.ok(note());
  setState({ settings: { ...SETTINGS, trading_enabled: false } }); // trading off again: gone
  assert.equal(note(), undefined);
  net.routes = put(200, { ...SETTINGS, trading_enabled: true, broker_resumed: true });
  toggle.click();
  await flush();
  assert.equal(note(), undefined, 'resumed: no note');
});

await test('the kill switch: a failure is announced', async () => {
  const kill = view.querySelector('.kill');
  kill.querySelector('.btn-danger').click();
  net.routes = [{ method: 'POST', match: '/api/kill-switch', status: 502, body: { error: 'kill_failed', message: 'Trading could not be stopped.' } }];
  kill.querySelector('.btn-danger-solid').click();
  await flush();
  assert.equal(kill.querySelector('.conn-message').textContent, 'Trading could not be stopped.');
  assert.equal(await announced(), 'Trading could not be stopped.');
});

await test('a mandate save that fails for a reason other than its fields is announced', async () => {
  const said = [];
  const form = { setBusy() {}, setErrors() {}, setStatus: (key, tone, text) => said.push([key, tone, text]) };
  net.routes = [{ method: 'PUT', match: '/api/mandate/', status: 500, body: { error: 'mandate_failed', message: 'The mandate could not be saved.' } }];
  assert.equal(await saveMandate('s1', { budget_usd: '1000' }, form), null);
  assert.deepEqual(said, [[null, 'bad', 'The mandate could not be saved.']]);
  assert.equal(await announced(), 'The mandate could not be saved.');
});

await test('an automatic model-list load never sends a typed key; only "Load models" does', async () => {
  setState({ settings: SETTINGS });
  const section = view.querySelector('.model-section');
  const key = section.querySelectorAll('input').find((i) => i.type === 'password');
  key.value = TYPED[3]; // typed, not saved
  net.calls.length = 0;
  net.routes = [{ method: 'POST', match: '/api/llm/models', status: 200,
    body: { provider: 'gemini', models: [{ id: 'gemini-3.8-flash' }], suggested: 'gemini-3.8-flash' } }];
  panel('settings').show(view); // the panel opens (or its tab is selected again)
  await flush();
  setState({ settings: { ...SETTINGS } }); // and the settings come back
  await flush();
  panel('settings').show(view);
  await flush();
  const lists = net.calls.filter((c) => c.path === '/api/llm/models');
  assert.deepEqual(lists.map((c) => c.body), [{ provider: 'gemini' }, { provider: 'gemini' }], 'exactly {provider}');
  assert.equal(key.value, TYPED[3], 'the typed key stays in its field');
  section.querySelectorAll('button').find((b) => b.textContent === 'Load models').click();
  await flush();
  assert.deepEqual(net.calls.filter((c) => c.path === '/api/llm/models').at(-1).body, { provider: 'gemini', api_key: TYPED[3] });
  key.value = '';
});

await test('"Model settings" scrolls the panel to the section, never the document', () => {
  setState({ settings: SETTINGS });
  // A browser's scrollIntoView can scroll every scrollable ancestor, the document included.
  const proto = Object.getPrototypeOf(view);
  const original = proto.scrollIntoView;
  let used = 0;
  proto.scrollIntoView = function scrollIntoView() {
    used += 1;
    document.documentElement.scrollTop = 56;
  };
  document.documentElement.scrollTop = 0;
  panelBody.scrollTop = 0;
  const section = view.querySelector('.model-section');
  section.getBoundingClientRect = () => ({ top: 640, bottom: 900, left: 0, right: 0, width: 0, height: 260 });
  panelBody.getBoundingClientRect = () => ({ top: 100, bottom: 800, left: 0, right: 0, width: 0, height: 700 });
  modeExplanation().querySelector('button').click();
  proto.scrollIntoView = original;
  assert.equal(used, 0, 'no scrollIntoView');
  assert.equal(document.documentElement.scrollTop, 0, 'the document did not move');
  assert.equal(panelBody.scrollTop, 540, 'the panel body scrolled to the section');
  assert.equal(document.activeElement, section.querySelector('.panel-h'));
});

await test('the DEMO badge and the note that typed keys are not used follow the settings', () => {
  const demoBadge = document.getElementById('demo-badge');
  const notes = () => view.querySelectorAll('.demo-note').filter((n) => !n.hidden);
  setState({ settings: SETTINGS }); // no `demo` field: an older server, or a real one
  assert.equal(demoBadge.hidden, true);
  assert.equal(notes().length, 0);
  setState({ settings: { ...SETTINGS, demo: true } });
  assert.equal(demoBadge.hidden, false);
  assert.equal(demoBadge.textContent, 'DEMO');
  assert.equal(demoBadge.getAttribute('title'), 'Demo server: fake broker and scripted model; nothing leaves this computer');
  assert.equal(demoBadge.getAttribute('aria-label'), demoBadge.getAttribute('title'));
  const shown = (root) => root.querySelectorAll('.demo-note').filter((n) => !n.hidden).length;
  assert.equal(shown(view.querySelector('.model-section')), 1, 'by the model key field');
  assert.deepEqual(rows().map((row) => shown(row)), [1, 1, 1], 'in each key row');
  assert.ok(notes().every((n) => n.textContent === dict.en.settings_demo_keys));
  setLang('ko');
  assert.equal(demoBadge.getAttribute('title'), dict.ko.demo_title);
  setLang('en');
  setState({ settings: { ...SETTINGS, demo: false } });
  assert.equal(demoBadge.hidden, true);
  assert.equal(notes().length, 0);
});

await test('no typed key ever reached a storage API or a cookie', () => {
  for (const [, , value] of writes) for (const key of TYPED) assert.ok(!value.includes(key), `${key} in ${value}`);
  assert.ok(!writes.some(([kind]) => kind === 'cookie'));
});

finish('settings panel checks');
