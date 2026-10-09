/* The model provider settings without the DOM: the providers, where their keys are made, what a
 * save sends and where an error belongs. Pure functions, covered by scripts/test_model_settings.mjs.
 *
 * `GET /api/settings` reports `provider`, `model`, `effort`, `gemini_paid`, the research depth in
 * force (`mode`, `max_rounds`, `time_budget_sec`) and `providers: {gemini, openai, anthropic: "set" |
 * "missing"}`, never a key. `PUT /api/settings` takes `llm_provider`, `llm_model`, `llm_effort`,
 * `gemini_paid` and one key field per provider; the server checks a new provider, model or key
 * with one model call before it saves anything (see src/finagent/api_llm.py).
 */

export const PROVIDERS = ['gemini', 'openai', 'anthropic'];
// Product names, the same in every language.
export const PROVIDER_NAMES = { gemini: 'Gemini', openai: 'OpenAI', anthropic: 'Anthropic' };
// Where each provider's API keys are created. The only addresses of other sites in the app's code
// (tests/test_web_app.py checks that): shown as plain links that open in a new tab.
export const KEY_LINKS = {
  gemini: 'https://aistudio.google.com/apikey',
  openai: 'https://platform.openai.com/api-keys',
  anthropic: 'https://console.anthropic.com/settings/keys',
};
// The PUT /api/settings field that carries each provider's key.
export const KEY_FIELDS = { gemini: 'gemini_key', openai: 'openai_key', anthropic: 'anthropic_key' };
// `llm_effort` values the server accepts (config.py), lowest first.
export const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];
export const MODEL_MAX = 256;

const text = (value) => (typeof value === 'string' ? value : '');

export const isProvider = (value) => PROVIDERS.includes(value);

/* The models of a `POST /api/llm/models` answer the select can show: `{id, label}` with a string id
 * of at most MODEL_MAX characters, without duplicates. `label` falls back to the id. */
export function modelOptions(models) {
  const seen = new Set();
  const out = [];
  for (const model of Array.isArray(models) ? models : []) {
    const id = text(model?.id).trim();
    if (!id || id.length > MODEL_MAX || seen.has(id)) continue;
    seen.add(id);
    const label = text(model.label).trim();
    out.push({ id, label: label && label !== id ? `${label} (${id})` : id });
  }
  return out;
}

/* The PUT /api/settings body for what the form says now, with only the fields that change.
 * `form` is {provider, key, model, paid, effort}; `model` is the custom id when one was typed, else
 * the selected one. Returns {body} or {error: code} for a check made before anything is sent. A new
 * provider always sends its model: the server would otherwise check the old provider's model with
 * the new provider's key. */
export function saveBody(settings, form) {
  const s = settings || {};
  const body = {};
  const provider = isProvider(form.provider) ? form.provider : s.provider;
  const model = text(form.model).trim();
  const key = text(form.key).trim();
  if (provider !== s.provider) {
    if (!model) return { error: 'llm_model_required' };
    body.llm_provider = provider;
    body.llm_model = model;
  } else if (model && model !== s.model) {
    body.llm_model = model;
  }
  if (model.length > MODEL_MAX) return { error: 'llm_model_too_long' };
  if (key && KEY_FIELDS[provider]) body[KEY_FIELDS[provider]] = key;
  if (provider === 'gemini' && typeof form.paid === 'boolean' && form.paid !== Boolean(s.gemini_paid)) {
    body.gemini_paid = form.paid;
  }
  if (EFFORTS.includes(form.effort) && form.effort !== s.effort) body.llm_effort = form.effort;
  return { body };
}

/* Whether the server will make a model call to check this save (a new provider, model or key). */
export function checksTheKey(body) {
  return Object.keys(body || {}).some((name) => name === 'llm_provider' || name === 'llm_model'
    || Object.values(KEY_FIELDS).includes(name));
}

// Error code from PUT /api/settings or POST /api/llm/models -> i18n key.
export const LLM_ERRORS = {
  llm_invalid_key: 'settings_llm_err_invalid_key',
  llm_key_required: 'settings_llm_err_key_required',
  llm_model_unavailable: 'settings_llm_err_model_unavailable',
  llm_model_required: 'settings_llm_err_model_required',
  llm_model_too_long: 'settings_llm_err_model_too_long',
  llm_unreachable: 'settings_llm_err_unreachable',
  llm_rate_limited: 'settings_llm_err_rate_limited',
  llm_list_failed: 'settings_llm_err_list_failed',
  invalid_value: 'settings_err_invalid_value',
};

/* Which field an error belongs next to: 'key', 'model' or 'save' (under the Save button). */
export function errorSpot(code, field) {
  if (code === 'llm_invalid_key' || code === 'llm_key_required') return 'key';
  if (code === 'llm_model_unavailable' || code === 'llm_model_required' || code === 'llm_model_too_long') return 'model';
  if (code === 'invalid_value') {
    if (field === 'llm_model') return 'model';
    if (field === 'api_key' || Object.values(KEY_FIELDS).includes(field)) return 'key';
  }
  return 'save';
}

/* The research depth in force, from a settings body or a `finish` event: {mode, rounds, seconds},
 * or null when it does not say. */
export function depthOf(source) {
  const mode = source && (source.mode === 'shallow' || source.mode === 'deep') ? source.mode : null;
  if (!mode) return null;
  const rounds = Number.isInteger(source.max_rounds) && source.max_rounds > 0 ? source.max_rounds : null;
  const seconds = Number.isFinite(source.time_budget_sec) && source.time_budget_sec > 0 ? source.time_budget_sec : null;
  return { mode, rounds, seconds };
}
