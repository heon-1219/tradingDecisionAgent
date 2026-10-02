/* The top bar: session title, model chip, the last turn's tokens and latency, language and theme.
 * It renders from `state` (`sessionTitle`, `settings`, `lastFinish`) and re-renders on a language
 * change, so nothing here is called with text directly. */
import { bindAttr, bindText, formatNumber, getLang, setLang, t } from './i18n.js';
import { on, state, storage, STORAGE_KEYS } from './state.js';

const count = (value) => (Number.isFinite(value) && value > 0 ? Math.round(value) : 0);

export function initTopbar() {
  const root = document.documentElement;
  const $ = (id) => document.getElementById(id);
  const title = $('session-title');
  const chip = $('model-chip');
  const stats = $('turn-stats');
  const themeButton = $('theme-toggle');
  const langButtons = { en: $('lang-en'), ko: $('lang-ko') };
  const lightScheme = window.matchMedia('(prefers-color-scheme: light)');

  function renderTitle() {
    if (state.sessionId && state.sessionTitle) {
      title.removeAttribute('data-i18n');
      title.textContent = state.sessionTitle;
      title.title = state.sessionTitle;
      document.title = `${state.sessionTitle} · ${t('app_name')}`;
    } else {
      bindText(title, 'new_chat');
      title.removeAttribute('title');
      document.title = t('doc_title');
    }
  }

  function renderModel() {
    const settings = state.settings;
    const model = settings && typeof settings.model === 'string' ? settings.model : '';
    chip.hidden = !model;
    if (!model) return;
    chip.textContent = model;
    chip.title = t('model_title', { model, effort: settings.effort || '–' });
  }

  function renderStats() {
    const finish = state.lastFinish;
    if (!finish) {
      stats.hidden = true;
      return;
    }
    const usage = finish.usage && typeof finish.usage === 'object' ? finish.usage : {};
    const input = count(usage.input);
    const output = count(usage.output);
    const seconds = formatNumber(count(finish.elapsed_ms) / 1000, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    const parts = [];
    if (input + output > 0) parts.push(t('stats_tokens', { n: formatNumber(input + output) }));
    parts.push(t('stats_seconds', { s: seconds }));
    stats.textContent = parts.join(' · ');
    stats.title = t('stats_title', { input: formatNumber(input), output: formatNumber(output), s: seconds });
    stats.hidden = false;
  }

  // Theme: the system setting until the user chooses; the choice is stored.
  function effectiveTheme() {
    const chosen = root.getAttribute('data-theme');
    if (chosen === 'light' || chosen === 'dark') return chosen;
    return lightScheme.matches ? 'light' : 'dark';
  }

  function syncTheme() {
    const dark = effectiveTheme() === 'dark';
    bindAttr(themeButton, 'aria-label', dark ? 'theme_to_light' : 'theme_to_dark');
    themeButton.title = themeButton.getAttribute('aria-label');
    $('theme-icon-sun').hidden = !dark;
    $('theme-icon-moon').hidden = dark;
  }

  themeButton.addEventListener('click', () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    storage.set(STORAGE_KEYS.theme, next);
    syncTheme();
  });
  lightScheme.addEventListener('change', syncTheme);

  function syncLang() {
    const lang = getLang();
    for (const [code, button] of Object.entries(langButtons)) {
      button.setAttribute('aria-pressed', String(code === lang));
    }
  }

  for (const [code, button] of Object.entries(langButtons)) {
    button.addEventListener('click', () => setLang(code));
  }

  on('lang', () => {
    syncLang();
    syncTheme();
    renderTitle();
    renderModel();
    renderStats();
  });
  on('change', (patch) => {
    if ('sessionTitle' in patch || 'sessionId' in patch) renderTitle();
    if ('settings' in patch) renderModel();
    if ('lastFinish' in patch) renderStats();
  });

  syncLang();
  syncTheme();
  renderTitle();
  renderModel();
  renderStats();
}
