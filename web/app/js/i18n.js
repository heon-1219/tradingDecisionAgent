/* Interface strings in English (the default and the source of truth) and Korean.
 *
 * Markup opts in with attributes, and `applyI18n` fills them on every language change:
 *   data-i18n="<key>"             -> textContent
 *   data-i18n-aria-label="<key>"  -> aria-label
 *   data-i18n-title="<key>"       -> title
 *   data-i18n-placeholder="<key>" -> placeholder
 * {placeholders} come from a JSON attribute per binding, so text and attributes on one element
 * never share variables: data-i18n-vars for the text, data-i18n-aria-label-vars,
 * data-i18n-title-vars and data-i18n-placeholder-vars for the attributes.
 * Strings are plain text, never markup. `scripts/check_i18n.mjs` checks that both languages have
 * the same keys and no empty value.
 *
 * The shell's strings are below; the conversation components' strings live in
 * `strings-components-en.js` and `strings-components-ko.js` and are merged into the same `dict`.
 */
import { storage, STORAGE_KEYS, emit } from './state.js';
import componentsEn from './strings-components-en.js';
import componentsKo from './strings-components-ko.js';

export const LANGS = ['en', 'ko'];
export const DEFAULT_LANG = 'en';

export const dict = {
  en: {
    app_name: 'FinAgent',
    doc_title: 'FinAgent · Paper trading',
    skip_to_composer: 'Skip to the message box',

    rail_label: 'Chats',
    rail_expand: 'Expand sidebar',
    rail_collapse: 'Collapse sidebar',
    new_chat: 'New chat',
    recent: 'Recent',
    sessions_empty: 'Your chats will appear here.',
    sessions_loading: 'Loading chats…',
    rename_chat: 'Rename “{title}”',
    rename_input: 'Chat title',
    drawer_close: 'Close chats',

    menu_open: 'Open chats',
    paper: 'PAPER',
    paper_title: 'Paper trading: orders go to a simulated Alpaca paper account. No real money moves.',
    model_title: 'Model: {model} · effort: {effort}',
    stats_tokens: '{n} tokens',
    stats_seconds: '{s} s',
    stats_title: 'Last response: {input} input and {output} output tokens in {s} s',
    lang_group: 'Interface language',
    lang_en_label: 'Show the interface in English',
    lang_ko_label: 'Show the interface in Korean',
    theme_to_light: 'Switch to the light theme',
    theme_to_dark: 'Switch to the dark theme',
    panel_show: 'Show the side panel',
    panel_hide: 'Hide the side panel',

    panel_label: 'Account and settings',
    tab_portfolio: 'Portfolio',
    tab_orders: 'Orders',
    tab_mandate: 'Mandate',
    tab_settings: 'Settings',
    panel_close: 'Close the panel',
    settings_heading: 'Connections',
    settings_alpaca: 'Alpaca paper account',
    settings_tavily: 'Tavily web search',
    settings_sec: 'SEC EDGAR contact',
    settings_gemini: 'Gemini API key',
    settings_trading_heading: 'Agent',
    settings_trading: 'Trading',
    settings_model: 'Model',
    settings_effort: 'Reasoning effort',
    settings_note: 'Status only. Keys live in your local .env file and are never shown here.',
    settings_refresh: 'Refresh',
    settings_error: 'Couldn’t load settings.',
    settings_loading: 'Loading…',
    status_connected: 'Connected',
    status_missing: 'Not set',
    status_invalid: 'Invalid key',
    status_unreachable: 'Unreachable',
    status_set: 'Set',
    status_enabled: 'Enabled',
    status_disabled: 'Disabled',
    status_unknown: 'Unknown',

    empty_greeting: 'What should we research today?',
    empty_lede: 'Paper trading only. I research with cited sources, explain a strategy and propose trades. Nothing is placed until you confirm it.',
    suggestions_label: 'Suggestions',
    suggest_1: 'Build a low-cost US ETF core portfolio with $5,000',
    suggest_2: 'Compare VOO and VTI as a long-term core holding',
    suggest_3: 'Add a $1,500 dividend sleeve using dividend ETFs',
    suggest_4: 'Review my current positions and suggest a rebalance',

    composer_label: 'Message',
    composer_placeholder: 'Ask about a stock, an ETF or a strategy…',
    composer_hint: 'Enter to send · Shift+Enter for a new line',
    send: 'Send',
    stop: 'Stop',
    disclaimer: 'Paper trading only. Not investment advice.',

    thread_label: 'Conversation',
    you: 'You',
    assistant: 'FinAgent',
    pending: 'Working…',
    thinking_live: 'Thinking…',
    thinking_done: 'Thoughts',
    tool_running: 'Running {name}…',
    tool_done: '✓ {name}',
    tool_error: '✕ {name} failed',
    tool_stopped: '{name} · stopped',
    tool_unnamed: 'tool',
    stopped_note: 'Stopped',
    md_image: '🖼 image: {alt}',
    md_image_bare: '🖼 image',
    md_unsafe_note: 'Formatting is off for the rest of this message because it contained HTML.',
    copy: 'Copy',
    copied: 'Copied',
    copy_failed: 'Couldn’t copy',
    copy_code: 'Copy code',
    copy_table: 'Copy table',
    code_label: 'Code',
    table_label: 'Table',
    loading_session: 'Loading chat…',

    err_network: 'Can’t reach the FinAgent server. Check that it’s running, then try again.',
    err_http: 'The server returned an error (HTTP {status}).',
    err_generic: 'Something went wrong. Please try again.',
    err_stream_cut: 'The connection closed before the answer finished. Send your message again to retry.',
    err_not_stream: 'The server sent an unexpected response.',
    err_too_long: 'That message is too long (at most 16,000 characters).',
    err_sessions: 'Couldn’t load your chats.',
    announce_done: 'Response complete',
    announce_stopped: 'Response stopped',
    announce_error: 'Response ended with an error',
    dismiss: 'Dismiss',
    ...componentsEn,
  },

  ko: {
    app_name: 'FinAgent',
    doc_title: 'FinAgent · 모의투자',
    skip_to_composer: '메시지 입력란으로 건너뛰기',

    rail_label: '대화 목록',
    rail_expand: '사이드바 펼치기',
    rail_collapse: '사이드바 접기',
    new_chat: '새 대화',
    recent: '최근 대화',
    sessions_empty: '대화를 시작하면 여기에 표시됩니다.',
    sessions_loading: '대화 목록을 불러오는 중…',
    rename_chat: '“{title}” 이름 바꾸기',
    rename_input: '대화 제목',
    drawer_close: '대화 목록 닫기',

    menu_open: '대화 목록 열기',
    paper: '모의투자',
    paper_title: '모의투자: 주문은 Alpaca 모의 계좌로만 전송되며 실제 돈은 움직이지 않습니다.',
    model_title: '모델: {model} · 추론 강도: {effort}',
    stats_tokens: '토큰 {n}개',
    stats_seconds: '{s}초',
    stats_title: '직전 응답: 입력 토큰 {input}개, 출력 토큰 {output}개, {s}초',
    lang_group: '화면 언어',
    lang_en_label: '화면을 영어로 표시',
    lang_ko_label: '화면을 한국어로 표시',
    theme_to_light: '밝은 테마로 전환',
    theme_to_dark: '어두운 테마로 전환',
    panel_show: '사이드 패널 열기',
    panel_hide: '사이드 패널 닫기',

    panel_label: '계좌 및 설정',
    tab_portfolio: '포트폴리오',
    tab_orders: '주문',
    tab_mandate: '투자 원칙',
    tab_settings: '설정',
    panel_close: '패널 닫기',
    settings_heading: '연결 상태',
    settings_alpaca: 'Alpaca 모의 계좌',
    settings_tavily: 'Tavily 웹 검색',
    settings_sec: 'SEC EDGAR 연락처',
    settings_gemini: 'Gemini API 키',
    settings_trading_heading: '에이전트',
    settings_trading: '거래 기능',
    settings_model: '모델',
    settings_effort: '추론 강도',
    settings_note: '상태만 표시합니다. 키는 로컬 .env 파일에 저장되며 여기에는 표시되지 않습니다.',
    settings_refresh: '새로 고침',
    settings_error: '설정을 불러오지 못했습니다.',
    settings_loading: '불러오는 중…',
    status_connected: '연결됨',
    status_missing: '미설정',
    status_invalid: '잘못된 키',
    status_unreachable: '연결 불가',
    status_set: '설정됨',
    status_enabled: '켜짐',
    status_disabled: '꺼짐',
    status_unknown: '알 수 없음',

    empty_greeting: '오늘은 무엇을 리서치해 볼까요?',
    empty_lede: '모의투자 전용입니다. 출처를 밝혀 리서치하고 전략을 설명한 뒤 주문을 제안합니다. 승인하기 전에는 어떤 주문도 넣지 않습니다.',
    suggestions_label: '추천 질문',
    suggest_1: '5,000달러로 저비용 미국 ETF 핵심 포트폴리오를 구성해 줘',
    suggest_2: '장기 핵심 보유 종목으로 VOO와 VTI를 비교해 줘',
    suggest_3: '배당 ETF로 1,500달러 규모의 배당 포트폴리오를 추가해 줘',
    suggest_4: '지금 보유 종목을 점검하고 리밸런싱 방안을 제안해 줘',

    composer_label: '메시지',
    composer_placeholder: '종목, ETF, 투자 전략에 대해 물어보세요…',
    composer_hint: 'Enter로 보내기 · Shift+Enter로 줄바꿈',
    send: '보내기',
    stop: '중지',
    disclaimer: '모의투자 전용이며 투자 권유가 아닙니다.',

    thread_label: '대화',
    you: '나',
    assistant: 'FinAgent',
    pending: '작업 중…',
    thinking_live: '생각하는 중…',
    thinking_done: '생각 과정',
    tool_running: '{name} 실행 중…',
    tool_done: '✓ {name}',
    tool_error: '✕ {name} 실패',
    tool_stopped: '{name} · 중지됨',
    tool_unnamed: '도구',
    stopped_note: '중지됨',
    md_image: '🖼 이미지: {alt}',
    md_image_bare: '🖼 이미지',
    md_unsafe_note: '이 메시지에 HTML이 포함되어 있어 나머지 부분은 서식 없이 표시합니다.',
    copy: '복사',
    copied: '복사됨',
    copy_failed: '복사하지 못했습니다',
    copy_code: '코드 복사',
    copy_table: '표 복사',
    code_label: '코드',
    table_label: '표',
    loading_session: '대화를 불러오는 중…',

    err_network: 'FinAgent 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인한 뒤 다시 시도해 주세요.',
    err_http: '서버에서 오류가 발생했습니다(HTTP {status}).',
    err_generic: '문제가 발생했습니다. 다시 시도해 주세요.',
    err_stream_cut: '응답이 끝나기 전에 연결이 끊겼습니다. 메시지를 다시 보내 주세요.',
    err_not_stream: '서버에서 예상하지 못한 응답을 받았습니다.',
    err_too_long: '메시지가 너무 깁니다(최대 16,000자).',
    err_sessions: '대화 목록을 불러오지 못했습니다.',
    announce_done: '응답 완료',
    announce_stopped: '응답 중지됨',
    announce_error: '오류로 응답이 끝났습니다',
    dismiss: '닫기',
    ...componentsKo,
  },
};

let current = null;

export function getLang() {
  if (current) return current;
  const stored = storage.get(STORAGE_KEYS.lang);
  current = LANGS.includes(stored) ? stored : DEFAULT_LANG;
  return current;
}

/* The string for `key` in the current language, English as the fallback, the key itself last.
 * `{name}` placeholders are filled from `vars`; unknown placeholders are left as they are. */
export function t(key, vars) {
  const table = dict[getLang()] || dict[DEFAULT_LANG];
  let text = table[key] ?? dict[DEFAULT_LANG][key] ?? key;
  if (vars) {
    text = text.replace(/\{(\w+)\}/g, (match, name) =>
      Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match);
  }
  return text;
}

export function setLang(lang) {
  const next = LANGS.includes(lang) ? lang : DEFAULT_LANG;
  current = next;
  storage.set(STORAGE_KEYS.lang, next);
  if (typeof document !== 'undefined') applyI18n(document);
  emit('lang', next);
  return next;
}

function varsOf(el, attribute) {
  const raw = el.getAttribute(attribute);
  if (!raw) return undefined;
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
}

const ATTRS = [
  ['data-i18n-aria-label', 'aria-label'],
  ['data-i18n-title', 'title'],
  ['data-i18n-placeholder', 'placeholder'],
];

/* Fill every tagged element under `root` (and `root` itself) in the current language, and set
 * `<html lang>` and the document title when `root` is the document. */
export function applyI18n(root) {
  if (!root) return;
  const lang = getLang();
  const isDoc = typeof document !== 'undefined' && root === document;
  if (isDoc) {
    document.documentElement.lang = lang;
  }
  const scope = isDoc ? document.documentElement : root;
  const tagged = [scope, ...scope.querySelectorAll('[data-i18n],[data-i18n-aria-label],[data-i18n-title],[data-i18n-placeholder]')];
  for (const el of tagged) {
    if (!el.getAttribute) continue;
    const key = el.getAttribute('data-i18n');
    if (key) el.textContent = t(key, varsOf(el, 'data-i18n-vars'));
    for (const [attr, target] of ATTRS) {
      const k = el.getAttribute(attr);
      if (k) el.setAttribute(target, t(k, varsOf(el, `${attr}-vars`)));
    }
  }
}

/* Tag `el` with `key` (and optional vars) and fill it now, so later language switches update it. */
export function bindText(el, key, vars) {
  el.setAttribute('data-i18n', key);
  if (vars) el.setAttribute('data-i18n-vars', JSON.stringify(vars));
  else el.removeAttribute('data-i18n-vars');
  el.textContent = t(key, vars);
  return el;
}

/* Same for an attribute: `bindAttr(btn, 'aria-label', 'send')`. Each attribute keeps its own
 * variables (`data-i18n-aria-label-vars`), apart from the text's. */
export function bindAttr(el, attr, key, vars) {
  el.setAttribute(`data-i18n-${attr}`, key);
  if (vars) el.setAttribute(`data-i18n-${attr}-vars`, JSON.stringify(vars));
  else el.removeAttribute(`data-i18n-${attr}-vars`);
  el.setAttribute(attr, t(key, vars));
  return el;
}

/* A number in the interface language's format, e.g. 1,234. */
export function formatNumber(value, options) {
  try {
    return new Intl.NumberFormat(getLang() === 'ko' ? 'ko-KR' : 'en-US', options).format(value);
  } catch {
    return String(value);
  }
}
