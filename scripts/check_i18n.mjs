// Checks the web app's interface strings (web/app/js/i18n.js). Run: node scripts/check_i18n.mjs
//
// Fails (exit 1) if English and Korean do not have exactly the same keys, if any value is empty or
// not a string, if the {placeholders} of a key differ between the languages, or if a key used in
// web/app/index.html or web/app/js/**/*.js (data-i18n attributes, t('key'), bindText/bindAttr) is
// missing from the dictionary. For the conversation components (web/app/js/components) it also
// checks the keys passed to their own helpers (label, badge, actionButton, iconButton) and every
// quoted word that looks like a key (it starts with one of KEY_PREFIXES), which covers the keys kept
// in lookup tables. A key may not be defined twice: the shell's strings and the components' strings
// are merged into one dictionary.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dict } from '../web/app/js/i18n.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'web', 'app');
const problems = [];

const langs = Object.keys(dict);
if (langs.join(',') !== 'en,ko') problems.push(`expected languages en,ko; found ${langs.join(',')}`);
const en = dict.en || {};
const ko = dict.ko || {};
const enKeys = new Set(Object.keys(en));
const koKeys = new Set(Object.keys(ko));

for (const key of enKeys) if (!koKeys.has(key)) problems.push(`missing in ko: ${key}`);
for (const key of koKeys) if (!enKeys.has(key)) problems.push(`missing in en: ${key}`);

const placeholders = (s) => [...String(s).matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(',');
for (const [lang, table] of Object.entries(dict)) {
  for (const [key, value] of Object.entries(table)) {
    if (typeof value !== 'string' || value.trim() === '') problems.push(`empty value: ${lang}.${key}`);
  }
}
for (const key of enKeys) {
  if (koKeys.has(key) && placeholders(en[key]) !== placeholders(ko[key])) {
    problems.push(`placeholders differ for ${key}: en {${placeholders(en[key])}} vs ko {${placeholders(ko[key])}}`);
  }
}

// Keys referenced by the markup and the scripts must exist.
function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}
const sources = [join(root, 'index.html'), ...files(join(root, 'js')).filter((p) => p.endsWith('.js'))];
const used = new Set();
const patterns = [
  /data-i18n(?:-aria-label|-title|-placeholder)?="([a-z0-9_]+)"/g,
  /\bt\(\s*'([a-z0-9_]+)'/g,
  /\bbindText\([^,]+,\s*'([a-z0-9_]+)'/g,
  /\bbindAttr\([^,]+,\s*'[a-z-]+',\s*'([a-z0-9_]+)'/g,
  // the components' helpers (components/dom.js)
  /\blabel\('[a-z0-9]+',\s*'[^']*',\s*'([a-z0-9_]+)'/g,
  /\bbadge\('[a-z]+',\s*'([a-z0-9_]+)'/g,
  /\bactionButton\('[^']*',\s*'([a-z0-9_]+)'/g,
  /\biconButton\('[a-z-]+',\s*'([a-z0-9_]+)'/g,
];
// In the components, keys also sit in lookup tables ({accept: ['verdict_accept', 'ok']}). Any quoted
// word starting with one of these prefixes is taken to be a key, except the names in NOT_KEYS.
const KEY_PREFIXES = [
  'family_', 'tool_', 'thinking_', 'source_', 'sources_', 'tier_', 'suggestions_', 'flags_', 'mandate_',
  'order_', 'orders_', 'portfolio_', 'positions_', 'stat_', 'col_', 'paper_', 'account_', 'ticket_',
  'timeline_', 'verdict_', 'severity_', 'clamp_', 'side_', 'size_', 'terms_', 'tif_', 'leg_', 'age_',
  'settings_', 'kill_', 'status_', 'open_', 'popover_', 'err_', 'announce_', 'mode_',
];
const NOT_KEYS = new Set(['order_types', 'kill_switch_pressed']); // a Mandate field name, an API error code
const componentsDir = join(root, 'js', 'components');
for (const path of sources) {
  const text = readFileSync(path, 'utf8');
  for (const re of patterns) for (const m of text.matchAll(re)) used.add(m[1]);
  if (!path.startsWith(componentsDir)) continue;
  for (const m of text.matchAll(/'([a-z]+_[a-z0-9_]+)'/g)) {
    if (!NOT_KEYS.has(m[1]) && KEY_PREFIXES.some((prefix) => m[1].startsWith(prefix))) used.add(m[1]);
  }
}
for (const key of used) if (!enKeys.has(key)) problems.push(`used but not defined: ${key}`);

// The component strings are spread into the shell's dictionary: a key defined in both, or twice in
// one file, would be silently overridden, so it is an error.
const defined = (path) => [...readFileSync(path, 'utf8').matchAll(/^ {2,4}([a-z0-9_]+): /gm)].map((m) => m[1]);
const componentKeys = new Set();
for (const lang of ['en', 'ko']) {
  const names = defined(join(root, 'js', `strings-components-${lang}.js`));
  for (const name of new Set(names.filter((n, index) => names.indexOf(n) !== index))) {
    problems.push(`defined twice in strings-components-${lang}.js: ${name}`);
  }
  for (const name of names) componentKeys.add(name);
}
const shellKeys = defined(join(root, 'js', 'i18n.js'));
for (const name of new Set(shellKeys.filter((n, index) => shellKeys.indexOf(n) !== index))) {
  // each shell key appears once per language
  if (shellKeys.filter((n) => n === name).length > 2) problems.push(`defined twice in i18n.js: ${name}`);
}
for (const name of new Set(shellKeys)) {
  if (componentKeys.has(name)) problems.push(`defined by the shell and by the components: ${name}`);
}

if (problems.length) {
  console.error(`i18n check failed (${problems.length} problem${problems.length === 1 ? '' : 's'}):`);
  for (const p of problems) console.error(`  - ${p}`);
  process.exit(1);
}
console.log(`i18n OK: ${enKeys.size} keys in en and ko, ${used.size} referenced by the app, no empty values.`);
