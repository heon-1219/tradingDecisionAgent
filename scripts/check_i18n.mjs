// Checks the web app's interface strings (web/app/js/i18n.js). Run: node scripts/check_i18n.mjs
//
// Fails (exit 1) if English and Korean do not have exactly the same keys, if any value is empty or
// not a string, if the {placeholders} of a key differ between the languages, or if a key used in
// web/app/index.html or web/app/js/**/*.js (data-i18n attributes, t('key'), bindText/bindAttr) is
// missing from the dictionary.
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
];
for (const path of sources) {
  const text = readFileSync(path, 'utf8');
  for (const re of patterns) for (const m of text.matchAll(re)) used.add(m[1]);
}
for (const key of used) if (!enKeys.has(key)) problems.push(`used but not defined: ${key}`);

if (problems.length) {
  console.error(`i18n check failed (${problems.length} problem${problems.length === 1 ? '' : 's'}):`);
  for (const p of problems) console.error(`  - ${p}`);
  process.exit(1);
}
console.log(`i18n OK: ${enKeys.size} keys in en and ko, ${used.size} referenced by the app, no empty values.`);
