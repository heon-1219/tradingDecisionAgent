// Unit tests for the pure parts of the web app's markdown pipeline (web/app/js/markdown.js).
// Run: node scripts/test_markdown.mjs
//
// 1. `safeHttpUrl`: only absolute http(s) URLs without credentials survive.
// 2. `writeMarkdownChunk`: "$" stays a literal dollar sign outside code. streaming-markdown reads
//    "$…$" as inline math, which would turn "Buy $VOO and $VTI" into an equation and drop the
//    dollar signs; finance answers are full of them. A recording renderer stands in for the DOM.
// The DOM half (the sanitize walk) needs a browser and is checked there.
import assert from 'node:assert/strict';
import * as smd from '../web/app/vendor/streaming-markdown.min.js';
import { safeHttpUrl, writeMarkdownChunk } from '../web/app/js/markdown.js';

let checks = 0;
function test(name, fn) {
  try {
    fn();
    checks += 1;
    console.log(`ok   ${name}`);
  } catch (err) {
    console.error(`FAIL ${name}`);
    throw err;
  }
}

// --- safeHttpUrl -------------------------------------------------------------------------------

test('safeHttpUrl keeps absolute http(s) URLs', () => {
  assert.equal(safeHttpUrl('https://www.sec.gov/a?b=c#d'), 'https://www.sec.gov/a?b=c#d');
  assert.equal(safeHttpUrl('  http://example.com  '), 'http://example.com/');
  assert.equal(safeHttpUrl('HTTPS://Example.com/Path'), 'https://example.com/Path');
});

test('safeHttpUrl refuses everything else', () => {
  const refused = [
    'javascript:alert(1)', 'JaVaScRiPt:alert(1)', '\tjavascript:alert(1)', 'java\nscript:alert(1)',
    'data:text/html,<script>alert(1)</script>', 'vbscript:msgbox(1)', 'file:///etc/passwd',
    'blob:https://example.com/x', 'ftp://example.com/', 'mailto:a@example.com', 'tel:123',
    '/api/kill-switch', '//evil.test/x', 'evil.test/x', '#frag', '', '   ',
    'https://user:pass@example.com/', 'https://user@example.com/', `https://example.com/${'a'.repeat(2100)}`,
    null, undefined, 42, {},
  ];
  for (const value of refused) assert.equal(safeHttpUrl(value), null, String(value));
});

// --- "$" stays literal -------------------------------------------------------------------------

const NAMES = {
  [smd.PARAGRAPH]: 'p', [smd.CODE_INLINE]: 'code', [smd.CODE_FENCE]: 'fence', [smd.CODE_BLOCK]: 'pre',
  [smd.LINK]: 'a', [smd.RAW_URL]: 'url', [smd.IMAGE]: 'img', [smd.TABLE]: 'table', [smd.TABLE_ROW]: 'tr',
  [smd.TABLE_CELL]: 'td', [smd.EQUATION_INLINE]: 'EQ', [smd.EQUATION_BLOCK]: 'EQBLOCK',
  [smd.STRONG_AST]: 'strong', [smd.ITALIC_UND]: 'em', [smd.LIST_UNORDERED]: 'ul', [smd.LIST_ORDERED]: 'ol',
  [smd.LIST_ITEM]: 'li', [smd.HEADING_2]: 'h2',
};

/* Render `pieces` and return {text, tokens, attrs}: all text in order, the token names opened. */
function render(pieces) {
  const out = { text: '', tokens: [], attrs: [] };
  const renderer = {
    data: {},
    add_token(_, type) { out.tokens.push(NAMES[type] || String(type)); },
    end_token() {},
    add_text(_, text) { out.text += text; },
    set_attr(_, type, value) { out.attrs.push(value); },
  };
  const parser = smd.parser(renderer);
  const state = { backslashes: 0 };
  for (const piece of pieces) writeMarkdownChunk(parser, piece, state);
  smd.parser_end(parser);
  return out;
}

const noMath = (out) => !out.tokens.includes('EQ') && !out.tokens.includes('EQBLOCK');

test('the unescaped parser really does eat cashtags (the reason for the escaping)', () => {
  const out = { text: '', tokens: [] };
  const parser = smd.parser({
    data: {},
    add_token(_, type) { out.tokens.push(NAMES[type] || String(type)); },
    end_token() {},
    add_text(_, text) { out.text += text; },
    set_attr() {},
  });
  smd.parser_write(parser, 'Buy $VOO and $VTI now.');
  smd.parser_end(parser);
  assert.ok(out.tokens.includes('EQ'));
  assert.ok(!out.text.includes('$VOO'));
});

test('cashtags and amounts keep their dollar signs', () => {
  for (const source of [
    'Buy $VOO today.\n\nNext paragraph with $5,000 cash.',
    'Price is $512.30 and $VTI.',
    '$VOO at the start of a line',
    '**$VOO** and _$VTI_',
    'Totals: $$ and $x$ are not math here',
  ]) {
    const out = render([source]);
    assert.ok(noMath(out), source);
    // Same characters once markup and whitespace are set aside (paragraph breaks are elements).
    const plain = source.replace(/\*\*|_/g, '');
    assert.equal(out.text.replace(/\s+/g, ''), plain.replace(/\s+/g, ''), source);
  }
});

test('headings, lists and tables keep their dollar signs', () => {
  const out = render(['## Plan for $5,000\n\n- $VOO 60%\n- $VTI 40%\n\n| sym | px |\n|---|---|\n| $VOO | $512 |\n']);
  assert.ok(noMath(out));
  for (const piece of ['Plan for $5,000', '$VOO 60%', '$VTI 40%', '$512']) assert.ok(out.text.includes(piece), piece);
  assert.ok(out.tokens.includes('h2') && out.tokens.includes('ul') && out.tokens.includes('table'));
});

test('code keeps "$" exactly, without a backslash', () => {
  const inline = render(['Run `echo $HOME` and `$x` now']);
  assert.ok(noMath(inline));
  assert.ok(inline.text.includes('echo $HOME') && !inline.text.includes('\\'));
  const fence = render(['```bash\necho $HOME\ncost=$5\n```\n']);
  assert.ok(noMath(fence));
  assert.ok(fence.text.includes('echo $HOME') && fence.text.includes('cost=$5') && !fence.text.includes('\\'));
  assert.deepEqual(fence.attrs, ['bash']);
});

test('a "$" inside a link URL is not escaped', () => {
  const out = render(['[$VOO page](https://x.test/a$b) ok']);
  assert.ok(noMath(out));
  assert.deepEqual(out.attrs, ['https://x.test/a$b']);
  assert.ok(out.text.includes('$VOO page'));
});

test('an already escaped "\\$" is left alone, also across a chunk boundary', () => {
  for (const pieces of [['Costs \\$VOO here'], ['Costs \\', '$VOO here'], ['Costs ', '\\', '$', 'VOO here']]) {
    const out = render(pieces);
    assert.ok(noMath(out), JSON.stringify(pieces));
    assert.equal(out.text, 'Costs $VOO here');
  }
});

test('every split point of a mixed message gives the same text', () => {
  const source = 'Buy $VOO at $512.30.\n\n`$x` costs \\$5 and **$VTI** too.\n\n- $1,000 in $BND\n';
  const whole = render([source]);
  assert.ok(noMath(whole));
  for (let cut = 1; cut < source.length; cut += 1) {
    const out = render([source.slice(0, cut), source.slice(cut)]);
    assert.ok(noMath(out), `cut ${cut}`);
    assert.equal(out.text, whole.text, `cut ${cut}`);
  }
  const chars = render([...source]);
  assert.equal(chars.text, whole.text);
});

console.log(`\n${checks} markdown checks passed`);
