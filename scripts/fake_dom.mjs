// A small fake DOM for Node tests of the web app (no dependencies).
//
// It implements only what streaming-markdown's DOM renderer and web/app/js/{markdown,i18n,icons}.js
// use: createElement / createElementNS / createTextNode, appendChild / append / replaceWith /
// remove, attributes, className / classList, textContent, children / childNodes, contains,
// normalize, and querySelector(All) for simple selectors ("*", "a", ".cls", "[attr]", "a[href]",
// comma lists). `installFakeDom()` puts a fresh document on globalThis and returns it;
// `document.created` lists every tag ever created and `document.attached` every tag ever inserted,
// so a test can prove an element (e.g. <img>) never existed.
//
// For the component tests (scripts/test_components.mjs) it also has: isConnected, insertBefore,
// after, lastChild, matches / closest (same simple selectors), getElementById, and events:
// addEventListener stores the listener, and `click()` / `dispatch(type, init)` call the listeners of
// the element and then of its ancestors (bubbling), unless the element is disabled.

class FakeNode {
  constructor(ownerDocument) {
    this.ownerDocument = ownerDocument;
    this.parentNode = null;
  }

  get parentElement() {
    return this.parentNode && this.parentNode.nodeType === 1 ? this.parentNode : null;
  }

  remove() {
    const parent = this.parentNode;
    if (!parent) return;
    parent.childNodes.splice(parent.childNodes.indexOf(this), 1);
    this.parentNode = null;
  }

  replaceWith(...nodes) {
    const parent = this.parentNode;
    if (!parent) return;
    const fresh = nodes.map((n) => parent._adopt(n));
    parent.childNodes.splice(parent.childNodes.indexOf(this), 1, ...fresh);
    this.parentNode = null;
  }

  after(...nodes) {
    const parent = this.parentNode;
    if (!parent) return;
    const fresh = nodes.map((n) => parent._adopt(n));
    parent.childNodes.splice(parent.childNodes.indexOf(this) + 1, 0, ...fresh);
  }

  get isConnected() {
    for (let n = this; n; n = n.parentNode) if (n === this.ownerDocument.documentElement) return true;
    return false;
  }
}

class FakeText extends FakeNode {
  constructor(ownerDocument, data) {
    super(ownerDocument);
    this.nodeType = 3;
    this.nodeName = '#text';
    this.data = String(data);
  }

  get textContent() { return this.data; }
  set textContent(value) { this.data = String(value); }
  get length() { return this.data.length; }
  appendData(text) { this.data += String(text); }
}

const SELECTOR = /^([a-zA-Z0-9*-]+)?((?:\.[\w-]+)*)((?:\[[\w:-]+\])*)$/;

function compile(selector) {
  return selector.split(',').map((part) => {
    const m = SELECTOR.exec(part.trim());
    if (!m) throw new Error(`fake DOM: unsupported selector "${part}"`);
    return {
      tag: m[1] && m[1] !== '*' ? m[1].toLowerCase() : null,
      classes: m[2] ? m[2].split('.').filter(Boolean) : [],
      attrs: m[3] ? m[3].slice(1, -1).split('][') : [],
    };
  });
}

class FakeElement extends FakeNode {
  constructor(ownerDocument, tag, namespaceURI) {
    super(ownerDocument);
    this.nodeType = 1;
    this.namespaceURI = namespaceURI || 'http://www.w3.org/1999/xhtml';
    this._html = !namespaceURI;
    this.localName = this._html ? String(tag).toLowerCase() : String(tag);
    this.childNodes = [];
    this._attrs = new Map();
  }

  get tagName() { return this._html ? this.localName.toUpperCase() : this.localName; }
  get nodeName() { return this.tagName; }

  // --- attributes ---
  _name(name) { return this._html ? String(name).toLowerCase() : String(name); }
  setAttribute(name, value) { this._attrs.set(this._name(name), String(value)); }
  getAttribute(name) { return this._attrs.has(this._name(name)) ? this._attrs.get(this._name(name)) : null; }
  hasAttribute(name) { return this._attrs.has(this._name(name)); }
  removeAttribute(name) { this._attrs.delete(this._name(name)); }
  get attributes() { return [...this._attrs].map(([name, value]) => ({ name, value })); }

  get className() { return this.getAttribute('class') || ''; }
  set className(value) { this.setAttribute('class', value); }
  get id() { return this.getAttribute('id') || ''; }
  set id(value) { this.setAttribute('id', value); }
  get title() { return this.getAttribute('title') || ''; }
  set title(value) { this.setAttribute('title', value); }
  get type() { return this.getAttribute('type') || ''; }
  set type(value) { this.setAttribute('type', value); }
  get tabIndex() { return Number(this.getAttribute('tabindex') ?? -1); }
  set tabIndex(value) { this.setAttribute('tabindex', value); }
  get disabled() { return this.hasAttribute('disabled'); }
  set disabled(value) { if (value) this.setAttribute('disabled', ''); else this.removeAttribute('disabled'); }
  get hidden() { return this.hasAttribute('hidden'); }
  set hidden(value) { if (value) this.setAttribute('hidden', ''); else this.removeAttribute('hidden'); }

  get classList() {
    const el = this;
    const list = () => el.className.split(/\s+/).filter(Boolean);
    return {
      contains: (name) => list().includes(name),
      add: (...names) => { el.className = [...new Set([...list(), ...names])].join(' '); },
      remove: (...names) => { el.className = list().filter((n) => !names.includes(n)).join(' '); },
      toggle(name, force) {
        const on = force === undefined ? !list().includes(name) : Boolean(force);
        if (on) this.add(name); else this.remove(name);
        return on;
      },
    };
  }

  // --- tree ---
  _adopt(node) {
    const child = typeof node === 'string' ? new FakeText(this.ownerDocument, node) : node;
    child.remove();
    child.parentNode = this;
    if (child.nodeType === 1) this.ownerDocument.attached.push(child.localName);
    return child;
  }

  appendChild(node) {
    this.childNodes.push(this._adopt(node));
    return node;
  }

  append(...nodes) { for (const n of nodes) this.appendChild(n); }
  prepend(...nodes) { this.childNodes.unshift(...nodes.map((n) => this._adopt(n))); }
  replaceChildren(...nodes) {
    for (const child of [...this.childNodes]) child.remove();
    this.append(...nodes);
  }

  insertBefore(node, reference) {
    if (!reference) return this.appendChild(node);
    const child = this._adopt(node);
    this.childNodes.splice(this.childNodes.indexOf(reference), 0, child);
    return node;
  }

  get children() { return this.childNodes.filter((n) => n.nodeType === 1); }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get firstElementChild() { return this.children[0] || null; }

  contains(node) {
    for (let n = node; n; n = n.parentNode) if (n === this) return true;
    return false;
  }

  get textContent() { return this.childNodes.map((n) => n.textContent).join(''); }
  set textContent(value) {
    for (const child of [...this.childNodes]) child.remove();
    if (String(value) !== '') this.appendChild(new FakeText(this.ownerDocument, value));
  }

  normalize() {
    const merged = [];
    for (const child of this.childNodes) {
      const last = merged[merged.length - 1];
      if (child.nodeType === 3) {
        if (child.data === '') { child.parentNode = null; continue; }
        if (last && last.nodeType === 3) { last.data += child.data; child.parentNode = null; continue; }
      } else {
        child.normalize();
      }
      merged.push(child);
    }
    this.childNodes = merged;
  }

  _matches(rules) {
    return rules.some((rule) => (!rule.tag || rule.tag === this.localName.toLowerCase())
      && rule.classes.every((c) => this.classList.contains(c))
      && rule.attrs.every((a) => this.hasAttribute(a)));
  }

  querySelectorAll(selector) {
    const rules = compile(selector);
    const out = [];
    const walk = (el) => {
      for (const child of el.children) {
        if (child._matches(rules)) out.push(child);
        walk(child);
      }
    };
    walk(this);
    return out;
  }

  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  matches(selector) { return this._matches(compile(selector)); }
  closest(selector) {
    const rules = compile(selector);
    for (let n = this; n && n.nodeType === 1; n = n.parentNode) if (n._matches(rules)) return n;
    return null;
  }

  // --- events (bubbling only) ---
  addEventListener(type, listener) {
    if (!this._listeners) this._listeners = new Map();
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    this._listeners.get(type).push(listener);
  }

  removeEventListener(type, listener) {
    const list = this._listeners && this._listeners.get(type);
    if (list && list.includes(listener)) list.splice(list.indexOf(listener), 1);
  }

  /* Call the `type` listeners of this element and its ancestors. Returns the event object. */
  dispatch(type, init = {}) {
    const event = {
      type, target: this, defaultPrevented: false, stopped: false, ...init,
      preventDefault() { this.defaultPrevented = true; },
      stopPropagation() { this.stopped = true; },
    };
    for (let n = this; n && !event.stopped; n = n.parentNode) {
      for (const listener of [...((n._listeners && n._listeners.get(type)) || [])]) listener(event);
    }
    return event;
  }

  click() {
    if (this.disabled) return null; // a disabled control does not fire click
    return this.dispatch('click');
  }

  focus() { this.ownerDocument.activeElement = this; }
  scrollIntoView() {}
  getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
}

export function installFakeDom() {
  const document = {
    created: [],
    attached: [],
    createElement(tag) {
      this.created.push(String(tag).toLowerCase());
      return new FakeElement(this, tag);
    },
    createElementNS(namespace, tag) {
      this.created.push(String(tag).toLowerCase());
      return new FakeElement(this, tag, namespace);
    },
    createTextNode(data) { return new FakeText(this, data); },
    getElementById(id) {
      return this.documentElement.querySelectorAll('*').find((el) => el.getAttribute('id') === id) || null;
    },
    addEventListener() {},
    activeElement: null,
  };
  document.documentElement = new FakeElement(document, 'html');
  document.body = document.documentElement.appendChild(new FakeElement(document, 'body'));
  globalThis.document = document;
  return document;
}

/* A compact outline of a subtree, for assertions and failure messages: `p{"Hi "a[href]{"x"}}`. */
export function outline(node) {
  if (node.nodeType === 3) return JSON.stringify(node.data);
  const attrs = node.attributes.filter((a) => ['href', 'class'].includes(a.name))
    .map((a) => `[${a.name}=${a.value}]`).join('');
  return `${node.localName}${attrs}{${node.childNodes.map(outline).join('')}}`;
}
