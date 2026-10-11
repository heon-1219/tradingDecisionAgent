/* One shared popover for elements that explain themselves on demand (source chips).
 *
 * `initPopovers(root, {selector, build})` watches `root` by delegation, so chips created later
 * (streamed answers, re-rendered cards) need no wiring of their own. For an element matching
 * `selector`:
 *   - hovering with a mouse opens it after a short delay, and leaving closes it;
 *   - keyboard focus opens it;
 *   - a click or tap pins it open; another click, a click elsewhere, scrolling or Esc closes it.
 * `build(anchor)` returns the content node, or null for "nothing to show".
 *
 * Keyboard: from the anchor, Tab moves into the popover's first link; Shift+Tab from there returns
 * to the anchor; Tab past its last link closes it and moves on. Esc closes it and returns focus to
 * the anchor (and is consumed, so it does not also stop the response).
 */
import { bindAttr } from '../i18n.js';

const HOVER_OPEN_MS = 140;
const HOVER_CLOSE_MS = 220;
const MARGIN = 8;
const FOCUSABLE = 'a[href], button:not([disabled])';

let open = null; // {anchor, node, pinned}
let hoverTimer = null;
let returning = null; // the anchor that focus is being handed back to

function clearHover() {
  clearTimeout(hoverTimer);
  hoverTimer = null;
}

/* Where to put a popover of `size` next to `anchor` inside `viewport`: below the anchor when it
 * fits, else above, and always inside the viewport's margins. Pure, for the tests. */
export function placePopover(anchor, size, viewport, margin = MARGIN) {
  const maxLeft = Math.max(margin, viewport.width - size.width - margin);
  const left = Math.min(Math.max(margin, anchor.left), maxLeft);
  const below = anchor.bottom + 6;
  const above = anchor.top - 6 - size.height;
  let top = below;
  if (below + size.height > viewport.height - margin && above >= margin) top = above;
  top = Math.min(Math.max(margin, top), Math.max(margin, viewport.height - size.height - margin));
  return { left: Math.round(left), top: Math.round(top) }; // not money: pixels
}

export function closePopover({ restoreFocus = false } = {}) {
  clearHover();
  if (!open) return;
  const { anchor, node } = open;
  open = null;
  node.remove();
  anchor.setAttribute('aria-expanded', 'false');
  if (restoreFocus && anchor.isConnected) {
    returning = anchor; // this focus must not reopen the popover that was just closed
    anchor.focus();
    returning = null;
  }
}

function show(anchor, build, pinned) {
  if (open && open.anchor === anchor) {
    open.pinned = open.pinned || pinned;
    return;
  }
  closePopover();
  const content = build(anchor);
  if (!content) return;
  const node = document.createElement('div');
  node.className = 'popover';
  node.setAttribute('role', 'group');
  bindAttr(node, 'aria-label', 'popover_label');
  node.appendChild(content);
  document.body.appendChild(node);
  const spot = placePopover(anchor.getBoundingClientRect(), node.getBoundingClientRect(),
    { width: document.documentElement.clientWidth, height: document.documentElement.clientHeight });
  node.style.left = `${spot.left}px`;
  node.style.top = `${spot.top}px`;
  anchor.setAttribute('aria-expanded', 'true');
  open = { anchor, node, pinned };

  node.addEventListener('pointerenter', clearHover);
  node.addEventListener('pointerleave', () => {
    if (open && !open.pinned) hoverTimer = setTimeout(() => closePopover(), HOVER_CLOSE_MS);
  });
  node.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      closePopover({ restoreFocus: true });
      return;
    }
    if (event.key !== 'Tab') return;
    const items = [...node.querySelectorAll(FOCUSABLE)];
    const index = items.indexOf(document.activeElement);
    if (event.shiftKey && index <= 0) {
      event.preventDefault();
      anchor.focus();
    } else if (!event.shiftKey && index === items.length - 1) {
      // Back to the anchor without preventing the default: the browser then tabs on from there.
      closePopover({ restoreFocus: true });
    }
  });
  node.addEventListener('focusout', (event) => {
    const next = event.relatedTarget;
    if (open && next && next !== anchor && !node.contains(next)) closePopover();
  });
}

export function initPopovers(root, { selector, build }) {
  if (!root) return;
  const anchorOf = (target) => (target && target.closest ? target.closest(selector) : null);

  root.addEventListener('pointerover', (event) => {
    if (event.pointerType !== 'mouse') return;
    const anchor = anchorOf(event.target);
    if (!anchor || (open && open.anchor === anchor)) {
      if (anchor) clearHover();
      return;
    }
    clearHover();
    hoverTimer = setTimeout(() => show(anchor, build, false), HOVER_OPEN_MS);
  });
  root.addEventListener('pointerout', (event) => {
    if (event.pointerType !== 'mouse') return;
    const anchor = anchorOf(event.target);
    if (!anchor || anchor.contains(event.relatedTarget)) return;
    clearHover();
    if (open && open.anchor === anchor && !open.pinned) {
      hoverTimer = setTimeout(() => closePopover(), HOVER_CLOSE_MS);
    }
  });
  root.addEventListener('focusin', (event) => {
    const anchor = anchorOf(event.target);
    // Keyboard focus only: a mouse click focuses the chip too, and the click handler deals with it.
    if (anchor && anchor !== returning && anchor.matches(':focus-visible')) show(anchor, build, false);
  });
  root.addEventListener('focusout', (event) => {
    const anchor = anchorOf(event.target);
    if (!anchor || !open || open.anchor !== anchor || open.pinned) return;
    if (!open.node.contains(event.relatedTarget)) closePopover();
  });
  root.addEventListener('click', (event) => {
    const anchor = anchorOf(event.target);
    if (!anchor) return;
    clearHover();
    if (open && open.anchor === anchor && open.pinned) closePopover();
    else show(anchor, build, true);
    if (open && open.anchor === anchor) open.pinned = true;
  });
  root.addEventListener('keydown', (event) => {
    const anchor = anchorOf(event.target);
    if (!anchor || !open || open.anchor !== anchor) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      closePopover({ restoreFocus: true });
    } else if (event.key === 'Tab' && !event.shiftKey) {
      const first = open.node.querySelector(FOCUSABLE);
      if (first) {
        event.preventDefault();
        first.focus();
      }
    }
  });

  // Anything that moves the anchor away from the popover closes it.
  document.addEventListener('pointerdown', (event) => {
    if (!open || open.node.contains(event.target) || open.anchor.contains(event.target)) return;
    closePopover();
  }, true);
  document.addEventListener('scroll', (event) => {
    if (open && !open.node.contains(event.target)) closePopover();
  }, true);
  if (typeof globalThis.addEventListener === 'function') {
    globalThis.addEventListener('resize', () => closePopover());
  }
}
