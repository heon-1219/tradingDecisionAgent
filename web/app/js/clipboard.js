/* Copy buttons for code blocks and tables rendered by markdown.js.
 * The Clipboard API needs a secure context (localhost counts); a hidden textarea with
 * execCommand('copy') is the fallback. */
import { bindText } from './i18n.js';

export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall back below */
  }
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.setAttribute('aria-hidden', 'true');
  area.tabIndex = -1;
  area.className = 'clipboard-proxy';
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  area.remove();
  return ok;
}

/* A table as tab-separated text, which pastes into a spreadsheet as cells. */
export function tableToTsv(table) {
  if (!table) return '';
  const clean = (cell) => cell.textContent.replace(/[\t\r\n]+/g, ' ').trim();
  return [...table.rows].map((row) => [...row.cells].map(clean).join('\t')).join('\n');
}

const timers = new WeakMap();

/* One delegated listener for every copy button under `container` (streamed or from history). */
export function initCopyButtons(container) {
  container.addEventListener('click', async (event) => {
    const btn = event.target.closest('button[data-copy]');
    if (!btn || !container.contains(btn)) return;
    const box = btn.closest('.md-block');
    if (!box) return;
    const text = btn.getAttribute('data-copy') === 'table'
      ? tableToTsv(box.querySelector('table'))
      : box.querySelector('code')?.textContent ?? '';
    const ok = await copyText(text);
    btn.focus();
    const label = btn.querySelector('.md-copy-label');
    btn.classList.toggle('is-copied', ok);
    if (label) bindText(label, ok ? 'copied' : 'copy_failed');
    clearTimeout(timers.get(btn));
    timers.set(btn, setTimeout(() => {
      btn.classList.remove('is-copied');
      if (label) bindText(label, 'copy');
    }, 1600));
  });
}
