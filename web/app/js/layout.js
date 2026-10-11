/* The frame: the rail (collapsible; a drawer on phones), the side panel (docked on wide screens,
 * a floating sheet below 1101px, a bottom sheet on phones), its tabs, and Esc.
 *
 * Layout state lives in attributes on <html> so CSS does the work and boot.js can set it before
 * the first paint: data-rail="collapsed|expanded", data-drawer="closed|open",
 * data-panel="closed|open". Every toggle keeps `aria-expanded` in step. Whatever is covered by an
 * open drawer or sheet is `inert`, so focus cannot wander behind the scrim.
 */
import { bindAttr } from './i18n.js';
import { isImeEvent } from './composer.js';
import { panel as panelView } from './registry.js';
import { on, storage, STORAGE_KEYS } from './state.js';

export function initLayout({ onEscape }) {
  const root = document.documentElement;
  const $ = (id) => document.getElementById(id);
  const rail = $('rail');
  const main = $('main');
  const panel = $('panel');
  const railToggle = $('rail-toggle');
  const menuButton = $('menu-button');
  const panelToggle = $('panel-toggle');
  const tabs = [...panel.querySelectorAll('[role="tab"]')];

  const phone = window.matchMedia('(max-width: 760px)');
  const docked = window.matchMedia('(min-width: 1101px)');

  const railExpanded = () => root.getAttribute('data-rail') === 'expanded';
  const drawerOpen = () => root.getAttribute('data-drawer') === 'open';
  const panelOpen = () => root.getAttribute('data-panel') === 'open';
  const currentTab = () => (tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') || tabs[0]);

  function syncInert() {
    const drawer = phone.matches && drawerOpen();
    const sheet = !docked.matches && panelOpen();
    main.inert = drawer || sheet;
    panel.inert = drawer || !panelOpen();
    rail.inert = sheet || (phone.matches && !drawerOpen());
  }

  function syncRail() {
    if (phone.matches) {
      // In the drawer this button closes it.
      railToggle.setAttribute('aria-expanded', String(drawerOpen()));
      bindAttr(railToggle, 'aria-label', 'drawer_close');
    } else {
      railToggle.setAttribute('aria-expanded', String(railExpanded()));
      bindAttr(railToggle, 'aria-label', railExpanded() ? 'rail_collapse' : 'rail_expand');
    }
    railToggle.title = railToggle.getAttribute('aria-label');
    menuButton.setAttribute('aria-expanded', String(drawerOpen()));
  }

  function setRail(expanded) {
    root.setAttribute('data-rail', expanded ? 'expanded' : 'collapsed');
    storage.set(STORAGE_KEYS.rail, expanded ? 'expanded' : 'collapsed');
    syncRail();
  }

  function setDrawer(open, { restoreFocus = true } = {}) {
    if (open && panelOpen() && !docked.matches) setPanel(false, { restoreFocus: false });
    root.setAttribute('data-drawer', open ? 'open' : 'closed');
    syncRail();
    syncInert();
    if (open) $('new-chat').focus();
    else if (restoreFocus && phone.matches) menuButton.focus();
  }

  function showTab(name) {
    const view = panelView(name);
    const container = $(`tabpanel-${name}`);
    if (!view || !container) return;
    try {
      if (!view.mounted) {
        view.mounted = true;
        if (typeof view.mount === 'function') view.mount(container);
      }
      if (typeof view.show === 'function') view.show(container);
    } catch (err) {
      console.error(`panel "${name}" failed`, err);
    }
  }

  function selectTab(name, { focus = false } = {}) {
    for (const tab of tabs) {
      const selected = tab.getAttribute('data-tab') === name;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      const view = $(tab.getAttribute('aria-controls'));
      if (view) view.hidden = !selected;
      if (selected && focus) tab.focus();
    }
    if (panelOpen()) showTab(name);
  }

  function setPanel(open, { restoreFocus = true } = {}) {
    if (open && drawerOpen()) setDrawer(false, { restoreFocus: false });
    root.setAttribute('data-panel', open ? 'open' : 'closed');
    panelToggle.setAttribute('aria-expanded', String(open));
    bindAttr(panelToggle, 'aria-label', open ? 'panel_hide' : 'panel_show');
    panelToggle.title = panelToggle.getAttribute('aria-label');
    syncInert();
    if (open) {
      showTab(currentTab().getAttribute('data-tab'));
      if (!docked.matches) currentTab().focus();
    } else if (restoreFocus && !docked.matches) {
      panelToggle.focus();
    }
  }

  railToggle.addEventListener('click', () => {
    if (phone.matches) setDrawer(false);
    else setRail(!railExpanded());
  });
  menuButton.addEventListener('click', () => setDrawer(!drawerOpen()));
  panelToggle.addEventListener('click', () => setPanel(!panelOpen()));
  $('panel-close').addEventListener('click', () => setPanel(false));
  $('scrim').addEventListener('click', () => {
    if (drawerOpen()) setDrawer(false);
    if (panelOpen() && !docked.matches) setPanel(false);
  });

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectTab(tab.getAttribute('data-tab')));
    tab.addEventListener('keydown', (event) => {
      let next = null;
      if (event.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length];
      else if (event.key === 'ArrowLeft') next = tabs[(index - 1 + tabs.length) % tabs.length];
      else if (event.key === 'Home') next = tabs[0];
      else if (event.key === 'End') next = tabs[tabs.length - 1];
      if (!next) return;
      event.preventDefault();
      selectTab(next.getAttribute('data-tab'), { focus: true });
    });
  });

  // Esc: close what is on top first; otherwise it stops the response.
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || event.defaultPrevented || isImeEvent(event)) return;
    if (phone.matches && drawerOpen()) setDrawer(false);
    else if (!docked.matches && panelOpen()) setPanel(false);
    else onEscape();
  });

  phone.addEventListener('change', () => {
    if (drawerOpen()) root.setAttribute('data-drawer', 'closed');
    syncRail();
    syncInert();
  });
  docked.addEventListener('change', () => {
    // Leaving the docked layout with the panel open would cover the conversation: close it.
    if (!docked.matches && panelOpen()) setPanel(false, { restoreFocus: false });
    syncInert();
  });
  on('lang', () => {
    railToggle.title = railToggle.getAttribute('aria-label');
    panelToggle.title = panelToggle.getAttribute('aria-label');
  });

  // boot.js chose data-rail and data-panel; bring the ARIA state in line.
  if (!root.hasAttribute('data-drawer')) root.setAttribute('data-drawer', 'closed');
  syncRail();
  panelToggle.setAttribute('aria-expanded', String(panelOpen()));
  bindAttr(panelToggle, 'aria-label', panelOpen() ? 'panel_hide' : 'panel_show');
  panelToggle.title = panelToggle.getAttribute('aria-label');
  selectTab(currentTab().getAttribute('data-tab'));
  syncInert();

  return {
    closeDrawer() {
      if (drawerOpen()) setDrawer(false, { restoreFocus: false });
    },
    /* Open the side panel on a tab: 'portfolio' | 'orders' | 'mandate' | 'settings'. */
    openPanel(name) {
      selectTab(name);
      setPanel(true);
    },
    isPhone: () => phone.matches,
  };
}
