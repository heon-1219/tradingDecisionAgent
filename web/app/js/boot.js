/* A small classic script loaded in <head>, so it runs before the first paint: it applies the stored
 * theme, language and rail state and opens the side panel on wide screens. Without it the page
 * would flash the wrong theme and shift when main.js (a deferred module) starts. It has no imports
 * and keeps everything in a block, so it adds no globals. */
{
  const root = document.documentElement;
  const stored = (key) => {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  };

  const theme = stored('finagent-app.theme');
  if (theme === 'light' || theme === 'dark') root.setAttribute('data-theme', theme);

  const lang = stored('finagent-app.lang');
  if (lang === 'en' || lang === 'ko') root.setAttribute('lang', lang);

  root.setAttribute('data-rail', stored('finagent-app.rail') === 'expanded' ? 'expanded' : 'collapsed');
  root.setAttribute('data-drawer', 'closed');

  let wide = false;
  try {
    wide = window.matchMedia('(min-width: 1101px)').matches;
  } catch {
    wide = false;
  }
  root.setAttribute('data-panel', wide ? 'open' : 'closed');
}
