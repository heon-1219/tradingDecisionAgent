/* Line icons (24×24, drawn with currentColor), built with DOM calls: never markup strings.
 * Markup asks for one with `<span class="icon" data-icon="name"></span>`; `hydrateIcons` fills it.
 * Icons are decorative (aria-hidden); the control around them carries the accessible name. */

const NS = 'http://www.w3.org/2000/svg';

const PATHS = {
  plus: ['M12 5v14', 'M5 12h14'],
  menu: ['M4 7h16', 'M4 12h16', 'M4 17h16'],
  rail: ['M5 4.5h14a1.5 1.5 0 0 1 1.5 1.5v12a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 18V6A1.5 1.5 0 0 1 5 4.5z', 'M9.5 4.5v15'],
  panel: ['M5 4.5h14a1.5 1.5 0 0 1 1.5 1.5v12a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 18V6A1.5 1.5 0 0 1 5 4.5z', 'M14.5 4.5v15'],
  pencil: ['M4.5 19.5h3.8L19 8.8a1.9 1.9 0 0 0 0-2.7l-1.1-1.1a1.9 1.9 0 0 0-2.7 0L4.5 15.7z', 'M13.5 6.5l4 4'],
  check: ['M5 12.5l4.5 4.5L19 7.5'],
  close: ['M6.5 6.5l11 11', 'M17.5 6.5l-11 11'],
  sun: ['M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z', 'M12 2.8v1.9', 'M12 19.3v1.9', 'M5.5 5.5l1.3 1.3', 'M17.2 17.2l1.3 1.3',
    'M2.8 12h1.9', 'M19.3 12h1.9', 'M5.5 18.5l1.3-1.3', 'M17.2 6.8l1.3-1.3'],
  moon: ['M19.5 14.6A7.8 7.8 0 0 1 9.4 4.5a7.8 7.8 0 1 0 10.1 10.1z'],
  copy: ['M9.5 8.5h9a1.5 1.5 0 0 1 1.5 1.5v9a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 8 19v-9a1.5 1.5 0 0 1 1.5-1.5z',
    'M16 8.5V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16h2.5'],
  chevron: ['M8 10l4 4 4-4'],
  send: ['M12 19V5.5', 'M6.5 11L12 5.5 17.5 11'],
  stop: { fill: ['M8 7h8a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z'] },
  alert: ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z', 'M12 7.5v5.5', 'M12 16.3v.2'],
  spark: { fill: ['M12 2.5c.7 5 3.4 8 9.5 9.5-6.1 1.5-8.8 4.5-9.5 9.5-.7-5-3.4-8-9.5-9.5 6.1-1.5 8.8-4.5 9.5-9.5z'] },
  logo: ['M5 19v-4', 'M10 19v-8', 'M15 19V8', 'M20 19V4'],
  pie: ['M12 3.5v8.5h8.5', 'M20.5 12A8.5 8.5 0 1 1 12 3.5'],
  receipt: ['M7 3.5h10A1.5 1.5 0 0 1 18.5 5v15.5l-3-2-3.5 2-3.5-2-3 2V5A1.5 1.5 0 0 1 7 3.5z', 'M9 8.5h6', 'M9 12.5h4'],
  shield: ['M12 3l7 3v5.2c0 4.4-2.9 8.1-7 9.8-4.1-1.7-7-5.4-7-9.8V6z', 'M9 12l2.2 2.2L15.5 10'],
  sliders: ['M4 7h9', 'M17 7h3', 'M4 17h3', 'M11 17h9', 'M15 9a2 2 0 1 0 0-4 2 2 0 0 0 0 4z', 'M9 19a2 2 0 1 0 0-4 2 2 0 0 0 0 4z'],
  compare: ['M4 8h15', 'M15.5 4.5L19 8l-3.5 3.5', 'M20 16H5', 'M8.5 12.5L5 16l3.5 3.5'],
  dollar: ['M12 3v18', 'M16.5 7.3C15.7 6.1 14.1 5.3 12 5.3c-2.8 0-4.5 1.3-4.5 3.2 0 4.3 9.2 2.3 9.2 6.6 0 1.9-1.9 3.2-4.7 3.2-2.2 0-3.9-.9-4.8-2.3'],
  scale: ['M12 4v16', 'M8 20h8', 'M5 7.5h14', 'M5 7.5L2.5 13.5a2.5 2.5 0 0 0 5 0z', 'M19 7.5l-2.5 6a2.5 2.5 0 0 0 5 0z'],
  refresh: ['M19.5 12a7.5 7.5 0 1 1-2.2-5.3', 'M19.5 4.5v4h-4'],
};

export function icon(name, { size = 20, className = '' } = {}) {
  const spec = PATHS[name] || PATHS.alert;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', String(size));
  svg.setAttribute('height', String(size));
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  svg.setAttribute('class', `svg-icon${className ? ` ${className}` : ''}`);
  const filled = !Array.isArray(spec);
  for (const d of filled ? spec.fill : spec) {
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', d);
    if (filled) {
      path.setAttribute('fill', 'currentColor');
    } else {
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', 'currentColor');
      path.setAttribute('stroke-width', '1.75');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('stroke-linejoin', 'round');
    }
    svg.appendChild(path);
  }
  return svg;
}

/* Fill every `[data-icon]` placeholder under `root` once. */
export function hydrateIcons(root) {
  for (const el of root.querySelectorAll('[data-icon]')) {
    if (el.firstElementChild) continue;
    const size = Number(el.getAttribute('data-icon-size')) || 20;
    el.appendChild(icon(el.getAttribute('data-icon'), { size }));
  }
}
