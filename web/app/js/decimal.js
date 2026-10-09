/* Exact decimal arithmetic and formatting for money, on strings.
 *
 * The API sends every amount as a decimal string ("1830.00"). Nothing here goes through a binary
 * float: a value is parsed into a scaled BigInt `{int, scale}` (value = int / 10^scale), arithmetic
 * is done on those integers, and formatting (thousands separators, rounding to cents) works on the
 * digits. The UI prefers numbers the server already computed; this module is for the few derived
 * figures it does not send (e.g. buying power after a trade), which the UI labels "est.".
 *
 * Every function accepts a decimal string or an already parsed value, and returns `null` (or the
 * `fallback` text, for the formatters) for anything that is not a plain decimal: no exponent, no
 * separators, no NaN. No DOM is touched, so `scripts/test_decimal.mjs` runs it under Node.
 */

const DECIMAL_RE = /^([+-]?)(\d{1,40})(?:\.(\d{1,20}))?$/;
const DASH = '–';

/* "12.340" -> {int: 12340n, scale: 3}. Strings only (plus values this module produced). */
export function parseDecimal(value) {
  if (value && typeof value === 'object' && typeof value.int === 'bigint' && Number.isInteger(value.scale)) {
    return value;
  }
  if (typeof value !== 'string') return null;
  const match = DECIMAL_RE.exec(value.trim());
  if (!match) return null;
  const [, sign, whole, fraction = ''] = match;
  const int = BigInt(whole + fraction);
  return { int: sign === '-' ? -int : int, scale: fraction.length };
}

function pow10(n) {
  return 10n ** BigInt(n);
}

function align(a, b) {
  const scale = Math.max(a.scale, b.scale);
  return [a.int * pow10(scale - a.scale), b.int * pow10(scale - b.scale), scale];
}

function both(a, b) {
  const x = parseDecimal(a);
  const y = parseDecimal(b);
  return x && y ? [x, y] : null;
}

export function add(a, b) {
  const pair = both(a, b);
  if (!pair) return null;
  const [x, y, scale] = align(...pair);
  return { int: x + y, scale };
}

export function sub(a, b) {
  const pair = both(a, b);
  if (!pair) return null;
  const [x, y, scale] = align(...pair);
  return { int: x - y, scale };
}

export function mul(a, b) {
  const pair = both(a, b);
  if (!pair) return null;
  return { int: pair[0].int * pair[1].int, scale: pair[0].scale + pair[1].scale };
}

/* -1, 0 or 1; null when either side is not a decimal. */
export function cmp(a, b) {
  const pair = both(a, b);
  if (!pair) return null;
  const [x, y] = align(...pair);
  return x < y ? -1 : x > y ? 1 : 0;
}

export function isZero(a) {
  const x = parseDecimal(a);
  return x ? x.int === 0n : null;
}

export function isNegative(a) {
  const x = parseDecimal(a);
  return x ? x.int < 0n : null;
}

export function abs(a) {
  const x = parseDecimal(a);
  return x ? { int: x.int < 0n ? -x.int : x.int, scale: x.scale } : null;
}

/* Integer division rounding half away from zero. */
function divRound(numerator, denominator) {
  const negative = (numerator < 0n) !== (denominator < 0n);
  const n = numerator < 0n ? -numerator : numerator;
  const d = denominator < 0n ? -denominator : denominator;
  const quotient = (2n * n + d) / (2n * d);
  return negative ? -quotient : quotient;
}

/* a / b to `places` decimals (half away from zero); null for a zero divisor. */
export function div(a, b, places = 4) {
  const pair = both(a, b);
  if (!pair || pair[1].int === 0n) return null;
  const [x, y] = pair;
  // (x.int / 10^x.scale) / (y.int / 10^y.scale) * 10^places
  const numerator = x.int * pow10(y.scale + places);
  const denominator = y.int * pow10(x.scale);
  return { int: divRound(numerator, denominator), scale: places };
}

/* Round to `places` decimals, half away from zero. */
export function round(a, places = 2) {
  const x = parseDecimal(a);
  if (!x) return null;
  if (x.scale <= places) return { int: x.int * pow10(places - x.scale), scale: places };
  return { int: divRound(x.int, pow10(x.scale - places)), scale: places };
}

/* The plain decimal string, with exactly its scale: {int: 12340n, scale: 3} -> "12.340". */
export function toPlain(a) {
  const x = parseDecimal(a);
  if (!x) return null;
  const negative = x.int < 0n;
  const digits = (negative ? -x.int : x.int).toString().padStart(x.scale + 1, '0');
  const whole = digits.slice(0, digits.length - x.scale);
  const fraction = x.scale ? `.${digits.slice(digits.length - x.scale)}` : '';
  return `${negative ? '-' : ''}${whole}${fraction}`;
}

/* The shortest plain string: trailing zeros of the fraction dropped ("3.000" -> "3"). */
export function toTrimmed(a) {
  const text = toPlain(a);
  if (text === null) return null;
  return text.includes('.') ? text.replace(/0+$/, '').replace(/\.$/, '') : text;
}

/* "1234567.5" -> "1,234,567.5": separators in the integer part of a plain decimal string. */
export function groupThousands(text) {
  if (typeof text !== 'string') return '';
  const match = /^(-?)(\d+)(\.\d+)?$/.exec(text);
  if (!match) return text;
  return `${match[1]}${match[2].replace(/\B(?=(\d{3})+(?!\d))/g, ',')}${match[3] || ''}`;
}

/* A number with separators and a fixed number of decimals: ("1234.5", 2) -> "1,234.50". */
export function formatFixed(value, places = 2, fallback = DASH) {
  const rounded = round(value, places);
  return rounded ? groupThousands(toPlain(rounded)) : fallback;
}

/* US dollars: cents for amounts of a dollar or more, up to four decimals below a dollar (a
 * sub-dollar price), a real minus sign in front. `signed` adds "+" to positive amounts. */
export function formatMoney(value, { signed = false, fallback = DASH } = {}) {
  const x = parseDecimal(value);
  if (!x) return fallback;
  const magnitude = abs(x);
  const small = cmp(magnitude, '1') < 0 && magnitude.int !== 0n;
  let shown = round(magnitude, 2);
  if (small) {
    const fine = round(magnitude, 4);
    const trimmed = toTrimmed(fine);
    const decimals = trimmed.includes('.') ? trimmed.length - trimmed.indexOf('.') - 1 : 0;
    shown = round(fine, Math.max(2, decimals));
  }
  const zero = shown.int === 0n;
  const sign = x.int < 0n && !zero ? '−' : signed && !zero ? '+' : '';
  return `${sign}$${groupThousands(toPlain(shown))}`;
}

/* A quantity of shares: separators, no trailing zeros ("1200.5000" -> "1,200.5"). */
export function formatQty(value, fallback = DASH) {
  const trimmed = toTrimmed(value);
  return trimmed === null ? fallback : groupThousands(trimmed);
}

/* A percentage with a fixed number of decimals: ("12.3456", 2) -> "12.35%". `signed` adds "+". */
export function formatPercent(value, { places = 2, signed = false, fallback = DASH } = {}) {
  const rounded = round(value, places);
  if (!rounded) return fallback;
  const zero = rounded.int === 0n;
  const sign = rounded.int < 0n ? '−' : signed && !zero ? '+' : '';
  return `${sign}${groupThousands(toPlain(abs(rounded)))}%`;
}

/* Is `text` something the server will accept as a positive decimal? Used by the forms. */
export function isPositiveDecimal(text) {
  const x = parseDecimal(text);
  return Boolean(x) && x.int > 0n;
}
