// Checks for the web app's exact decimal helpers (web/app/js/decimal.js).
// Run: node scripts/test_decimal.mjs
//
// Money never goes through a binary float in the UI. These checks use the classic cases where a
// float gives the wrong answer (0.1 + 0.2, 1.005 rounding, amounts beyond 2^53) and compare a few
// thousand random cases against BigInt arithmetic done independently here.
import assert from 'node:assert/strict';
import {
  abs, add, cmp, div, formatFixed, formatMoney, formatPercent, formatQty, groupThousands, isNegative,
  isPositiveDecimal, isZero, mul, parseDecimal, round, sub, toPlain, toTrimmed,
} from '../web/app/js/decimal.js';

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

const plain = (value) => toPlain(value);

test('parse: plain decimals only', () => {
  assert.deepEqual(parseDecimal('12.340'), { int: 12340n, scale: 3 });
  assert.deepEqual(parseDecimal('-0.5'), { int: -5n, scale: 1 });
  assert.deepEqual(parseDecimal(' 7 '), { int: 7n, scale: 0 });
  assert.deepEqual(parseDecimal('+3.0'), { int: 30n, scale: 1 });
  for (const bad of ['', ' ', 'abc', '1,000', '.5', '5.', '--1', 'NaN', 'Infinity', '0x10', '1.2.3', 'e5', '1e',
    '1e+', '1E+1.5', '1e999', '1e-999',
    null, undefined, 12.5, 12, true, {}, [], '１２']) {
    assert.equal(parseDecimal(bad), null, String(bad));
  }
});

test('exponent notation, as Python writes some Decimals ("1E+1"), parses exactly', () => {
  assert.deepEqual(parseDecimal('1E+1'), { int: 10n, scale: 0 });
  assert.deepEqual(parseDecimal('1.20E+2'), { int: 120n, scale: 0 });
  assert.deepEqual(parseDecimal('2.5e-3'), { int: 25n, scale: 4 });
  assert.deepEqual(parseDecimal('-3E-2'), { int: -3n, scale: 2 });
  assert.deepEqual(parseDecimal('5E+0'), { int: 5n, scale: 0 });
  assert.equal(toPlain('1E+1'), '10');
  assert.equal(formatQty('1E+1'), '10');
  assert.equal(formatMoney('6.1001E+2'), '$610.01');
  assert.equal(plain(add('1E+1', '0.5')), '10.5');
});

test('add and sub are exact where floats are not', () => {
  assert.equal(plain(add('0.1', '0.2')), '0.3');
  assert.equal(plain(sub('0.3', '0.1')), '0.2');
  assert.equal(plain(add('9007199254740993', '0.01')), '9007199254740993.01'); // beyond 2^53
  assert.equal(plain(sub('100000.00', '1830.00')), '98170.00');
  assert.equal(plain(sub('10', '10.5')), '-0.5');
  assert.equal(add('1', 'x'), null);
  assert.equal(sub(null, '1'), null);
});

test('mul and div', () => {
  assert.equal(plain(mul('3', '610.00')), '1830.00');
  assert.equal(plain(mul('0.001', '0.001')), '0.000001');
  assert.equal(plain(mul('-2', '1.5')), '-3.0');
  assert.equal(plain(div('1', '3', 4)), '0.3333');
  assert.equal(plain(div('2', '3', 4)), '0.6667');
  assert.equal(plain(div('-2', '3', 2)), '-0.67');
  assert.equal(plain(div('1830.00', '132100.00', 6)), '0.013853');
  assert.equal(div('1', '0'), null);
  assert.equal(div('1', '0.000'), null);
});

test('round is half away from zero, on the digits', () => {
  assert.equal(plain(round('1.005', 2)), '1.01'); // (1.005).toFixed(2) is "1.00"
  assert.equal(plain(round('2.675', 2)), '2.68'); // (2.675).toFixed(2) is "2.67"
  assert.equal(plain(round('-1.005', 2)), '-1.01');
  assert.equal(plain(round('1.004', 2)), '1.00');
  assert.equal(plain(round('7', 2)), '7.00');
  assert.equal(plain(round('0.5', 0)), '1');
  assert.equal(plain(round('-0.5', 0)), '-1');
  assert.equal(plain(round('0.4', 0)), '0');
});

test('cmp, isZero, isNegative, abs', () => {
  assert.equal(cmp('1.10', '1.1'), 0);
  assert.equal(cmp('1.09', '1.1'), -1);
  assert.equal(cmp('-1', '-2'), 1);
  assert.equal(cmp('1', 'x'), null);
  assert.equal(isZero('0.000'), true);
  assert.equal(isZero('0.001'), false);
  assert.equal(isZero('zero'), null);
  assert.equal(isNegative('-0.01'), true);
  assert.equal(isNegative('0'), false);
  assert.equal(plain(abs('-12.5')), '12.5');
});

test('toPlain and toTrimmed', () => {
  assert.equal(toPlain('0.50'), '0.50');
  assert.equal(toPlain({ int: -5n, scale: 3 }), '-0.005');
  assert.equal(toTrimmed('3.000'), '3');
  assert.equal(toTrimmed('3.140'), '3.14');
  assert.equal(toTrimmed('300'), '300');
  assert.equal(toTrimmed('0.0'), '0');
  assert.equal(toTrimmed('bad'), null);
});

test('thousands separators work on the string', () => {
  assert.equal(groupThousands('1234567.5'), '1,234,567.5');
  assert.equal(groupThousands('-1000'), '-1,000');
  assert.equal(groupThousands('999'), '999');
  assert.equal(groupThousands('1000.0001'), '1,000.0001');
  assert.equal(groupThousands('12345678901234567890.12'), '12,345,678,901,234,567,890.12');
  assert.equal(groupThousands('n/a'), 'n/a');
  assert.equal(groupThousands(1000), '');
});

test('formatMoney', () => {
  assert.equal(formatMoney('1830'), '$1,830.00');
  assert.equal(formatMoney('1830.005'), '$1,830.01');
  assert.equal(formatMoney('100000.00'), '$100,000.00');
  assert.equal(formatMoney('-12.5'), '−$12.50');
  assert.equal(formatMoney('12.5', { signed: true }), '+$12.50');
  assert.equal(formatMoney('0', { signed: true }), '$0.00');
  assert.equal(formatMoney('-0.001'), '−$0.001');
  assert.equal(formatMoney('0.5'), '$0.50');
  assert.equal(formatMoney('0.1234'), '$0.1234');
  assert.equal(formatMoney('0.12345'), '$0.1235');
  assert.equal(formatMoney('0.00004'), '$0.00');
  assert.equal(formatMoney('9007199254740993.129'), '$9,007,199,254,740,993.13');
  assert.equal(formatMoney(null), '–');
  assert.equal(formatMoney('abc', { fallback: 'n/a' }), 'n/a');
  assert.equal(formatMoney(12.5), '–'); // a float is refused, never formatted
});

test('formatQty, formatFixed, formatPercent', () => {
  assert.equal(formatQty('3.000'), '3');
  assert.equal(formatQty('1200.5000'), '1,200.5');
  assert.equal(formatQty('0.125'), '0.125');
  assert.equal(formatQty(undefined), '–');
  assert.equal(formatFixed('1234.5', 2), '1,234.50');
  assert.equal(formatFixed('x', 2), '–');
  assert.equal(formatPercent('12.3456'), '12.35%');
  assert.equal(formatPercent('-1.9', { places: 1 }), '−1.9%');
  assert.equal(formatPercent('3.2', { places: 1, signed: true }), '+3.2%');
  assert.equal(formatPercent('0.001', { signed: true }), '0.00%');
  assert.equal(formatPercent(null), '–');
});

test('isPositiveDecimal', () => {
  for (const good of ['1', '0.01', '5000', '12.50']) assert.equal(isPositiveDecimal(good), true, good);
  for (const bad of ['0', '0.00', '-1', '', 'abc', '1,000', '0E+3']) assert.equal(isPositiveDecimal(bad), false, bad);
  assert.equal(isPositiveDecimal('1e3'), true);
});

test('3,000 random cases agree with independent BigInt arithmetic', () => {
  let seed = 20260926;
  const next = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed;
  };
  const SCALE = 6n;
  const UNIT = 10n ** SCALE;
  const toMicros = (text) => {
    const negative = text.startsWith('-');
    const [whole, fraction = ''] = text.replace('-', '').split('.');
    const value = BigInt(whole) * UNIT + BigInt(fraction.padEnd(6, '0'));
    return negative ? -value : value;
  };
  const random = () => {
    const whole = next() % 2000000;
    const places = next() % 5;
    const fraction = places ? `.${String(next() % 10 ** places).padStart(places, '0')}` : '';
    return `${next() % 4 === 0 ? '-' : ''}${whole}${fraction}`;
  };
  for (let i = 0; i < 3000; i += 1) {
    const a = random();
    const b = random();
    assert.equal(toMicros(toPlain(round(add(a, b), 6))), toMicros(a) + toMicros(b), `${a} + ${b}`);
    assert.equal(toMicros(toPlain(round(sub(a, b), 6))), toMicros(a) - toMicros(b), `${a} - ${b}`);
    const product = mul(a, b);
    assert.equal(product.int * 10n ** BigInt(12 - product.scale), toMicros(a) * toMicros(b), `${a} * ${b}`);
    const expected = toMicros(a) < toMicros(b) ? -1 : toMicros(a) > toMicros(b) ? 1 : 0;
    assert.equal(cmp(a, b), expected, `cmp ${a} ${b}`);
    // grouping never changes the digits
    assert.equal(groupThousands(toPlain(parseDecimal(a))).replaceAll(',', ''), toPlain(parseDecimal(a)));
  }
});

console.log(`${checks} decimal checks passed`);
