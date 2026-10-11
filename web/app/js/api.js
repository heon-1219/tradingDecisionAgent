/* The /api client: same-origin relative URLs only.
 *
 * Every state-changing request (POST, PUT, PATCH, DELETE) carries `X-FinAgent-Client: 1` (the
 * server's anti-CSRF check) and `Content-Type: application/json`. A failed request throws an
 * `ApiError` carrying `{status, error, message}` from the JSON error body; `message` is written for
 * the user and is what the UI shows. Aborts are rethrown untouched (`name === 'AbortError'`), so a
 * Stop never looks like an error.
 */
import { t } from './i18n.js';

export const CLIENT_HEADER = Object.freeze({ 'X-FinAgent-Client': '1' });
const WRITE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

export class ApiError extends Error {
  constructor({ status = 0, error = 'error', message, fields, body } = {}) {
    super(message || t('err_generic'));
    this.name = 'ApiError';
    this.status = status;
    this.error = error;
    this.fields = Array.isArray(fields) ? fields : undefined;
    this.body = body;
  }
}

export function isAbort(err) {
  return Boolean(err) && (err.name === 'AbortError' || err.code === 20);
}

/* Build an ApiError from a non-2xx response. The server's error bodies are `{error, message, ...}`;
 * anything else becomes a generic HTTP message. */
export async function readError(res) {
  let body = null;
  try {
    const text = await res.text();
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  const ok = body && typeof body === 'object';
  return new ApiError({
    status: res.status,
    error: ok && typeof body.error === 'string' ? body.error : `http_${res.status}`,
    message: ok && typeof body.message === 'string' && body.message
      ? body.message
      : t('err_http', { status: res.status }),
    fields: ok ? body.fields : undefined,
    body: ok ? body : undefined,
  });
}

export function networkError() {
  return new ApiError({ status: 0, error: 'network_error', message: t('err_network') });
}

/* Headers for a request; write methods get the anti-CSRF header and a JSON content type. */
export function requestHeaders(method, extra) {
  const headers = { Accept: 'application/json', ...extra };
  if (WRITE_METHODS.has(method)) {
    Object.assign(headers, CLIENT_HEADER, { 'Content-Type': 'application/json' });
  }
  return headers;
}

function assertPath(path) {
  if (typeof path !== 'string' || !path.startsWith('/api/') || path.includes('//')) {
    throw new ApiError({ error: 'bad_path', message: t('err_generic') });
  }
}

export async function request(method, path, body, { signal } = {}) {
  assertPath(path);
  const upper = method.toUpperCase();
  const init = {
    method: upper,
    headers: requestHeaders(upper),
    credentials: 'same-origin',
    cache: 'no-store',
    signal,
  };
  if (WRITE_METHODS.has(upper)) init.body = JSON.stringify(body ?? {});
  let res;
  try {
    res = await fetch(path, init);
  } catch (err) {
    if (isAbort(err)) throw err;
    throw networkError();
  }
  if (!res.ok) throw await readError(res);
  if (res.status === 204) return null;
  try {
    return await res.json();
  } catch (err) {
    if (isAbort(err)) throw err;
    throw new ApiError({ status: res.status, error: 'bad_json', message: t('err_generic') });
  }
}

const seg = (value) => encodeURIComponent(String(value));

/* Endpoint helpers used by the shell. Task 4.2's components can call `request` directly for the
 * portfolio, mandate, ticket and settings writes. */
export const api = {
  listSessions: (opts) => request('GET', '/api/sessions', undefined, opts),
  createSession: ({ title, lang } = {}, opts) => request('POST', '/api/sessions', { title, lang }, opts),
  getSession: (id, opts) => request('GET', `/api/sessions/${seg(id)}`, undefined, opts),
  renameSession: (id, title, opts) => request('PATCH', `/api/sessions/${seg(id)}`, { title }, opts),
  getSettings: (opts) => request('GET', '/api/settings', undefined, opts),
};
