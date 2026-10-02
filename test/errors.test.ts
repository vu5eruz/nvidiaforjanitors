import { describe, expect, it } from 'vitest';
import { fetchWorker } from './utils';

/**
 * Unit tests for the error responses `src/index.ts` can emit.
 *
 * Every request below is rejected by the worker itself, before it dispatches
 * anything to NVIDIA NIM, so these tests never touch the network. The few
 * non-error branches (landing page, browser redirects, CORS preflights) are
 * included at the top to pin the behavior of the remaining request flow.
 */

const HOST = 'https://proxy.example';

/** Authorization header carrying a syntactically valid (fake) API key. */
const AUTH = { Authorization: 'Bearer nvapi-fake-key-for-tests' };

function newRequest(path = '/', init: RequestInit = {}): Request {
	return new Request(HOST + path, init);
}

/** A POST request with a JSON body, open for header overrides. */
function post(headers: Record<string, string>, body: string): Request {
	return newRequest('/', { method: 'POST', headers, body });
}

function postJson(body: string, headers: Record<string, string> = {}): Request {
	return post({ ...AUTH, 'Content-Type': 'application/json', ...headers }, body);
}

/** Consumes the body asserting the exact status and message. */
async function expectErrorResponse(response: Response, status: number, message: string): Promise<Response> {
	expect(response.status).toBe(status);
	expect(await response.text()).toBe(message);
	return response;
}

/** Consumes the body asserting the 400 status, the message prefix and any details. */
async function expectPayloadError(response: Response, ...fragments: string[]): Promise<Response> {
	expect(response.status).toBe(400);
	const body = await response.text();
	expect(body.startsWith('Missing/Invalid request payload')).toBe(true);
	// The parsing details are always included after the generic message.
	expect(body.length).toBeGreaterThan('Missing/Invalid request payload'.length);
	for (const fragment of fragments) expect(body).toContain(fragment);
	return response;
}

/**
 * A request body that passes schema validation. It is only ever used as the
 * base object for mutations that break exactly one field, so the request is
 * always rejected before the upstream dispatch.
 */
const VALID_PAYLOAD = {
	messages: [{ content: 'Why is the sky blue?', role: 'user' }],
	model: 'meta/llama-3.1-405b-instruct',
	stream: false,
	temperature: 0.2,
};

describe('405 Method Not Allowed', () => {
	it.each(['PUT', 'DELETE', 'PATCH', 'TRACE'])('rejects %s requests', async (method) => {
		const response = await fetchWorker(newRequest('/', { method }));
		const rejected = await expectErrorResponse(response, 405, 'Method Not Allowed');
		expect(rejected.headers.get('Allow')).toBe('GET, HEAD, OPTIONS, POST');
	});

	it('uses a wildcard CORS policy when the request has no Origin', async () => {
		const response = await fetchWorker(newRequest('/', { method: 'PUT' }));
		const rejected = await expectErrorResponse(response, 405, 'Method Not Allowed');
		expect(rejected.headers.get('Access-Control-Allow-Origin')).toBe('*');
		expect(rejected.headers.get('Access-Control-Allow-Credentials')).toBeNull();
		expect(rejected.headers.get('Vary')).toBe('Origin');
	});

	it('reflects the request Origin when present', async () => {
		const response = await fetchWorker(newRequest('/', { method: 'PUT', headers: { Origin: 'https://app.example' } }));
		const rejected = await expectErrorResponse(response, 405, 'Method Not Allowed');
		expect(rejected.headers.get('Access-Control-Allow-Origin')).toBe('https://app.example');
		expect(rejected.headers.get('Access-Control-Allow-Credentials')).toBe('true');
	});
});

describe('301 redirect for browser traffic', () => {
	it('redirects GET requests on API paths to the home page', async () => {
		const response = await fetchWorker(newRequest('/v1/chat/completions'));
		expect(response.status).toBe(301);
		expect(response.headers.get('Location')).toBe(HOST + '/');
		// CORS headers are intentionally omitted on browser-facing redirects.
		expect(response.headers.get('Access-Control-Allow-Origin')).toBeNull();
	});

	it('redirects HEAD requests on API paths too', async () => {
		const response = await fetchWorker(newRequest('/v1/chat/completions', { method: 'HEAD' }));
		expect(response.status).toBe(301);
		expect(response.headers.get('Location')).toBe(HOST + '/');
	});

	it('serves the landing page on the root path', async () => {
		const response = await fetchWorker(newRequest('/'));
		expect(response.status).toBe(200);
		expect(await response.text()).toBe('Hello, Images!');
	});
});

describe('204 CORS preflight', () => {
	it('answers preflights with the full contract', async () => {
		const response = await fetchWorker(
			newRequest('/', {
				method: 'OPTIONS',
				headers: {
					Origin: 'https://app.example',
					'Access-Control-Request-Headers': 'Authorization, Content-Type',
				},
			}),
		);
		expect(response.status).toBe(204);
		expect(await response.text()).toBe('');
		expect(response.headers.get('Access-Control-Allow-Origin')).toBe('https://app.example');
		expect(response.headers.get('Access-Control-Allow-Credentials')).toBe('true');
		expect(response.headers.get('Access-Control-Allow-Headers')).toBe('Authorization, Content-Type');
		expect(response.headers.get('Access-Control-Allow-Methods')).toBe('GET, HEAD, OPTIONS, POST');
		expect(response.headers.get('Access-Control-Max-Age')).toBe('86400');
		expect(response.headers.get('Allow')).toBe('GET, HEAD, OPTIONS, POST');
	});

	it('answers preflights without an Origin with a wildcard', async () => {
		const response = await fetchWorker(
			newRequest('/', {
				method: 'OPTIONS',
				headers: { 'Access-Control-Request-Headers': 'Authorization' },
			}),
		);
		expect(response.status).toBe(204);
		expect(response.headers.get('Access-Control-Allow-Origin')).toBe('*');
		expect(response.headers.get('Access-Control-Allow-Credentials')).toBeNull();
		expect(response.headers.get('Access-Control-Allow-Headers')).toBe('Authorization');
	});

	it('omits Access-Control-Allow-Headers when the preflight lists none', async () => {
		const response = await fetchWorker(newRequest('/', { method: 'OPTIONS', headers: { Origin: 'https://app.example' } }));
		expect(response.status).toBe(204);
		expect(response.headers.get('Access-Control-Allow-Headers')).toBeNull();
	});
});

describe('401 Unauthorized', () => {
	it('rejects requests without an Authorization header', async () => {
		const response = await fetchWorker(post({ 'Content-Type': 'application/json' }, '{}'));
		const rejected = await expectErrorResponse(response, 401, 'Missing/Invalid Authorization Header');
		expect(rejected.headers.get('WWW-Authenticate')).toBe('Bearer');
	});

	it.each([
		'Basic dXNlcjpwYXNz', // wrong scheme
		'ApiKey nvapi-xxxx', // non-standard scheme
		'nvapi-xxxx', // bare key with no scheme at all
		'Bearer', // Bearer with nothing after the scheme
		'BEARER', // ...regardless of case
	])('rejects malformed Authorization header %j', async (authorization) => {
		const response = await fetchWorker(postJson('{}', { Authorization: authorization }));
		const rejected = await expectErrorResponse(response, 401, 'Missing/Invalid Authorization Header');
		expect(rejected.headers.get('WWW-Authenticate')).toBe('Bearer');
	});

	it.each(['Bearer ,', 'Bearer , ,'])('rejects Bearer credentials that contain no keys (%j)', async (authorization) => {
		const response = await fetchWorker(postJson('{}', { Authorization: authorization }));
		const rejected = await expectErrorResponse(response, 401, 'At least one API key is required');
		expect(rejected.headers.get('WWW-Authenticate')).toBe('Bearer');
	});

	it('reflects the request Origin on error responses', async () => {
		// Auth-less request, so the error under test is the 401 itself.
		const response = await fetchWorker(post({ 'Content-Type': 'application/json', Origin: 'https://app.example' }, '{}'));
		const rejected = await expectErrorResponse(response, 401, 'Missing/Invalid Authorization Header');
		expect(rejected.headers.get('Access-Control-Allow-Origin')).toBe('https://app.example');
		expect(rejected.headers.get('Access-Control-Allow-Credentials')).toBe('true');
		expect(rejected.headers.get('Vary')).toBe('Origin');
	});
});

describe('415 Unsupported Media Type: Content-Type header', () => {
	it('rejects requests without a Content-Type header', async () => {
		const response = await fetchWorker(post({ ...AUTH }, '{}'));
		await expectErrorResponse(response, 415, 'Missing/Invalid Content-Type Header');
	});

	it.each([
		'text/plain',
		'application/xml',
		'multipart/form-data',
		'application/jsonp', // not application/json
		'application/json; charset=utf-16', // only utf-8 is accepted
	])('rejects Content-Type %j', async (contentType) => {
		const response = await fetchWorker(postJson('{}', { 'Content-Type': contentType }));
		await expectErrorResponse(response, 415, 'Missing/Invalid Content-Type Header');
	});

	it.each([
		'application/json',
		'application/JSON', // the check is case-insensitive
		'application/json; charset=utf-8',
		'application/json;charset=utf-8', // the space after ; is optional
	])('accepts Content-Type %j, so the request fails later at the payload', async (contentType) => {
		const response = await fetchWorker(postJson('{}', { 'Content-Type': contentType }));
		await expectPayloadError(response);
	});
});

describe('400 Bad Request: request payload', () => {
	it.each(['this is not json', '{', ''])('rejects bodies that are not JSON (%j)', async (body) => {
		const response = await fetchWorker(postJson(body));
		// Whatever detail the runtime reports for the syntax error is surfaced.
		await expectPayloadError(response);
	});

	it.each(['null', '[]', '"hello"', '42'])('rejects JSON values that are not request objects (%j)', async (body) => {
		const response = await fetchWorker(postJson(body));
		await expectPayloadError(response, 'payload: ', 'expected object');
	});

	it('rejects payloads missing required fields, naming each one', async () => {
		const response = await fetchWorker(postJson('{}'));
		await expectPayloadError(response, 'messages:', 'model:');
	});

	it.each([
		['messages is not an array', { messages: 'hello' }, ['messages:', 'expected array']],
		[
			'messages contains a role outside the enum',
			{ messages: [{ content: 'hello', role: 'banana' }] },
			['messages.0.role: ', 'Invalid option'],
		],
		['model is empty', { model: '' }, ['model:', 'Too small']],
		['stream is not a boolean', { stream: 'false' }, ['stream:', 'expected boolean']],
		['temperature is above the maximum', { temperature: 2.5 }, ['temperature:', 'Too big']],
		['temperature is negative', { temperature: -0.1 }, ['temperature:', 'Too small']],
		['temperature is not a number', { temperature: '0.5' }, ['temperature:', 'expected number']],
	])('rejects payloads where %s', async (_description, mutation, fragments) => {
		const response = await fetchWorker(postJson(JSON.stringify({ ...VALID_PAYLOAD, ...mutation })));
		await expectPayloadError(response, ...fragments);
	});
});

describe('validation order', () => {
	it('checks the HTTP method before authentication', async () => {
		const response = await fetchWorker(newRequest('/', { method: 'PUT' }));
		await expectErrorResponse(response, 405, 'Method Not Allowed');
	});

	it('checks the Authorization header before the Content-Type', async () => {
		const response = await fetchWorker(newRequest('/', { method: 'POST', headers: { 'Content-Type': 'text/plain' } }));
		await expectErrorResponse(response, 401, 'Missing/Invalid Authorization Header');
	});

	it('checks the Content-Type before the payload', async () => {
		const response = await fetchWorker(postJson('this is not json', { 'Content-Type': 'text/plain' }));
		await expectErrorResponse(response, 415, 'Missing/Invalid Content-Type Header');
	});
});
