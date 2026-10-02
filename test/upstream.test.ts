import { afterEach, describe, expect, it, vi, type Mock } from 'vitest';
import { fetchWorker } from './utils';

/**
 * Unit tests for the requests `src/index.ts` dispatches to NVIDIA NIM.
 *
 * The global `fetch` is replaced with a recording mock, so no test ever touches
 * the network while still being able to assert exactly what the proxy sends
 * upstream: the endpoint, the method, the headers and, above all, the JSON
 * payload after the worker applied its transformations (stream defaulting,
 * max_tokens flooring, //image embedding, unknown field stripping). How the
 * proxy surfaces upstream answers and errors back to the client is covered
 * at the bottom to complete the dispatch contract.
 *
 * Notice: when testing retrieval of images, make sure to use globally unique URLs
 * as the image cache only gets cleared at the start of the test suite!
 */

const HOST = 'https://proxy.example';
const NIM_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';

/** Authorization header carrying a syntactically valid (fake) API key. */
const AUTH = { Authorization: 'Bearer nvapi-fake-key-for-tests' };

/** A successful chat completion, as NVIDIA NIM would answer. */
const NIM_COMPLETION = {
	id: 'chatcmpl-fake',
	choices: [{ index: 0, message: { content: 'The sky is blue.', role: 'assistant' }, finish_reason: 'stop' }],
};

/** Magic bytes of a tiny PNG/JPEG file, small enough to inline as fixtures. */
const PNG_BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]);
const PNG_DATA_URL = 'data:image/png;base64,iVBORw0KGgo=';
const JPEG_DATA_URL = 'data:image/jpeg;base64,/9j/4A==';

/** A request payload that passes validation and needs no transformation. */
const VALID_PAYLOAD = {
	messages: [{ content: 'Why is the sky blue?', role: 'user' }],
	model: 'meta/llama-3.1-405b-instruct',
};

/** What the proxy sent to NVIDIA NIM, with the JSON body parsed for assertions. */
interface NimDispatch {
	url: string;
	method?: string;
	headers: Record<string, string>;
	body: unknown;
}

/** An outbound fetch call the proxy made for anything but the NVIDIA NIM dispatch. */
interface ImageFetch {
	url: string;
	init?: RequestInit;
}

type FetchHandler = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
type FetchMock = Mock<FetchHandler>;

/** A fetch response serving image bytes, as an image host would. */
function imageResponse(bytes: Uint8Array, mimeType: string): Response {
	return new Response(bytes, { headers: { 'Content-Type': mimeType, 'Content-Length': String(bytes.length) } });
}

/**
 * Replaces the global `fetch` with a mock that answers NVIDIA NIM dispatches
 * with `options.nim` and image fetches from `options.images` (keyed by URL),
 * while recording every call the proxy makes. Image URLs missing from the
 * table answer 404, which the worker reports as an image retrieval failure.
 */
function stubFetch(options: { nim?: () => Response; images?: Record<string, Response> } = {}): FetchMock {
	const nim = options.nim ?? (() => Response.json(NIM_COMPLETION));
	const images = options.images ?? {};
	const fetchMock = vi.fn<FetchHandler>(async (input, _init) => {
		const url = String(input);
		if (url === NIM_URL) return nim();
		const image = images[url];
		if (image) return image;
		return new Response('Not Found', { status: 404 });
	});
	vi.stubGlobal('fetch', fetchMock);
	return fetchMock;
}

afterEach(() => {
	vi.unstubAllGlobals();
});

/** The single dispatch the proxy made to NVIDIA NIM. */
function nimDispatch(fetchMock: FetchMock): NimDispatch {
	const calls = fetchMock.mock.calls.filter(([input]) => String(input) === NIM_URL);
	expect(calls).toHaveLength(1);
	const [input, init] = calls[0]!;
	const rawBody = init?.body;
	if (typeof rawBody !== 'string') throw new Error('the NVIDIA NIM dispatch had no string body');
	return {
		url: String(input),
		method: init?.method,
		headers: (init?.headers ?? {}) as Record<string, string>,
		body: JSON.parse(rawBody),
	};
}

/** Every fetch call the proxy made other than the NVIDIA NIM dispatch. */
function imageFetches(fetchMock: FetchMock): ImageFetch[] {
	return fetchMock.mock.calls.filter(([input]) => String(input) !== NIM_URL).map(([input, init]) => ({ url: String(input), init }));
}

/** A POST of a JSON payload to the proxy, open for header overrides. */
function postJson(payload: unknown, headers: Record<string, string> = {}): Request {
	return new Request(HOST + '/', {
		method: 'POST',
		headers: { ...AUTH, 'Content-Type': 'application/json', ...headers },
		body: JSON.stringify(payload),
	});
}

describe('upstream request', () => {
	it('POSTs to the NVIDIA NIM chat completions endpoint with the API key headers', async () => {
		const fetchMock = stubFetch();
		await fetchWorker(postJson(VALID_PAYLOAD));
		expect(fetchMock).toHaveBeenCalledTimes(1);
		const dispatch = nimDispatch(fetchMock);
		expect(dispatch.url).toBe(NIM_URL);
		expect(dispatch.method).toBe('POST');
		expect(dispatch.headers).toEqual({
			Accept: 'application/json, text/event-stream',
			Authorization: 'Bearer nvapi-fake-key-for-tests',
			'Content-Type': 'application/json',
			'User-Agent': 'nvidiaforjanitors/0.1', // keeps this in sync with index.ts
		});
	});

	it('authenticates with only the first API key when several are provided', async () => {
		const fetchMock = stubFetch();
		await fetchWorker(postJson(VALID_PAYLOAD, { Authorization: 'Bearer nvapi-first, nvapi-second, nvapi-third' }));
		expect(nimDispatch(fetchMock).headers.Authorization).toBe('Bearer nvapi-first');
	});
});

describe('dispatched payload', () => {
	it('forwards the whole conversation unchanged when no transformation applies', async () => {
		const fetchMock = stubFetch();
		const payload = {
			messages: [
				{ content: 'You are a helpful assistant.', role: 'system' },
				{ content: 'Hi!', role: 'user' },
				{ content: 'Hello!', role: 'assistant' },
				{ content: 'How are you?', role: 'user' },
			],
			model: 'meta/llama-3.1-405b-instruct',
			stream: true,
			max_tokens: 2048,
			temperature: 0.7,
			top_p: 0.9,
			top_k: 40,
			frequency_penalty: 0.3,
			repetition_penalty: 1.1,
		};
		await fetchWorker(postJson(payload));
		expect(nimDispatch(fetchMock).body).toStrictEqual(payload);
	});

	it('defaults stream to false when the request omits it', async () => {
		const fetchMock = stubFetch();
		await fetchWorker(postJson(VALID_PAYLOAD));
		expect(nimDispatch(fetchMock).body).toStrictEqual({ ...VALID_PAYLOAD, stream: false });
	});

	it.each([0, 1, 512, 1023])('raises max_tokens %d below the floor up to 1024', async (maxTokens) => {
		const fetchMock = stubFetch();
		await fetchWorker(postJson({ ...VALID_PAYLOAD, max_tokens: maxTokens }));
		expect(nimDispatch(fetchMock).body).toStrictEqual({ ...VALID_PAYLOAD, stream: false, max_tokens: 1024 });
	});

	it.each([1024, 4096])('keeps max_tokens %d at or above the floor unchanged', async (maxTokens) => {
		const fetchMock = stubFetch();
		await fetchWorker(postJson({ ...VALID_PAYLOAD, max_tokens: maxTokens }));
		expect(nimDispatch(fetchMock).body).toStrictEqual({ ...VALID_PAYLOAD, stream: false, max_tokens: maxTokens });
	});

	it('leaves max_tokens out when the request omits it', async () => {
		const fetchMock = stubFetch();
		await fetchWorker(postJson(VALID_PAYLOAD));
		expect(nimDispatch(fetchMock).body).not.toHaveProperty('max_tokens');
	});

	it('strips unknown fields from the payload', async () => {
		const fetchMock = stubFetch();
		await fetchWorker(
			postJson({
				...VALID_PAYLOAD,
				janitor_field: 'should be dropped',
				messages: [{ content: 'Hello!', role: 'user', attachment: 'should be dropped' }],
			}),
		);
		expect(nimDispatch(fetchMock).body).toStrictEqual({
			messages: [{ content: 'Hello!', role: 'user' }],
			model: VALID_PAYLOAD.model,
			stream: false,
		});
	});

	it('passes client-provided image parts through unmodified', async () => {
		const fetchMock = stubFetch();
		const payload = {
			...VALID_PAYLOAD,
			messages: [{ content: [{ type: 'image_url', image_url: { url: 'https://example.test/client.png' } }], role: 'user' }],
		};
		await fetchWorker(postJson(payload));
		expect(nimDispatch(fetchMock).body).toStrictEqual({ ...payload, stream: false });
		// Images that arrived as content parts are neither fetched nor re-embedded.
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});
});

describe('//image command embedding', () => {
	it('splits the message around the command and embeds the fetched image', async () => {
		const fetchMock = stubFetch({ images: { 'https://example.test/pic.png': imageResponse(PNG_BYTES, 'image/png') } });
		await fetchWorker(
			postJson({
				...VALID_PAYLOAD,
				messages: [{ content: 'Alice: Look at this\n//image https://example.test/pic.png\nWhat do you think?', role: 'user' }],
			}),
		);
		// The image is fetched before dispatching, with a cancellation signal attached.
		const fetches = imageFetches(fetchMock);
		expect(fetchMock).toHaveBeenCalledTimes(2);
		expect(fetches).toHaveLength(1);
		expect(fetches[0]?.url).toBe('https://example.test/pic.png');
		expect(fetches[0]?.init?.signal).toBeInstanceOf(AbortSignal);
		expect(nimDispatch(fetchMock).body).toStrictEqual({
			...VALID_PAYLOAD,
			stream: false,
			messages: [
				{ content: 'Alice: Look at this', role: 'user' },
				{ content: [{ type: 'image_url', image_url: { url: PNG_DATA_URL } }], role: 'user' },
				{ content: 'Alice: What do you think?', role: 'user' },
			],
		});
	});

	it.each([
		['https://example.test/pic.png', 'image/png', PNG_BYTES, PNG_DATA_URL],
		['https://example.test/pic.jpg', 'image/jpeg', JPEG_BYTES, JPEG_DATA_URL],
	])('embeds %s images as data URLs under the fetched mime type', async (url, mimeType, bytes, dataUrl) => {
		const fetchMock = stubFetch({ images: { [url]: imageResponse(bytes, mimeType) } });
		await fetchWorker(postJson({ ...VALID_PAYLOAD, messages: [{ content: `//image ${url}`, role: 'user' }] }));
		expect(nimDispatch(fetchMock).body).toStrictEqual({
			...VALID_PAYLOAD,
			stream: false,
			messages: [{ content: [{ type: 'image_url', image_url: { url: dataUrl } }], role: 'user' }],
		});
	});

	it('embeds multiple images in command order', async () => {
		const fetchMock = stubFetch({
			images: {
				'https://example.test/a.png': imageResponse(PNG_BYTES, 'image/png'),
				'https://example.test/b.png': imageResponse(JPEG_BYTES, 'image/jpeg'),
			},
		});
		await fetchWorker(
			postJson({
				...VALID_PAYLOAD,
				messages: [{ content: 'Bob: //image https://example.test/a.png\n//image https://example.test/b.png', role: 'user' }],
			}),
		);
		expect(fetchMock).toHaveBeenCalledTimes(3);
		expect(imageFetches(fetchMock).map((fetch) => fetch.url)).toEqual(['https://example.test/a.png', 'https://example.test/b.png']);
		expect(nimDispatch(fetchMock).body).toStrictEqual({
			...VALID_PAYLOAD,
			stream: false,
			messages: [
				{ content: [{ type: 'image_url', image_url: { url: PNG_DATA_URL } }], role: 'user' },
				{ content: [{ type: 'image_url', image_url: { url: JPEG_DATA_URL } }], role: 'user' },
			],
		});
	});

	it('deduplicates image fetch calls during retrieval', async () => {
		const fetchMock = stubFetch({
			images: {
				'https://example.test/xyz.png': imageResponse(PNG_BYTES, 'image/png'),
				'https://example.test/123.jpg': imageResponse(JPEG_BYTES, 'image/jpeg'),
			},
		});
		await fetchWorker(
			postJson({
				...VALID_PAYLOAD,
				messages: [
					{ content: 'Carl: //image https://example.test/xyz.png', role: 'user' },
					{ content: 'Carl: //image https://example.test/xyz.png', role: 'user' },
					{ content: 'Carl: //image https://example.test/xyz.png', role: 'user' },
					{ content: 'Carl: //image https://example.test/123.jpg', role: 'user' },
					{ content: 'Carl: //image https://example.test/123.jpg', role: 'user' },
				],
			}),
		);
		// two unique image fetches and the NVIDIA NIM fetch
		expect(fetchMock).toHaveBeenCalledTimes(3);
		expect(nimDispatch(fetchMock).body).toStrictEqual({
			...VALID_PAYLOAD,
			stream: false,
			messages: [
				{ content: [{ type: 'image_url', image_url: { url: PNG_DATA_URL } }], role: 'user' },
				{ content: [{ type: 'image_url', image_url: { url: PNG_DATA_URL } }], role: 'user' },
				{ content: [{ type: 'image_url', image_url: { url: PNG_DATA_URL } }], role: 'user' },
				{ content: [{ type: 'image_url', image_url: { url: JPEG_DATA_URL } }], role: 'user' },
				{ content: [{ type: 'image_url', image_url: { url: JPEG_DATA_URL } }], role: 'user' },
			],
		});
	});

	it.each(['assistant', 'system'])('does not process //image commands in %s messages', async (role) => {
		const fetchMock = stubFetch();
		const payload = { ...VALID_PAYLOAD, messages: [{ content: '//image https://example.test/pic.png', role }] };
		await fetchWorker(postJson(payload));
		// Only the NVIDIA NIM dispatch happens; the command text is forwarded verbatim.
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(nimDispatch(fetchMock).body).toStrictEqual({ ...payload, stream: false });
	});

	it('dispatches nothing to NVIDIA NIM when an image cannot be resolved', async () => {
		// The image URL is absent from the stub table, so the fetch answers 404.
		const fetchMock = stubFetch();
		const response = await fetchWorker(
			postJson({ ...VALID_PAYLOAD, messages: [{ content: '//image https://example.test/gone.png', role: 'user' }] }),
		);
		expect(response.status).toBe(503);
		expect(await response.text()).toBe("\nProxy couldn't resolve image(s):\n - Got 404 from https://example.test/gone.png");
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(imageFetches(fetchMock)).toHaveLength(1);
	});

	it('lists all failed URLs when multiple images were provided', async () => {
		const fetchMock = stubFetch();
		const response = await fetchWorker(
			postJson({
				...VALID_PAYLOAD,
				messages: [{ content: '//image https://example.test/gone.png\n//image https://example.test/lost.png', role: 'user' }],
			}),
		);
		expect(response.status).toBe(503);
		expect(await response.text()).toBe(
			"\nProxy couldn't resolve image(s):\n - Got 404 from https://example.test/gone.png\n - Got 404 from https://example.test/lost.png",
		);
		expect(fetchMock).toHaveBeenCalledTimes(2);
		expect(imageFetches(fetchMock)).toHaveLength(2);
	});
});

describe('upstream responses', () => {
	it('relays the upstream response to the client', async () => {
		stubFetch();
		const response = await fetchWorker(postJson(VALID_PAYLOAD));
		expect(response.status).toBe(200);
		expect(await response.json()).toEqual(NIM_COMPLETION);
		// The proxy's CORS policy is merged into the relayed response.
		expect(response.headers.get('Access-Control-Allow-Origin')).toBe('*');
		expect(response.headers.get('Vary')).toBe('Origin');
	});

	it('wraps JSON errors from NVIDIA NIM for chat messages', async () => {
		stubFetch({ nim: () => Response.json({ detail: 'model is overloaded' }, { status: 500 }) });
		const response = await fetchWorker(postJson(VALID_PAYLOAD));
		expect(response.status).toBe(500);
		expect(await response.text()).toBe('\nmodel is overloaded');
		// Errors keep the proxy's CORS policy.
		expect(response.headers.get('Access-Control-Allow-Origin')).toBe('*');
	});

	it('falls back to the message field of JSON errors', async () => {
		stubFetch({ nim: () => Response.json({ message: 'invalid request' }, { status: 400 }) });
		const response = await fetchWorker(postJson(VALID_PAYLOAD));
		expect(response.status).toBe(400);
		expect(await response.text()).toBe('\ninvalid request');
	});

	it('relays plain text errors from NVIDIA NIM verbatim', async () => {
		stubFetch({ nim: () => new Response('gateway exploded', { status: 502, headers: { 'Content-Type': 'text/plain' } }) });
		const response = await fetchWorker(postJson(VALID_PAYLOAD));
		expect(response.status).toBe(502);
		expect(await response.text()).toBe('\ngateway exploded');
	});

	it('wraps JSON errors in the JSON format for proxy tests', async () => {
		const fetchMock = stubFetch({ nim: () => Response.json({ detail: 'invalid API key' }, { status: 401 }) });
		const proxyTest = { messages: [{ content: 'Just say TEST', role: 'user' }], model: VALID_PAYLOAD.model };
		const response = await fetchWorker(postJson(proxyTest));
		expect(response.status).toBe(401);
		expect(await response.json()).toEqual({ error: 'PROXY ERROR 401:\ninvalid API key' });
		// Proxy tests get their stream defaulted too, so NVIDIA NIM never streams back at them.
		expect(nimDispatch(fetchMock).body).toStrictEqual({ ...proxyTest, stream: false });
	});
});
