import 'zod/compile';
import z from 'zod';

const JaiRequest = z.object({
	messages: z.array(
		z.object({
			content: z.string().trim().nonempty(),
			role: z.enum(['system', 'user', 'assistant']),
		}),
	),
	model: z.string().nonempty(),
	stream: z.boolean().optional(),
	/////
	frequency_penalty: z.number().optional(),
	repetition_penalty: z.number().optional(),
	temperature: z.number().min(0.0).max(2.0).optional(),
	top_k: z.number().optional(),
	top_p: z.number().optional(),
});

type JaiRequest = z.infer<typeof JaiRequest>;

export default {
	async fetch(request: Request): Promise<Response> {
		// Prepare common response headers for permissive CORS support.
		// All origins are allowed to maximize coverage. Since users have to first fully
		// trust websites with their API keys, there are no security implications.
		const corsHeaders: Record<string, string> = { Vary: 'Origin' };
		const origin = request.headers.get('Origin');
		if (origin) {
			corsHeaders['Access-Control-Allow-Credentials'] = 'true';
			corsHeaders['Access-Control-Allow-Origin'] = origin;
		} else {
			corsHeaders['Access-Control-Allow-Origin'] = '*';
		}

		// Handle anomalous request as soon as possible.
		const allowedMethods = ['GET', 'HEAD', 'OPTIONS', 'POST'];
		if (!allowedMethods.includes(request.method)) {
			return new Response('Method Not Allowed', {
				status: 405,
				headers: {
					Allow: 'GET, HEAD, OPTIONS, POST',
					...corsHeaders,
				},
			});
		}

		// The proxy worker is only intended to run on POST/OPTIONS on a given set of paths.
		// To be as helpful as possible when an user opens a proxy URL in their browser,
		// redirect thems to the home page where they can get more information.
		// CORS headers are not included since doing GET/HEAD through JS is not intended.
		if (request.method === 'GET' || request.method === 'HEAD') {
			const url = new URL(request.url);
			if (url.pathname !== '/') {
				// The use of 301 Moved Permanently has aggressive caching implications of the
				// redirect, which is the intended behavior as neither users nor crawlers are
				// supposed to GET any of the worker's endpoints, and are always better pointed
				// to index.html which contains info about how to use this proxy's URLs.
				// Fun fact: AIs are terrified of 301, they will shy away and plead to change it.
				return Response.redirect(url.origin, 301);
			}
			// TODO: Make a pretty home page on Static Assets
			return new Response('Hello, World!');
		}

		// Handle CORS-preflight requests.
		if (request.method === 'OPTIONS') {
			const headers = request.headers.get('Access-Control-Request-Headers');
			if (headers) {
				corsHeaders['Access-Control-Allow-Headers'] = headers;
			}
			const response = new Response(null, {
				status: 204,
				headers: {
					Allow: 'GET, HEAD, OPTIONS, POST',
					'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS, POST',
					'Access-Control-Max-Age': '86400',
					...corsHeaders,
				},
			});
			return response;
		}

		// Validate the Authorization header and extract all API keys present.
		// Multi-key support is non-standard according to RFC 6750, yet it is possible
		// and a fun hack since users can put arbitraty strings into the API key field.
		// However, until key rotation is implemented, only the first one will be used.
		const rawAuthHeader = (request.headers.get('Authorization') || '').trim();
		const authorization = rawAuthHeader.match(/^Bearer\s+(.*)$/i);
		if (!authorization) {
			return new Response('Missing/Invalid Authorization Header', {
				status: 401,
				headers: { ...corsHeaders, 'WWW-Authenticate': 'Bearer' },
			});
		}
		const rawApiKeys = authorization[1]
			.split(',')
			.map((t) => t.trim())
			.filter(Boolean);
		if (rawApiKeys.length < 1) {
			return new Response('At least one API key is required', {
				status: 401,
				headers: { ...corsHeaders, 'WWW-Authenticate': 'Bearer' },
			});
		}

		// Validate that the request had set the correct content type and encoding.
		// This is purely a sanity check.
		const rawContentTypeHeader = (request.headers.get('Content-Type') || '').trim();
		const applicationJson = rawContentTypeHeader.match(/^application\/json(; *charset=utf-8)?$/i);
		if (!applicationJson) {
			return new Response('Missing/Invalid Content-Type Header', {
				status: 415,
				headers: { ...corsHeaders },
			});
		}

		// Extract and process the request's JSON payload.
		let payload: JaiRequest;
		try {
			payload = JaiRequest.parse(await request.json());
		} catch (error) {
			const summary =
				error instanceof z.ZodError
					? error.issues.map((issue) => `${issue.path.join('.') || 'payload'}: ${issue.message}`).join('; ')
					: error instanceof Error
						? error.message
						: String(error);
			return new Response(`Missing/Invalid request payload: ${summary}`, {
				status: 400,
				headers: { ...corsHeaders },
			});
		}

		// Dispatch the request to NVIDIA NIM.
		const response = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
			method: 'POST',
			body: JSON.stringify(payload),
			headers: {
				Accept: 'application/json, text/event-stream',
				Authorization: `Bearer ${rawApiKeys[0]}`,
				'Content-Type': 'application/json',
			},
		});

		// Rely on Cloudflare Workers' passthrough behavior to avoid recompression
		return new Response(response.body, {
			status: response.status,
			statusText: response.statusText,
			headers: {
				...corsHeaders,
				...response.headers,
			},
		});
	},
} satisfies ExportedHandler<Env>;
