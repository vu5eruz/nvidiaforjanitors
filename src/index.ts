import 'zod/compile';
import z from 'zod';

// As of September 26, 2026, JanitorAI only supports sending plain text in messages.
// Should JAI some day add image support matching the schema, the proxy shall let it
// pass through unmodified.

const JaiMessageText = z.string().trim().nonempty();
type JaiMessageText = z.infer<typeof JaiMessageText>;

const JaiMessageContentText = z.object({
	type: z.literal('text'),
	text: JaiMessageText,
});
type JaiMessageContentText = z.infer<typeof JaiMessageContentText>;

const JaiMessageContentImage = z.object({
	type: z.literal('image_url'),
	image_url: z.object({
		url: z.string().nonempty(),
	}),
});
type JaiMessageContentImage = z.infer<typeof JaiMessageContentImage>;

const JaiMessageContent = z.union([JaiMessageContentText, JaiMessageContentImage]);
type JaiMessageContent = z.infer<typeof JaiMessageContent>;

const JaiMessage = z.object({
	content: z.union([JaiMessageText, z.array(JaiMessageContent).nonempty()]),
	role: z.enum(['system', 'user', 'assistant']),
});
type JaiMessage = z.infer<typeof JaiMessage>;

const JaiRequest = z.object({
	messages: z.array(JaiMessage).nonempty(),
	model: z.string().nonempty(),
	stream: z.boolean().optional(),
	/////
	max_tokens: z.int().min(0).optional(),
	frequency_penalty: z.number().optional(),
	repetition_penalty: z.number().optional(),
	temperature: z.number().min(0.0).max(2.0).optional(),
	top_k: z.number().optional(),
	top_p: z.number().optional(),
});
type JaiRequest = z.infer<typeof JaiRequest>;

export default {
	async fetch(request: Request, _env: Env): Promise<Response> {
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
			return new Response('Hello, Images!');
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
				headers: corsHeaders,
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
			return new Response(`Missing/Invalid request payload:\n${summary}`, {
				status: 400,
				headers: corsHeaders,
			});
		}

		// If stream is omitted, like during proxy tests, default it to false, as NVIDIA NIM
		// might default to a streaming response, which messes up proxy tests.
		if (payload.stream === undefined) payload.stream = false;

		// NVIDIA NIM mostly hosts reasoning models that might take an arbitrary amount of
		// tokens while thinking and generating responses. If the request contains a max_tokens
		// value too low, the models might be unable to do anything useful at all, even failing
		// proxy tests. Set max_tokens, if present, to an arbitrarily chosen minimum value.
		if (payload.max_tokens !== undefined && payload.max_tokens < 1024) payload.max_tokens = 1024;

		// Detect if the request is either a chat message or a proxy test.
		// As of September 25, 2026, JanitorAI's proxy test requests can be identified
		// as a single user message with the text "Just say TEST".
		const isProxyTest =
			payload.messages.length === 1 && //
			payload.messages[0].role === 'user' && //
			payload.messages[0].content === 'Just say TEST';

		// JanitorAI's proxy test requests expect errors to be wrapped in a JSON object
		// with an error key, whose string content will be displayed unformatted, thus
		// we have to add the formatting ourselves.
		function errorResponseForProxyTest(status: number, message: string) {
			return Response.json(
				{ error: `PROXY ERROR ${status}:\n${message}` },
				{
					status: status,
					headers: corsHeaders,
				},
			);
		}

		// JanitorAI's chat message requests, as well as other requests besides proxy test,
		// expect errors to be provided in plain text, which will be formatted before being
		// shown in the UI.
		function errorResponseForChatMessage(status: number, message: string) {
			return new Response(`\n${message}`, {
				status: status,
				headers: corsHeaders,
			});
		}

		// Select error response function depending on whether the request is a proxy test.
		const errorResponse = isProxyTest ? errorResponseForProxyTest : errorResponseForChatMessage;

		//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---

		// Embed external images from chat commands into the request payload.
		// This goes as follow:
		// 	1. The user includes commands in their chat message. For example:
		//
		// 			Lorem ipsum
		// 			//image https://example.com/test.png
		// 			dolor sit amet.
		//
		// 	2. The code accumulates a list of URLs to retrieve and where to put the image data,
		// 		then each message is split into its text content and image_url content parts as follow:
		//
		// 			{
		//				content: "Lorem ipsum",
		// 				role: "user"
		// 			},
		// 			{
		//				content: [{
		// 					type: "image_url",
		// 					image_url: {
		// 						url: "https://example.com/test.png",
		// 					},
		// 				}],
		// 				role: "user"
		// 			},
		// 			{
		//				content: "dolor sit amet.",
		// 				role: "user"
		// 			},
		//
		//	3. The code tries to retrieve all the images from the image cache, then injects
		// 		into the payload all found images into their respective content.image_url.url as follow:
		//
		//			`data:${mimeType};base64,${imageData}`
		//
		// 	4. Images that weren't found in the cache are fetched from the network, inspected
		// 		for their mime type, stored in the cache with an expirationTtl of 1 hour, and then
		// 		injected into the payload.
		//
		// 	5. If any image couldn't be retrieved, an error is shown to the user with the list of
		// 		all URLs that failed to fetch.

		const imageParts: JaiMessageContentImage[] = [];
		for (let i = 0; i < payload.messages.length; ++i) {
			const message = payload.messages[i];
			// Don't concern ourselves with images in the chat messages that aren't ours.
			if (message.role !== 'user' || typeof message.content !== 'string') continue;
			let content = message.content;
			// JanitorAI adds the user's persona name at the start of most user messages.
			// Remove this, if present, so we can correctly parse commands at the start.
			let personaName = '';
			const personaNameMatch = content.match(/^[^:]+: /);
			if (personaNameMatch) {
				content = content.substring(personaNameMatch[0].length);
				personaName = personaNameMatch[0];
			}
			// - Don't accept whitespace other than spaces since users on JanitorAI are
			// 	 unlikely if not unable to type such things. If they actually do type some, then
			//   it shall be quietly ignored until an user complains.
			// - Grab everything as part of the URL. It is up to the user to type a valid URL.
			const match = content.match(/^ *\/\/image +(\S+) *$/dm);
			if (!match) continue;
			const [left, right] = match.indices![0];
			const prefix = content.substring(0, left).trim();
			const suffix = content.substring(right).trim();

			const imagePayload: JaiMessageContentImage = {
				type: 'image_url',
				image_url: {
					url: match[1],
				},
			};
			imageParts.push(imagePayload);

			const newMessages: JaiMessage[] = [];
			if (prefix) newMessages.push({ content: personaName + prefix, role: 'user' });
			newMessages.push({ content: [imagePayload], role: 'user' });
			if (suffix) newMessages.push({ content: personaName + suffix, role: 'user' });
			payload.messages.splice(i, 1, ...newMessages);

			// Make sure that the index, when incremented, lands on the suffix, if present, of
			// this message, so the next iteration can process any additional //image commands.
			i += newMessages.length - (suffix ? 2 : 1);
		}

		// TODO: cache layer

		// Resolve uncached images
		if (imageParts.length > 10) {
			// An user might hit this error if they resume an old chat with lots of images.
			// Increase the limit once someone complains.
			return errorResponse(403, 'No more than 10 //image commands allowed.');
		}

		// Workers have a 128 MB memory limit per isolate.
		// While an isolate can handle concurrent requests, for simpliciy, let's assume that
		// we have the full 128 MB of memory to ourselves on each individual request.
		// We allow downloading up to 10 images, 6 MiB each, at worst. That'll make the isolate
		// consume 60 MiB of memory. We then have to encode these images into base64, which will
		// increase memory consumption by an approximate factor of 4/3, thus we'll consume 80 MiB
		// or so in the worst case.
		// Let's hope that leaves enough wiggle room for anything else going on.
		const maxContentLength = 6 * 1024 * 1024;

		// TODO: Promise.all this stuff
		const imageErrorList: string[] = [];
		for (const imagePart of imageParts) {
			console.log(imagePart.image_url.url);

			let url: URL;
			try {
				url = new URL(imagePart.image_url.url);
			} catch {
				imageErrorList.push(`Invalid URL "${imagePart.image_url.url}"`);
				continue;
			}

			if (!['http:', 'https:'].includes(url.protocol)) {
				imageErrorList.push(`Non-HTTP(S) URL disallowed "${url}"`);
				continue;
			}

			let response: Response;
			try {
				response = await fetch(url, { signal: AbortSignal.timeout(10000) });
			} catch {
				imageErrorList.push(`Failed to fetch "${url}"`);
				continue;
			}

			if (response.status !== 200) {
				imageErrorList.push(`Got ${response.status} from ${url}`);
				continue;
			}

			const contentLength = Number.parseInt((response.headers.get('Content-Length') || '').trim(), 10);
			if (Number.isNaN(contentLength) || contentLength <= 0) {
				imageErrorList.push(`Missing/Invalid Content-Length ${url}`);
				continue;
			}
			if (contentLength > maxContentLength) {
				imageErrorList.push(`Content-Length is larger than 6 MiB ${url}`);
				continue;
			}

			// Let's not allow something as cursed as "image/png; charset=utf-8".
			const mimeType = (response.headers.get('Content-Type') || '').trim().toLowerCase();
			if (!['image/png', 'image/jpeg'].includes(mimeType)) {
				imageErrorList.push(`Not a PNG or JPEG from ${url}`);
				continue;
			}

			// While a malicious server can send a response so big it leads the worker to OOM,
			// a faster way to cause OOM is having the user make a massive request. For simplicity
			// of implementation, we trust the users and external servers not to OOM the worker.
			let imageData: Buffer;
			try {
				imageData = Buffer.from(await response.arrayBuffer());
			} catch {
				imageErrorList.push(`Failed to fetch body "${url}"`);
				continue;
			}

			if (imageData.length > maxContentLength) {
				imageErrorList.push(`Image data is larger than 6 MiB ${url}`);
				continue;
			}

			imagePart.image_url.url = `data:${mimeType};base64,${imageData.toString('base64')}`;
			// TODO: cache
		}

		if (imageErrorList.length > 0) {
			for (const imageError of imageErrorList) console.log(imageError);
			return errorResponse(503, "Proxy couldn't resolve image(s):" + imageErrorList.map((e) => `\n - ${e}`));
		}

		//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---//---

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

		if (response.status !== 200) {
			let errorContentType = (response.headers.get('Content-Type') || '').trim().toLowerCase();
			let errorIsJson = errorContentType.match(/json/i);

			let message: string = errorIsJson
				? await response
						.json()
						.then((data: any) => data.detail || data.message)
						.catch((error) => error)
				: await response.text();
			return errorResponse(response.status, message);
		}

		// Rely on Cloudflare Workers' passthrough behavior to avoid recompression
		return new Response(response.body, {
			status: response.status,
			statusText: response.statusText,
			headers: {
				...Object.fromEntries(response.headers),
				...corsHeaders,
			},
		});
	},
} satisfies ExportedHandler<Env>;
