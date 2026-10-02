import { env } from 'cloudflare:workers';
import { createExecutionContext, waitOnExecutionContext } from 'cloudflare:test';
import worker from '../src/index';

export async function fetchWorker(request: Request): Promise<Response> {
	const ctx = createExecutionContext();
	const res = worker.fetch(request, env, ctx);
	await waitOnExecutionContext(ctx);
	return res;
}
