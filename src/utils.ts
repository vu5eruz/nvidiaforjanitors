import z from 'zod';

export function parseCommaList(s: string): string[] {
	return s
		.split(',')
		.map((t) => t.trim())
		.filter(Boolean);
}

// Turns a payload parsing failure into a human-readable summary for the error
// response. Zod issues name the offending field; a body that is not valid JSON
// at all is described by the runtime's own error message.
export function summarizePayloadError(error: unknown): string {
	if (error instanceof z.ZodError) {
		return error.issues.map((issue) => `${issue.path.join('.') || 'payload'}: ${issue.message}`).join('; ');
	}
	return error instanceof Error ? error.message : String(error);
}
