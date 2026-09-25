export function parseCommaList(s: string): string[] {
	return s
		.split(',')
		.map((t) => t.trim())
		.filter(Boolean);
}
