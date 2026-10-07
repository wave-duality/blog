export const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

export const siteUrl = (
  process.env.NEXT_PUBLIC_SITE_URL || `http://localhost:3000${basePath}`
).replace(/\/$/, "");

// Next's Link handles basePath, but Markdown, image sources, and RSS do not.
export function withBasePath(url: string, prefix = basePath): string {
  if (!prefix || !url.startsWith("/") || url.startsWith("//")) return url;
  if (url === prefix || url.startsWith(`${prefix}/`) || url.startsWith(`${prefix}?`) || url.startsWith(`${prefix}#`)) return url;
  return `${prefix}${url}`;
}
