export function calculateReadingStats(markdown: string) {
  // Use the source so KaTeX's visual and accessibility output aren't counted twice.
  const text = markdown
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]*>/g, " ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
  const words = text ? text.split(/\s+/).length : 0;
  return {
    readingMinutes: Math.ceil(words / 200),
    // A lightweight estimate, intentionally rounded rather than model-specific.
    tokenCount: text ? Math.max(10, Math.round(text.length / 4 / 10) * 10) : 0,
  };
}
