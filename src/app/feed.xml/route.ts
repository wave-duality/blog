import { getAllPosts } from "@/lib/posts";
import { siteUrl } from "@/lib/site";

export const dynamic = "force-static";

export async function GET() {
  const posts = getAllPosts();

  const items = posts
    .map(
      (post) => `
    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${escapeXml(siteUrl)}/posts/${encodeURIComponent(post.slug)}/</link>
      <description>${escapeXml(post.excerpt)}</description>
      <pubDate>${new Date(post.date).toUTCString()}</pubDate>
      <guid>${escapeXml(siteUrl)}/posts/${encodeURIComponent(post.slug)}/</guid>
    </item>`
    )
    .join("");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Owen Zhang</title>
    <link>${escapeXml(siteUrl)}/</link>
    <description>Personal blog by Owen Zhang</description>
    <language>en-us</language>
    <atom:link href="${escapeXml(siteUrl)}/feed.xml" rel="self" type="application/rss+xml"/>
    ${items}
  </channel>
</rss>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
    },
  });
}

function escapeXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}
