import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { renderMarkdown } from "../src/lib/markdown";

// Test a separate content repository; never create or change the author's posts.
const root = fs.mkdtempSync(path.join(os.tmpdir(), "blog-content-test-"));
const posts = path.join(root, "content/posts");
fs.mkdirSync(posts, { recursive: true });
const cwd = process.cwd();
process.chdir(root);
const contentModule = import("../src/lib/posts");

test("publication controls hide drafts from lists and direct access", async () => {
  try {
    const { getAllPosts, getPostBySlug } = await contentModule;
    for (const [slug, flag] of Object.entries({
      live: "published: true", draft: "published: false", missing: "",
      string: 'published: "true"', legacyDraft: "published: true\ndraft: true",
    })) {
      fs.writeFileSync(path.join(posts, `${slug}.md`), `---\ntitle: ${slug}\ndate: "2026-10-04"\n${flag}\n---\nHello.`);
    }
    assert.deepEqual(getAllPosts().map(p => p.slug), ["live"]);
    for (const slug of ["draft", "missing", "string", "legacyDraft", "../outside", "not-found"]) {
      assert.equal(getPostBySlug(slug), null);
    }
    fs.writeFileSync(path.join(posts, "live.md"), '---\ntitle: live\ndate: "2026"\npublished: false\n---\nHello.');
    assert.equal(getPostBySlug("live"), null);
    assert.deepEqual(getAllPosts(), []);
    const previousEnvironment = process.env.NODE_ENV;
    try {
      Object.assign(process.env, { NODE_ENV: "development" });
      assert.equal(getPostBySlug("draft")?.published, false);
    } finally {
      if (previousEnvironment === undefined) delete process.env.NODE_ENV;
      else Object.assign(process.env, { NODE_ENV: previousEnvironment });
    }
  } finally {
    process.chdir(cwd);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("renders inline and display math, tables, and footnotes", () => {
  const html = renderMarkdown('Inline $E=mc^2$.\n\n$$\n\\sum_{i=1}^{n} i\n$$\n\n| x | y |\n| - | - |\n| 1 | 2 |\n\nText[^1]\n\n[^1]: Reference');
  assert.match(html, /class="katex"/);
  assert.match(html, /katex-display/);
  assert.match(html, /<math/);
  assert.match(html, /<table>/);
  assert.match(html, /data-footnotes/);
});

test("keeps book ratings but removes executable HTML", () => {
  const html = renderMarkdown('<div class="book-list"><div class="book"><div class="book-rating" data-stars="4"></div><div class="book-title">A book</div></div></div>\n\n<script>alert(1)</script><img src="x" onerror="alert(1)"><a href="javascript:alert(1)">bad</a>');
  assert.match(html, /class="book-list"/);
  assert.match(html, /data-stars="4"/);
  assert.doesNotMatch(html, /<script|onerror|javascript:/);
});
