import assert from "node:assert/strict";
import { test } from "node:test";
import { renderMarkdown } from "../src/lib/markdown";
import { withBasePath } from "../src/lib/site";

test("local figures survive sanitization with enforced isolation", () => {
  const html = renderMarkdown('<iframe src="/figures/catan-tile-vector.html" title="Tile vector" sandbox="allow-same-origin" srcdoc="evil" onload="evil()" style="position:fixed"></iframe>');
  assert.ok(html.includes(`src="${withBasePath('/figures/catan-tile-vector.html')}"`));
  assert.ok(html.includes('title="Tile vector"'));
  assert.ok(html.includes('sandbox="allow-scripts"'));
  assert.ok(html.includes('class="figure-embed"'));
  assert.doesNotMatch(html, /srcdoc|onload|allow-same-origin|position:fixed/);
});

test("external and malformed figure embeds are removed", () => {
  for (const src of ['https://example.com/embed', '//example.com/embed', '/figures/../secret.html', '/figures/a.html?url=evil', '/other/a.html', 'javascript:alert(1)', '/figures/%2e%2e/secret.html']) {
    assert.doesNotMatch(renderMarkdown(`<iframe src="${src}"></iframe>`), /<iframe/);
  }
  assert.doesNotMatch(renderMarkdown('<iframe srcdoc="hello"></iframe>'), /<iframe/);
});

test("article scripts remain prohibited while details and images survive", () => {
  const html = renderMarkdown('<script>alert(1)</script>\n\n<details><summary>Details</summary>Body</details>\n\n![Board](/figures/board.svg)');
  assert.doesNotMatch(html, /<script/);
  assert.match(html, /<details>/);
  assert.match(html, /<img /);
});
