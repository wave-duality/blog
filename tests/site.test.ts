import assert from "node:assert/strict";
import { test } from "node:test";
import { withBasePath } from "../src/lib/site";

test("GitHub project paths cover images, posts, and PDFs without changing external URLs", () => {
  const prefix = "/personal-blog";
  for (const url of ["/me.png", "/uploads/figure.png", "/uploads/paper.pdf", "/posts/example/"]) {
    assert.equal(withBasePath(url, prefix), `${prefix}${url}`);
  }
  for (const url of ["#equation-1", "https://example.com/image.png", "//example.com/image.png", "mailto:me@example.com", "/personal-blog/uploads/figure.png"]) {
    assert.equal(withBasePath(url, prefix), url);
  }
  assert.equal(withBasePath("/uploads/image.png", ""), "/uploads/image.png");
});
