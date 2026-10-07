import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

test("creates an unpublished paper draft and never overwrites an existing article", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "new-blog-post-"));
  const script = path.resolve("scripts/new-post.mjs");
  try {
    const args = [script, "A new idea", "--paper"];
    assert.equal(spawnSync(process.execPath, args, { cwd: directory }).status, 0);
    const filename = path.join(directory, "content/posts/a-new-idea.md");
    const content = fs.readFileSync(filename, "utf8");
    assert.match(content, /published: false/);
    assert.match(content, /paper: true/);
    fs.writeFileSync(filename, "My existing article");
    assert.equal(spawnSync(process.execPath, args, { cwd: directory }).status, 1);
    assert.equal(fs.readFileSync(filename, "utf8"), "My existing article");
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
