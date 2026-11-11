import assert from "node:assert/strict";
import { test } from "node:test";
import { renderArticleMarkdown } from "../src/lib/markdown";

test("section navigation includes sections and one child level with stable unique anchors", () => {
  const { content, sections } = renderArticleMarkdown("## **Setup**\n\n### Details\n\n## Setup\n\n## Setup 2\n\n<!-- ## Hidden -->\n\n<details><summary>More</summary><h1>Nested heading</h1></details>");
  assert.deepEqual(sections, [
    { id: "section-setup", title: "Setup", level: 2 },
    { id: "section-details", title: "Details", level: 3 },
    { id: "section-setup-2", title: "Setup", level: 2 },
    { id: "section-setup-2-2", title: "Setup 2", level: 2 },
  ]);
  for (const section of sections) assert.ok(content.includes(`id="${section.id}"`));
  assert.deepEqual(renderArticleMarkdown("Only a paragraph").sections, []);
  assert.equal(renderArticleMarkdown("## Section\n\n### Child\n\n#### Detail").sections.length, 2);
  assert.equal(renderArticleMarkdown("### Third level\n\n#### Fourth level").sections.length, 2);
});
