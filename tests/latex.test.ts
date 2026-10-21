import assert from "node:assert/strict";
import { test } from "node:test";
import { renderMarkdown } from "../src/lib/markdown";
import { normalizeLatex } from "../src/lib/latex";

test("native delimiters preserve math commands and coexist with dollar math", () => {
  const html = renderMarkdown(String.raw`Inline \(V_{\text{trade}}\geq v\) and $x$.

\[
v \leq V_{\text{trade}} \leq v \quad\Rightarrow\quad V_{\text{trade}}=v.
\]

$$y=2$$`);
  assert.equal((html.match(/class="katex"/g) ?? []).length, 4);
  assert.match(html, /katex-display/);
  assert.doesNotMatch(html, /katex-error/);
});

test("standalone native environments render without dollar wrappers", () => {
  for (const source of [
    String.raw`\begin{equation}E=mc^2\end{equation}`,
    String.raw`\begin{align}a&=b\\c&=d\end{align}`,
    String.raw`\begin{gather*}a=b\\c=d\end{gather*}`,
    String.raw`\begin{equation*}A=\begin{pmatrix}1&2\\3&4\end{pmatrix}\end{equation*}`,
    String.raw`\begin{displaymath}x=1\end{displaymath}`,
  ]) {
    const html = renderMarkdown(source);
    assert.match(html, /katex-display/);
    assert.doesNotMatch(html, /katex-error/);
  }
});

test("code, existing math, links, comments, and incomplete syntax stay unchanged", () => {
  for (const source of [
    '`\\(x\\)`', '```tex\n\\[x\\]\n```', '    \\begin{align}x\\end{align}',
    String.raw`$\begin{matrix}1\end{matrix}$`, String.raw`<!-- \(secret\) -->`,
    String.raw`[link](https://example.com/\(x\))`, String.raw`Unfinished \[x`,
    String.raw`\\(literal\\)`, String.raw`\begin{itemize}\item Item\end{itemize}`,
  ]) assert.equal(normalizeLatex(source), source);
});

test("math renders inside a list and a details block without enabling unsafe HTML", () => {
  const html = renderMarkdown(String.raw`- Value: \(x < y\)

<details>
<summary>Proof</summary>

\[x=1\]

</details>

\(\href{javascript:alert(1)}{bad}\)`);
  assert.match(html, /<li>/);
  assert.match(html, /katex-display/);
  assert.doesNotMatch(html, /href="javascript:/);
});
