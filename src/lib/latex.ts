import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { RootContent, Root } from "mdast";

const parser = remark().use(remarkGfm).use(remarkMath);
const environments = new Set([
  "equation", "equation*", "align", "align*", "alignat", "alignat*",
  "gather", "gather*", "aligned", "alignedat", "gathered", "split",
  "cases", "dcases", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
  "array", "smallmatrix", "displaymath", "math",
]);

// Preserve source math before Markdown consumes backslashes, underscores, or HTML.
// Generated code markers pass through the existing sanitizer and KaTeX pipeline.
export function normalizeLatex(source: string): string {
  const protectedRanges: [number, number][] = [];
  const visit = (node: Root | RootContent) => {
    if (["code", "inlineCode", "math", "inlineMath", "html", "link", "image", "definition"].includes(node.type)) {
      const start = node.position?.start.offset;
      const end = node.position?.end.offset;
      if (start !== undefined && end !== undefined) protectedRanges.push([start, end]);
      return;
    }
    if ("children" in node) for (const child of node.children) visit(child);
  };
  visit(parser.parse(source));
  // Encode before Markdown parsing, including backslashes and emphasis markers.
  const escape = (value: string) => Array.from(value, char => `&#${char.codePointAt(0)};`).join("");
  const opening = /\\([([])|\\begin\{([A-Za-z]+\*?)\}/g;
  let result = "";
  let cursor = 0;
  for (let match; (match = opening.exec(source));) {
    const start = match.index;
    let slashes = 0;
    for (let i = start - 1; i >= 0 && source[i] === "\\"; i--) slashes++;
    if (slashes % 2 || protectedRanges.some(([a, b]) => start >= a && start < b)) continue;
    const environment = match[2];
    if (environment && !environments.has(environment)) continue;
    let end = -1;
    let body = "";
    if (environment) {
      // Balance nested environments, including nested instances of the same type.
      const tokens = /\\(begin|end)\{([A-Za-z]+\*?)\}/g;
      tokens.lastIndex = opening.lastIndex;
      const stack = [environment];
      for (let token; (token = tokens.exec(source));) {
        if (token[1] === "begin") stack.push(token[2]);
        else if (stack.pop() !== token[2]) break;
        if (!stack.length) {
          end = tokens.lastIndex;
          body = ["displaymath", "math"].includes(environment)
            ? source.slice(opening.lastIndex, token.index) : source.slice(start, end);
          break;
        }
      }
    } else {
      const close = match[1] === "(" ? "\\)" : "\\]";
      const closeIndex = source.indexOf(close, opening.lastIndex);
      if (closeIndex !== -1) {
        end = closeIndex + close.length;
        body = source.slice(opening.lastIndex, closeIndex);
      }
    }
    if (end === -1 || protectedRanges.some(([a, b]) => start < b && end > a)) continue;
    const display = match[1] === "[" || (environment && environment !== "math");
    result += source.slice(cursor, start) + `<code class="math-${display ? "display" : "inline"}">${escape(body)}</code>`;
    cursor = end;
    opening.lastIndex = end;
  }
  return result + source.slice(cursor);
}
