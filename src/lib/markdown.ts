import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";
import type { Root, Element } from "hast";
import { withBasePath } from "./site";

function prefixLocalUrls() {
  return (tree: Root) => {
    const walk = (parent: Root | Element) => {
      for (const node of parent.children) {
        if (node.type !== "element") continue;
        for (const property of ["href", "src"]) {
          const value = node.properties[property];
          if (typeof value === "string") node.properties[property] = withBasePath(value);
        }
        walk(node);
      }
    };
    walk(tree);
  };
}

// Preserve existing book markup and math markers, but never executable HTML.
// Sanitize before KaTeX so its generated MathML and styles remain intact.
const processor = remark()
  .use(remarkGfm)
  .use(remarkMath)
  .use(remarkRehype, { allowDangerousHtml: true })
  .use(rehypeRaw)
  .use(rehypeSanitize, {
    ...defaultSchema,
    attributes: {
      ...defaultSchema.attributes,
      div: [
        ...(defaultSchema.attributes?.div ?? []),
        ["className", "book-list", "book", "book-rating", "book-title", "book-author"],
        ["dataStars", "1", "2", "3", "4", "5"],
      ],
      code: [["className", /^language-./, "math-inline", "math-display"]],
    },
  })
  .use(rehypeKatex, { trust: false, strict: "warn" })
  .use(prefixLocalUrls)
  .use(rehypeStringify);

export function renderMarkdown(markdown: string): string {
  return processor.processSync(markdown).toString();
}
