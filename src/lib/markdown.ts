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

export interface ArticleSection { id: string; title: string }

function articleSections() {
  return (tree: Root, file: { data: Record<string, unknown> }) => {
    const headings = tree.children.filter((node): node is Element =>
      node.type === "element" && /^h[1-6]$/.test(node.tagName));
    const text = (node: Element): string => node.children.map(child =>
      child.type === "text" ? child.value : child.type === "element" ? text(child) : "").join("");
    const level = Math.min(...headings.map(node => Number(node.tagName.slice(1))));
    const used = new Set<string>();
    const sections: ArticleSection[] = [];
    for (const heading of headings) {
      const title = text(heading);
      const base = "section-" + (title.toLowerCase().normalize("NFKC")
        .replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "") || "heading");
      let id = base;
      let suffix = 2;
      while (used.has(id)) id = `${base}-${suffix++}`;
      used.add(id);
      heading.properties.id = id;
      if (Number(heading.tagName.slice(1)) === level) sections.push({ id, title });
    }
    file.data.sections = sections;
  };
}

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
  .use(articleSections)
  .use(rehypeKatex, { trust: false, strict: "warn" })
  .use(prefixLocalUrls)
  .use(rehypeStringify);

export function renderMarkdown(markdown: string): string {
  return processor.processSync(markdown).toString();
}

export function renderArticleMarkdown(markdown: string) {
  const file = processor.processSync(markdown);
  return { content: file.toString(), sections: file.data.sections as ArticleSection[] };
}
