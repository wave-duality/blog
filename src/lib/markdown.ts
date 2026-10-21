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
import { normalizeLatex } from "./latex";

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

// Only local, standalone figures may execute scripts inside an isolated frame.
function localFigureEmbeds() {
  return (tree: Root) => {
    const walk = (parent: Root | Element) => {
      parent.children = parent.children.filter(node => {
        if (node.type !== "element") return true;
        if (node.tagName === "iframe") {
          const src = node.properties.src;
          if (typeof src !== "string" || !/^\/figures\/[a-z0-9][a-z0-9-]*\.html$/.test(src)) return false;
          node.properties = {
            src,
            title: typeof node.properties.title === "string" ? node.properties.title : "Interactive figure",
            className: ["figure-embed"], width: "100%", height: "1200",
            loading: "lazy", sandbox: ["allow-scripts"], referrerPolicy: "no-referrer",
          };
          node.children = [];
        } else walk(node);
        return true;
      });
    };
    walk(tree);
  };
}

// Preserve book markup and math markers; scripts remain prohibited in the article.

// Sanitize before KaTeX so its generated MathML and styles remain intact.
const processor = remark()
  .use(remarkGfm)
  .use(remarkMath)
  .use(remarkRehype, { allowDangerousHtml: true })
  .use(rehypeRaw)
  .use(rehypeSanitize, {
    ...defaultSchema,
    tagNames: [...(defaultSchema.tagNames ?? []), "iframe"],
    attributes: {
      ...defaultSchema.attributes,
      iframe: ["src", "title"],
      div: [
        ...(defaultSchema.attributes?.div ?? []),
        ["className", "book-list", "book", "book-rating", "book-title", "book-author", "claim"],
        ["dataStars", "1", "2", "3", "4", "5"],
      ],
      code: [["className", /^language-./, "math-inline", "math-display"]],
    },
  })
  .use(localFigureEmbeds)
  .use(articleSections)
  .use(rehypeKatex, { trust: false, strict: "warn" })
  .use(prefixLocalUrls)
  .use(rehypeStringify);

export function renderMarkdown(markdown: string): string {
  return processor.processSync(normalizeLatex(markdown)).toString();
}

export function renderArticleMarkdown(markdown: string) {
  const file = processor.processSync(normalizeLatex(markdown));
  return { content: file.toString(), sections: file.data.sections as ArticleSection[] };
}
