import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { renderMarkdown } from "./markdown";

const postsDirectory = path.join(process.cwd(), "content/posts");
const pagesDirectory = path.join(process.cwd(), "content/pages");

export interface Post {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  content: string;
  paper: boolean;
  author: string;
  abstract: string;
  published: boolean;
}

export interface Page {
  slug: string;
  title: string;
  content: string;
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function readContent(directory: string, slug: string) {
  // A slug must name a single Markdown file within the content directory.
  if (!slug || /[/\\\x00]/.test(slug) || slug === "." || slug === "..") return null;
  const filePath = path.join(directory, `${slug}.md`);
  if (!fs.existsSync(filePath)) return null;
  return matter(fs.readFileSync(filePath, "utf8"));
}

export function getAllPosts(): Post[] {
  if (!fs.existsSync(postsDirectory)) return [];
  return fs.readdirSync(postsDirectory)
    .filter((filename) => filename.endsWith(".md"))
    .map((filename) => getPostBySlug(filename.slice(0, -3)))
    .filter((post): post is Post => post !== null)
    .sort((a, b) => b.date.localeCompare(a.date) || a.slug.localeCompare(b.slug));
}

export function getPostBySlug(slug: string): Post | null {
  const source = readContent(postsDirectory, slug);
  // Missing or malformed publication flags fail closed, including direct URLs.
  if (!source) return null;
  const published = source.data.published === true && source.data.draft !== true;
  if (!published && process.env.NODE_ENV !== "development") return null;
  const { data, content } = source;
  const date = data.date instanceof Date
    ? data.date.toISOString().slice(0, 10)
    : text(data.date);
  if (!/^\d{4}(-\d{2}-\d{2})?$/.test(date) || Number.isNaN(Date.parse(date))) {
    throw new Error(`Invalid publication date in content/posts/${slug}.md`);
  }
  return {
    slug,
    published,
    title: text(data.title, slug),
    date,
    excerpt: text(data.excerpt),
    content: renderMarkdown(content),
    paper: data.paper === true,
    author: text(data.author) || "Owen Zhang",
    abstract: text(data.abstract),
  };
}

export function getPage(slug: string): Page | null {
  const source = readContent(pagesDirectory, slug);
  if (!source) return null;
  return {
    slug,
    title: text(source.data.title, slug),
    content: renderMarkdown(source.content),
  };
}
