# Owen's blog

A personal blog for ordinary writing and technical articles, built from Markdown and hosted on GitHub Pages.

Public site: **https://wave-duality.github.io/blog/**

## Write a post

```sh
npm run new:post -- "My next post"
```

This creates a Markdown file in `content/posts/`. Open that file in your editor and write below the `---` header. To edit an existing post, open its file in that same folder. Your bio is `content/pages/about.md`.

```markdown
---
title: "My next post"
date: "2026-10-07"
excerpt: "A short summary for the homepage."
published: false
paper: false
---

Write normal paragraphs here. **Bold**, *italics*, headings, lists,
links, images, and code blocks are supported. No LaTeX is required.
```

## Preview locally

```sh
npm install
npm run dev
```

Open **http://localhost:3000**. Local development shows drafts as well as published posts, so you can preview the actual article while editing. Saved file changes appear through the development server; no copy-and-paste preview or CMS is needed.

## Publish or update

Set `published: true` when ready. Commit and push to `main`:

```sh
git add content/posts public/uploads content/pages
git commit -m "Update blog articles"
git push origin main
```

The [deployment workflow](https://github.com/wave-duality/blog/actions/workflows/deploy-pages.yml) checks and builds the blog, then updates GitHub Pages. Wait for the workflow to succeed before sharing the post URL. Changing a title does not change its URL; renaming its file does.

**`published: false` hides a post from the built website, not from a public GitHub repository.** Keep confidential drafts outside this repository until you are ready to make their contents public. Unpublishing a post does not erase it from Git history or third-party caches.

## Technical articles

Create a paper-style article with:

```sh
npm run new:post -- "A technical article" --paper
```

Set `paper: true` for serif typography, a centered title, author, and abstract. The normal blog layout uses `paper: false`. Both layouts support LaTeX equations, tables, and footnotes.

Inline math: `$E = mc^2$`. Escape literal currency as `\$` when it might be interpreted as math.

Display math:

```markdown
$$
\mathbb{E}[X] = \sum_{i=1}^{n} x_i p_i
$$
```

Place images and PDFs in `public/uploads/`, then use `![Description](/uploads/figure.png)` or `[Download PDF](/uploads/paper.pdf)`. The GitHub Pages path is added automatically. PDFs are files you provide, not generated automatically. Number sections and equations explicitly when needed; automatic academic citations and equation cross-references are not implemented.

Markdown HTML is sanitized before math rendering. The existing book-list classes and star ratings are retained. Post titles and descriptions populate social metadata; custom social images are not yet generated.

## Checks and deployment

```sh
npm run lint
npm test
npm run build:pages
```

`build:pages` exports the static site to `out/`. `.github/workflows/deploy-pages.yml` runs on pushes to `main`, with GitHub Actions selected as the Pages source in repository settings. GitHub supplies the deployment URL and path, so custom-domain settings can also be supported. For local export testing, override `NEXT_PUBLIC_BASE_PATH` and `NEXT_PUBLIC_SITE_URL` as needed.

The published website has no server-side editor, database, or API credentials. Never commit `.env` files, access tokens, private keys, personal contact details, or sensitive draft content. Use a GitHub no-reply address for public commits. Avoid reintroducing history from an older private repository.
