import PostContent from "./PostContent";
import ArticleContents from "./ArticleContents";
import ReadingTime from "./ReadingTime";
import type { ArticleSection } from "@/lib/markdown";

interface ArticleProps {
  title: string;
  date: string;
  content: string;
  paper: boolean;
  author: string;
  abstract: string;
  tokenCount: number;
  sections: ArticleSection[];
}

export default function Article(post: ArticleProps) {
  return (
    <article className={post.paper ? "paper-article" : undefined}>
      <header className="article-header">
        <h1 className="text-[1.8rem] font-semibold mb-2 tracking-tight leading-tight">{post.title}</h1>
        <div className="article-meta">
        {post.paper && <span>{post.author}</span>}
        {post.date && <time dateTime={/^\d{4}-\d{2}-\d{2}$/.test(post.date) ? post.date : undefined} className="block text-[0.8rem] text-[#777] mb-1 tabular-nums">{post.date}</time>}
        <ReadingTime tokens={post.tokenCount} />
        </div>
      </header>
      <ArticleContents title={post.title} sections={post.sections} />
      {post.paper && post.abstract && (
        <section className="paper-abstract" aria-label="Abstract">
          <h2>Abstract</h2>
          <p>{post.abstract}</p>
        </section>
      )}
      <PostContent html={post.content} />
    </article>
  );
}
