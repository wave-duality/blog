import PostContent from "./PostContent";
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
      {post.sections.length > 0 && (
        <nav className="article-sections" aria-label="Article sections">
          <span className="article-sections-label">On this page</span>
          <ul>{post.sections.map(section => (
            <li key={section.id}><a href={`#${section.id}`}>{section.title}</a></li>
          ))}</ul>
        </nav>
      )}
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
