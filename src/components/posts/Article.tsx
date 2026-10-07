import PostContent from "./PostContent";
import ReadingTime from "./ReadingTime";
import { calculateReadingTime } from "@/lib/reading-time";

interface ArticleProps {
  title: string;
  date: string;
  content: string;
  paper: boolean;
  author: string;
  abstract: string;
}

export default function Article(post: ArticleProps) {
  return (
    <article className={post.paper ? "paper-article" : undefined}>
      <header className={post.paper ? "paper-header" : "mb-8"}>
        <h1 className="text-[1.8rem] font-semibold mb-2 tracking-tight leading-tight">{post.title}</h1>
        {post.paper && <p className="paper-author">{post.author}</p>}
        {post.date && <time dateTime={/^\d{4}-\d{2}-\d{2}$/.test(post.date) ? post.date : undefined} className="block text-[0.8rem] text-[#777] mb-1 tabular-nums">{post.date}</time>}
        <ReadingTime minutes={calculateReadingTime(post.content)} />
      </header>
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
