import Link from "next/link";
import { formatDate } from "@/lib/utils";

interface PostCardProps {
  title: string;
  slug: string;
  excerpt: string;
  publishedAt: Date | string;
}

export default function PostCard({
  title,
  slug,
  excerpt,
  publishedAt,
}: PostCardProps) {
  return (
    <article className="mb-8">
      <time className="block text-[0.8rem] text-[#999] mb-1 tabular-nums tracking-wide">
        {formatDate(publishedAt)}
      </time>
      <h2 className="text-[1.15rem] font-medium mb-1 tracking-tight leading-snug">
        <Link href={`/posts/${slug}`} className="hover:opacity-60 transition-opacity">
          {title}
        </Link>
      </h2>
      <p className="text-[#555] text-[0.95rem] leading-relaxed">{excerpt}</p>
    </article>
  );
}
