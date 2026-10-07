import { getAllPosts } from "@/lib/posts";
import BackLink from "@/components/layout/BackLink";
import PostCard from "@/components/posts/PostCard";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Archive - Owen Zhang",
};

export default function ArchivePage() {
  const posts = getAllPosts();

  return (
    <main className="max-w-[680px] mx-auto xl:ml-[420px]">
      <BackLink />

      <header className="mb-12">
        <h1 className="text-2xl font-semibold tracking-tight leading-tight">
          Archive
        </h1>
      </header>

      <section className="posts">
        {posts.map((post) => (
          <PostCard
            key={post.slug}
            title={post.title}
            slug={post.slug}
            excerpt={post.excerpt}
            publishedAt={post.date}
          />
        ))}
      </section>
    </main>
  );
}
