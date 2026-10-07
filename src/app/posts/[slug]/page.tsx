import { getPostBySlug, getAllPosts } from "@/lib/posts";
import { notFound } from "next/navigation";
import BackLink from "@/components/layout/BackLink";
import Article from "@/components/posts/Article";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string }>;
}

export const dynamicParams = false;

export async function generateStaticParams() {
  const posts = getAllPosts();
  return posts.map((post) => ({ slug: post.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) return {};
  return {
    title: `${post.title} - Owen Zhang`,
    description: post.excerpt,
    openGraph: {
      title: post.title,
      description: post.excerpt,
      type: "article",
      authors: [post.author],
    },
    twitter: {
      card: "summary",
      title: post.title,
      description: post.excerpt,
    },
  };
}

export default async function PostPage({ params }: Props) {
  const { slug } = await params;
  const post = getPostBySlug(slug);

  if (!post) notFound();

  return (
    <main className={post.paper ? "max-w-[760px] mx-auto" : "max-w-[680px] mx-auto xl:ml-[420px]"}>
      <BackLink />
      {!post.published && (
        <p className="mb-6 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">Draft preview — visible locally only. Set published: true to include it on the website.</p>
      )}
      <Article {...post} />
    </main>
  );
}
