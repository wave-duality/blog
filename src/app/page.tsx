import { getAllPosts } from "@/lib/posts";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import Sidebar from "@/components/layout/Sidebar";
import PostCard from "@/components/posts/PostCard";
import KeyboardNav from "@/components/keyboard/KeyboardNav";
import { SearchOverlay } from "@/components/search/SearchOverlay";

export default function HomePage() {
  const posts = getAllPosts();

  const searchPosts = posts.map((p) => ({
    title: p.title,
    slug: p.slug,
    excerpt: p.excerpt,
  }));

  return (
    <>
      <Sidebar />
      <main className="max-w-[680px] mx-auto xl:ml-[420px]">
        <Header
          name="Owen Zhang"
          bio="19, yale, jane street, prev sea12"
          tagline="writing about things i want to write about"
        />

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

        <Footer
          githubUrl="https://github.com/wave-duality"
        />
      </main>

      <KeyboardNav />
      <SearchOverlay posts={searchPosts} />
    </>
  );
}
