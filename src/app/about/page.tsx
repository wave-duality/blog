import { getPage } from "@/lib/posts";
import BackLink from "@/components/layout/BackLink";
import PostContent from "@/components/posts/PostContent";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About - Owen Zhang",
};

export default function AboutPage() {
  const page = getPage("about");

  return (
    <main className="max-w-[680px] mx-auto xl:ml-[420px]">
      <BackLink />

      <article>
        <div className="mb-8">
          <h1 className="text-[1.8rem] font-semibold mb-2 tracking-tight leading-tight">
            {page?.title ?? "About"}
          </h1>
        </div>

        {page ? (
          <PostContent html={page.content} />
        ) : (
          <p className="text-[#666]">About page content coming soon.</p>
        )}
      </article>
    </main>
  );
}
