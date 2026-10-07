"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

export default function KeyboardNav() {
  const router = useRouter();
  const currentIndexRef = useRef(-1);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;

      const posts = document.querySelectorAll<HTMLElement>(".posts article");
      if (posts.length === 0) return;

      if (e.key === "j" || e.key === "J") {
        e.preventDefault();
        currentIndexRef.current = Math.min(
          currentIndexRef.current + 1,
          posts.length - 1
        );
        highlightPost(posts, currentIndexRef.current);
      } else if (e.key === "k" || e.key === "K") {
        e.preventDefault();
        currentIndexRef.current = Math.max(currentIndexRef.current - 1, -1);
        if (currentIndexRef.current >= 0) {
          highlightPost(posts, currentIndexRef.current);
        } else {
          posts.forEach((p) => (p.style.background = ""));
        }
      } else if (e.key === "Enter" && currentIndexRef.current >= 0) {
        const link = posts[currentIndexRef.current]?.querySelector("a");
        if (link) {
          const href = link.getAttribute("href");
          if (href) router.push(href);
        }
      }
    }

    function highlightPost(
      posts: NodeListOf<HTMLElement>,
      index: number
    ) {
      posts.forEach((post, i) => {
        if (i === index) {
          post.style.background = "#f9f9f9";
          post.scrollIntoView({ behavior: "smooth", block: "center" });
          const link = post.querySelector("a");
          if (link) (link as HTMLElement).focus();
        } else {
          post.style.background = "";
        }
      });
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [router]);

  return null;
}
