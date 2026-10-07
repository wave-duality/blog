"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useSearch } from "./SearchProvider";

interface SearchPost {
  title: string;
  slug: string;
  excerpt: string;
}

interface SearchOverlayProps {
  posts: SearchPost[];
}

export function SearchOverlay({ posts }: SearchOverlayProps) {
  const { isOpen, open, close } = useSearch();
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const filtered = query.trim()
    ? posts.filter(
        (p) =>
          p.title.toLowerCase().includes(query.toLowerCase()) ||
          p.excerpt.toLowerCase().includes(query.toLowerCase())
      )
    : [];

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        open();
      } else if (
        e.key === "/" &&
        !isOpen &&
        document.activeElement?.tagName !== "INPUT" &&
        document.activeElement?.tagName !== "TEXTAREA"
      ) {
        e.preventDefault();
        open();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, open]);

  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[1000] bg-white/95 p-8"
      style={{ backdropFilter: "blur(4px)" }}
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <input
        ref={inputRef}
        type="text"
        placeholder="Search posts..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") close();
        }}
        className="w-full max-w-[600px] mx-auto block text-[1.2rem] p-4 border border-[#ddd] bg-white font-[inherit]"
      />

      <div className="max-w-[600px] mx-auto mt-4">
        {query.trim() && filtered.length === 0 && (
          <p className="text-[#999] text-center mt-8">No results found</p>
        )}
        {filtered.map((post) => (
          <button
            key={post.slug}
            onClick={() => {
              close();
              router.push(`/posts/${post.slug}`);
            }}
            className="block w-full text-left p-4 mb-2 border border-[#eee] hover:bg-[#f9f9f9] hover:border-[#ddd] transition-colors"
          >
            <div className="font-medium mb-1">{post.title}</div>
            <div className="text-[0.85rem] text-[#666]">{post.excerpt}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
