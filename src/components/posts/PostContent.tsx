"use client";

import { useEffect, useRef } from "react";

interface PostContentProps {
  html: string;
}

export default function PostContent({ html }: PostContentProps) {
  const content = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = content.current;
    if (!root) return;
    const frames = () => Array.from(root.querySelectorAll<HTMLIFrameElement>("iframe.figure-embed"));
    const measure = () => frames().forEach(frame => {
      frame.contentWindow?.postMessage({ type: "catan-figure-measure" }, "*");
    });
    const resize = (event: MessageEvent) => {
      if (event.data?.type !== "catan-figure-resize") return;
      const height = event.data.height;
      if (typeof height !== "number" || !Number.isFinite(height) || height < 100 || height > 10000) return;
      // Sandboxed frames have opaque origins; authenticate by their Window identity.
      const frame = frames().find(item => item.contentWindow === event.source);
      if (frame) frame.style.height = `${Math.ceil(height)}px`;
    };
    window.addEventListener("message", resize);
    root.addEventListener("load", measure, true);
    measure();
    return () => {
      window.removeEventListener("message", resize);
      root.removeEventListener("load", measure, true);
    };
  }, [html]);

  return <div ref={content} className="post-content" dangerouslySetInnerHTML={{ __html: html }} />;
}
