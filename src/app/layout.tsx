import type { Metadata } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";
import { SearchProvider } from "@/components/search/SearchProvider";
import { siteUrl } from "@/lib/site";

export const metadata: Metadata = {
  metadataBase: new URL(`${siteUrl}/`),
  title: "Owen Zhang",
  description: "Personal blog by Owen Zhang",
  openGraph: {
    title: "Owen Zhang",
    type: "website",
  },
  alternates: {
    types: {
      "application/rss+xml": `${siteUrl}/feed.xml`,
    },
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <SearchProvider>{children}</SearchProvider>
      </body>
    </html>
  );
}
