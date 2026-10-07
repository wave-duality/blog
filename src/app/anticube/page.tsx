import BackLink from "@/components/layout/BackLink";
import Script from "next/script";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Anticube - Owen Zhang",
};

export default function AnticubePage() {
  return (
    <>
      <Script
        id="mathjax-config"
        strategy="beforeInteractive"
        dangerouslySetInnerHTML={{
          __html: `
            window.MathJax = {
              tex: {
                inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
                displayMath: [['\\\\[', '\\\\]']]
              }
            };
          `,
        }}
      />
      <Script
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        strategy="afterInteractive"
      />

      <main className="max-w-[680px] mx-auto xl:ml-[420px]">
        <BackLink />

        <article>
          <div className="mb-8">
            <h1 className="text-[1.8rem] font-semibold mb-2 tracking-tight leading-tight">
              Anticube
            </h1>
            <p className="text-[#666]">natural compromises</p>
          </div>

          <div className="post-content">
            <p>
              This is an anticube, or more formally &ldquo;square antiprism.&rdquo; Think
              of it as a cube but with a pair of opposite faces twisted away from
              each other. I find it cool because it&apos;s the optimal structure for
              eight atoms in a canonical potential energy landscape.
            </p>

            <p>
              Put more formally, consider a system of {"\\(N = 8\\)"} atoms
              interacting via the Morse potential:
            </p>

            <p style={{ textAlign: "center", margin: "1.5rem 0" }}>
              {"\\[V(r) = D\\left[1 - e^{-a(r - r_e)}\\right]^2\\]"}
            </p>

            <p>
              where {"\\(D\\)"} is the well depth, {"\\(a\\)"} controls the
              width, and {"\\(r_e\\)"} is the equilibrium distance. The total
              potential energy of the system is:
            </p>

            <p style={{ textAlign: "center", margin: "1.5rem 0" }}>
              {
                "\\[E = \\sum_{i < j} V(\\|\\mathbf{r}_i - \\mathbf{r}_j\\|)\\]"
              }
            </p>

            <p>
              For {"\\(N = 8\\)"}, the square antiprism configuration minimizes{" "}
              {"\\(E\\)"} over all possible arrangements of atomic positions{" "}
              {
                "\\(\\{\\mathbf{r}_1, \\ldots, \\mathbf{r}_8\\} \\in \\mathbb{R}^3\\)"
              }
              . This result holds for standard Morse parameters (e.g.,{" "}
              {"\\(D = 1\\)"}, {"\\(a = 1\\)"}, {"\\(r_e = 1\\)"}) and can be
              verified through Monte Carlo simulated annealing or other global
              optimization methods.
            </p>

            <p>
              Here&apos;s another cool way to think about it. If we give points access
              to two dimensions, then the equilateral triangle is the lowest
              energy state for three points—all pairwise distances can be set to{" "}
              {"\\(r_e\\)"}, where {"\\(V(r_e) = 0\\)"} (the minimum of the
              Morse potential). Similarly, if we allow three dimensions, then the
              regular tetrahedron achieves {"\\(V(r_e) = 0\\)"} for all pairs of
              four points. In general, {"\\(n\\)"} dimensions can support up to{" "}
              {"\\(n+1\\)"} points (the {"\\(n\\)"}-simplex) with all distances
              at the optimal {"\\(r_e\\)"}. But if we try to place{" "}
              {"\\(>n+1\\)"} points into {"\\(n\\)"}-dimensional space, then
              compromises must be made—not all pairwise distances can be exactly{" "}
              {"\\(r_e\\)"}. There is not enough dimension to optimize every pair
              locally.
            </p>

            <p>The anticube is the result of some global set of compromises.</p>
          </div>
        </article>
      </main>
    </>
  );
}
