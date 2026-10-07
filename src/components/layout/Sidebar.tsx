"use client";

import Image from "next/image";
import Link from "next/link";
import dynamic from "next/dynamic";
import { withBasePath } from "@/lib/site";

const Antiprism = dynamic(
  () => import("@/components/visualizations/Antiprism"),
  { ssr: false }
);

export default function Sidebar() {
  return (
    <div className="hidden xl:block">
      {/* Profile photo */}
      <div className="fixed left-8 top-8 w-[320px] h-[320px] z-10 p-4 bg-white shadow-[0_1px_3px_rgba(0,0,0,0.08)] overflow-hidden">
        <div className="h-full w-full overflow-hidden">
          <Image
            src={withBasePath("/me.png")}
            alt="Owen Zhang"
            width={288}
            height={288}
            className="block h-full w-full object-cover object-[18%_48%] scale-[2.2] origin-[18%_48%]"
            priority
          />
        </div>
      </div>
      <div className="fixed left-8 z-10 text-[0.85rem] text-[#999]" style={{ top: "calc(2rem + 320px + 0.5rem)" }}>
        <Link href="/about" className="text-[#999] no-underline hover:text-black transition-colors">
          who is this?
        </Link>
      </div>

      {/* Anticube visualization */}
      <div className="fixed left-8 w-[320px] h-[320px] z-10 cursor-grab active:cursor-grabbing" style={{ top: "calc(2rem + 320px + 2rem + 1rem)" }}>
        <Antiprism />
      </div>
      <div className="fixed left-8 z-10 text-[0.85rem] text-[#999]" style={{ top: "calc(2rem + 320px + 2rem + 1rem + 320px + 0.5rem)" }}>
        <Link href="/anticube" className="text-[#999] no-underline hover:text-black transition-colors">
          what is this?
        </Link>
      </div>
    </div>
  );
}
