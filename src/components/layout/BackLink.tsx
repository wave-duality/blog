import Link from "next/link";

export default function BackLink() {
  return (
    <Link
      href="/"
      className="inline-block mb-8 text-[#666] text-[0.9rem] hover:opacity-60 transition-opacity"
    >
      ← Back
    </Link>
  );
}
