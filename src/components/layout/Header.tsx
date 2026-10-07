import Link from "next/link";

interface HeaderProps {
  name: string;
  bio: string;
  tagline: string;
}

export default function Header({ name, bio, tagline }: HeaderProps) {
  return (
    <header className="mb-12">
      <h1 className="text-2xl font-semibold mb-2 tracking-tight leading-tight">
        <Link href="/">{name}</Link>
      </h1>
      <p className="text-[#666] text-[0.95rem] mb-4">{bio}</p>
      <p className="text-[#999] text-[0.9rem] leading-relaxed">{tagline}</p>
    </header>
  );
}
