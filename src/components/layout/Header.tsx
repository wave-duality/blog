import Link from "next/link";

interface HeaderProps {
  name: string;
  bio: string;
}

export default function Header({ name, bio }: HeaderProps) {
  return (
    <header className="mb-12">
      <h1 className="text-2xl font-semibold mb-2 tracking-tight leading-tight">
        <Link href="/">{name}</Link>
      </h1>
      <p className="text-[#666] text-[0.95rem] mb-4">{bio}</p>
    </header>
  );
}
