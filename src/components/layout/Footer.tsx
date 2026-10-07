interface FooterProps {
  githubUrl: string;
}

export default function Footer({ githubUrl }: FooterProps) {
  return (
    <footer className="mt-14 pt-8 border-t border-[#eee] text-[0.9rem] text-[#666]">
      <p>
        <a
          href={githubUrl}
          className="hover:opacity-60 transition-opacity"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub
        </a>
      </p>
      <p className="mt-2 text-[0.8rem] text-[#999]">
        Press <kbd className="kbd">/ </kbd> or <kbd className="kbd">⌘K</kbd> to search
      </p>
    </footer>
  );
}
