interface PostContentProps {
  html: string;
}

export default function PostContent({ html }: PostContentProps) {
  return (
    <div
      className="post-content"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
