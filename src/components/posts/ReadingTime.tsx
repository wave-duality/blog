interface ReadingTimeProps {
  minutes?: number;
  tokens: number;
}

export default function ReadingTime({ minutes, tokens }: ReadingTimeProps) {
  return (
    <div className="text-[#999] text-[0.85rem] mt-2">
      {minutes !== undefined && <>{minutes} min read, </>}
      <span title="Approximate token count, estimated from article text and math source. Counts vary by tokenizer.">
        ~{tokens.toLocaleString("en-US")} tokens
      </span>
    </div>
  );
}
