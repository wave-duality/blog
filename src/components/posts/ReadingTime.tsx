interface ReadingTimeProps {
  minutes: number;
}

export default function ReadingTime({ minutes }: ReadingTimeProps) {
  return (
    <div className="text-[#999] text-[0.85rem] mt-2">
      {minutes} min read
    </div>
  );
}
