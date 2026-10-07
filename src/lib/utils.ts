export function formatDate(date: Date | string): string {
  return new Date(date).toISOString().split("T")[0];
}
