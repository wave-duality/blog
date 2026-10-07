import fs from "node:fs";
import path from "node:path";

const args = process.argv.slice(2);
const paper = args.includes("--paper");
const title = args.filter((arg) => arg !== "--paper").join(" ").trim();
if (!title) {
  console.error('Usage: npm run new:post -- "Article title" [--paper]');
  process.exit(1);
}
const slug = title.normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
  .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
if (!slug) {
  console.error("Include at least one English letter or number in the title.");
  process.exit(1);
}
const filename = path.join("content/posts", `${slug}.md`);
fs.mkdirSync(path.dirname(filename), { recursive: true });
const date = new Date().toISOString().slice(0, 10);
const content = `---\ntitle: ${JSON.stringify(title)}\ndate: "${date}"\nexcerpt: ""\npublished: false\npaper: ${paper}\n${paper ? 'author: "Owen Zhang"\nabstract: ""\n' : ""}---\n\nWrite your article here.\n`;
try {
  fs.writeFileSync(filename, content, { flag: "wx" });
} catch (error) {
  if (error.code === "EEXIST") {
    console.error(`An article already exists at ${filename}; it was not overwritten.`);
    process.exit(1);
  }
  throw error;
}
console.log(`Created ${filename}\nPreview with npm run dev at http://localhost:3000/posts/${slug}/\nSet published: true when ready, then commit and push.\nDraft files committed to a public repository are readable on GitHub.`);
