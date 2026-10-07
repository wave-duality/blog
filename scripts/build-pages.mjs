import { spawnSync } from "node:child_process";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "/blog";
const result = spawnSync(process.execPath, ["node_modules/next/dist/bin/next", "build"], {
  stdio: "inherit",
  env: {
    ...process.env,
    GITHUB_PAGES: "true",
    NEXT_PUBLIC_BASE_PATH: basePath,
    NEXT_PUBLIC_SITE_URL: process.env.NEXT_PUBLIC_SITE_URL ?? `https://wave-duality.github.io${basePath}`,
  },
});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
