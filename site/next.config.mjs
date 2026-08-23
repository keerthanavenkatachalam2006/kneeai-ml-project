/** @type {import('next').NextConfig} */

/**
 * The showcase site is fully static: no API, no server, no patient data. `output:
 * "export"` writes plain HTML to `out/`, which can be dropped on any static host
 * (Pages, S3, Netlify) with nothing to run and nothing to secure.
 *
 * Set `NEXT_PUBLIC_BASE_PATH=/repo-name` when deploying under a subpath, e.g. GitHub
 * Pages project sites.
 */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
  basePath,
  images: { unoptimized: true },
};

export default nextConfig;
