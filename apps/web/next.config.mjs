/** @type {import('next').NextConfig} */
const API = process.env.KNEEAI_API_URL ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  // Proxy the API through Next so the browser makes same-origin requests. Avoids
  // CORS entirely in dev and keeps the API bound to loopback.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
