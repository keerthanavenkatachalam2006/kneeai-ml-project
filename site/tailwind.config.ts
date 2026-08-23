import type { Config } from "tailwindcss";

/**
 * Showcase-site palette.
 *
 * The product is a dark clinical workspace; this site is deliberately the opposite —
 * light, airy, and built around framed screenshots, in the register commercial
 * orthopaedic-planning sites use. Dark product captures on a light page also read far
 * better than dark-on-dark, which is the practical reason as well as the stylistic one.
 *
 * `brand` is a medical blue for interaction and emphasis. `accent` is the product's teal,
 * used sparingly so the two feel related. `amber` stays reserved for caveats and
 * regulatory copy and is never decorative.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./content/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: {
          DEFAULT: "#ffffff",
          soft: "#f4f9fb",
          tint: "#e9f3f8",
        },
        ink: {
          900: "#062230",
          800: "#0b2f41",
          700: "#134256",
          600: "#265a70",
          500: "#4a7387",
          400: "#7797a7",
          300: "#a9c1cd",
          200: "#d3e2ea",
        },
        brand: {
          50: "#eaf6fc",
          100: "#cfeaf7",
          200: "#a7d9ef",
          300: "#6fc1e4",
          400: "#33a4d3",
          500: "#1289ba",
          600: "#0a6f9b",
          700: "#08587c",
          900: "#053a53",
        },
        accent: {
          400: "#2dc4b6",
          500: "#12a89a",
          600: "#0b8579",
        },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Inter",
          "Helvetica Neue",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      maxWidth: {
        reading: "66ch",
      },
      boxShadow: {
        card: "0 1px 2px rgba(6,34,48,0.04), 0 8px 24px -12px rgba(6,34,48,0.12)",
        lift: "0 2px 4px rgba(6,34,48,0.05), 0 24px 48px -20px rgba(6,34,48,0.25)",
        frame: "0 4px 12px rgba(6,34,48,0.08), 0 40px 80px -32px rgba(6,34,48,0.45)",
      },
      borderRadius: {
        card: "16px",
        frame: "20px",
      },
      keyframes: {
        rise: {
          from: { opacity: "0", transform: "translateY(16px)" },
          to: { opacity: "1", transform: "none" },
        },
      },
      animation: {
        rise: "rise 0.7s cubic-bezier(0.22, 1, 0.36, 1) both",
      },
    },
  },
  plugins: [],
};

export default config;
