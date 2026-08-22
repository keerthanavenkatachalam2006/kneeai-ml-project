import type { Config } from "tailwindcss";

// Tailwind's own slate scale carries the chrome, and colour is reserved for
// meaning: sky for interaction, emerald for pass, amber for caution, rose for
// failure. Nothing is coloured decoratively, so a coloured element always tells
// the reader something.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
