import type { Config } from "tailwindcss";

/**
 * Surgical-planning palette.
 *
 * `clinic` is a deep desaturated teal used for every surface, so the chrome recedes and
 * the anatomy in the viewer is the brightest thing on screen. `teal` carries interaction
 * and selection.
 *
 * Colour still means something: teal for interactive/selected, emerald for pass, amber
 * for caution, rose for failure. Nothing is coloured decoratively, so a coloured element
 * always tells the reader something.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        clinic: {
          950: "#03202b",
          900: "#052e3c",
          850: "#073a4b",
          800: "#0a4557",
          700: "#0f5468",
          600: "#166b80",
          500: "#1d8298",
        },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Inter",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        panel: "0 8px 32px -8px rgba(0, 0, 0, 0.55)",
        float: "0 12px 40px -12px rgba(0, 0, 0, 0.7)",
      },
      borderRadius: {
        panel: "14px",
      },
    },
  },
  plugins: [],
};

export default config;
