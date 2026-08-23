/**
 * The small icon set, inline.
 *
 * Six glyphs do not justify an icon dependency, and inlining them means they inherit
 * `currentColor` and never arrive after the text they label. All are drawn on a 24-unit
 * grid with a 1.5 stroke so they sit together evenly.
 */

export type IconName =
  | "scan"
  | "cube"
  | "ruler"
  | "implant"
  | "chart"
  | "robot"
  | "check"
  | "chevron"
  | "arrow";

const PATHS: Record<IconName, React.ReactNode> = {
  // A volume being swept by a scan plane.
  scan: (
    <>
      <path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16" />
      <path d="M3 12h18" />
    </>
  ),
  // A reconstructed solid.
  cube: (
    <>
      <path d="M12 3.5 20 8v8l-8 4.5L4 16V8l8-4.5Z" />
      <path d="M4 8l8 4.5L20 8M12 12.5V20.5" />
    </>
  ),
  // Calipers.
  ruler: (
    <>
      <path d="M5 4v11a7 7 0 0 0 14 0V4" />
      <path d="M5 8h3M16 8h3M12 15v5" />
    </>
  ),
  // A femoral component sitting on a resected surface.
  implant: (
    <>
      <path d="M4 15c0-4.4 3.6-8 8-8s8 3.6 8 8" />
      <path d="M4 15v2.5M20 15v2.5M3 20.5h18" />
    </>
  ),
  // Measured distribution.
  chart: (
    <>
      <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
    </>
  ),
  // Gantry with a tool.
  robot: (
    <>
      <path d="M3 5h18M7 5v5.5a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V5" />
      <path d="M12 12.5V17M9 20h6" />
    </>
  ),
  check: <path d="M4.5 12.5l4.5 4.5L19.5 6.5" />,
  chevron: <path d="M6 9.5l6 6 6-6" />,
  arrow: <path d="M4 12h15m-5.5-5.5L19.5 12l-6 5.5" />,
};

export default function Icon({
  name,
  className = "h-5 w-5",
}: {
  name: IconName;
  className?: string;
}) {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      {PATHS[name]}
    </svg>
  );
}
