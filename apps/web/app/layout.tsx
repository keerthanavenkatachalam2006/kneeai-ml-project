import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "MORP — Morphometry · Reconstruction · Planning",
    template: "%s · MORP",
  },
  description:
    "AI knee segmentation, 3D reconstruction, measurement, implant matching, surgical planning and robotic simulation. Research prototype, not a medical device.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* Persistent and unconditional. Every view sits below it. */}
        <div
          role="note"
          className="flex h-7 items-center justify-center border-b border-amber-700/50
                     bg-amber-950/40 px-4 text-center text-[10.5px] tracking-wide text-amber-200/90"
        >
          <strong className="font-semibold">RESEARCH / SIMULATION PROTOTYPE</strong>
          <span className="mx-1.5 text-amber-700">·</span>
          not a medical device, not clinically validated, not for patient care
          <span className="mx-1.5 text-amber-700">·</span>
          <strong className="font-semibold">SIMULATION ONLY — NO HARDWARE CONTROL</strong>
        </div>
        {children}
      </body>
    </html>
  );
}
