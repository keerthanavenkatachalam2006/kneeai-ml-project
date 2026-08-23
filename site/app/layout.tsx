import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MORP — patient-specific knee planning from a single MR scan",
  description:
    "MORP turns one knee MR volume into a 3D model, twenty-six morphometric measurements, an osteoarthritis assessment, ranked implant candidates and a resection plan — each with its reasoning attached. Research prototype, not a medical device.",
  keywords: [
    "orthopaedic planning",
    "knee arthroplasty planning",
    "MRI segmentation",
    "implant templating",
    "morphometry",
    "explainable AI",
  ],
  openGraph: {
    title: "MORP — patient-specific knee planning from a single MR scan",
    description:
      "Automatic segmentation, 3D reconstruction, morphometry, implant matching and a simulated robot cell. Every number carries its reasoning.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* Unconditional and above the navigation. A polished marketing page is exactly
            where a research prototype is most likely to be mistaken for a cleared device,
            so the status is the first thing on the page rather than a footnote. */}
        <div
          role="note"
          className="flex flex-wrap items-center justify-center gap-x-2 gap-y-0.5 border-b border-amber-300/60 bg-amber-50 px-4 py-1.5 text-center text-[11px] text-amber-900"
        >
          <strong className="font-semibold uppercase tracking-[0.1em]">Research prototype</strong>
          <span className="text-amber-500">·</span>
          <span>not a medical device, not clinically validated, not for patient care</span>
        </div>
        {children}
      </body>
    </html>
  );
}
