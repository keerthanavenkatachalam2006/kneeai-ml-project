"use client";

/**
 * The provenance badge.
 *
 * Contract rule 2: `source` must be surfaced in every view and every export, and
 * phantom or ground-truth geometry must never be presented as an AI result.
 *
 * Two decisions enforce that. `source` is a required prop, so a caller cannot
 * render the badge without deciding what it says. And an unrecognised value renders
 * as a loud `UNKNOWN SOURCE` in failure colours rather than defaulting to something
 * reassuring — unreadable provenance is worse than phantom data, not better.
 */

import type { SegmentationSource } from "@/lib/types";

const STYLES: Record<SegmentationSource, { label: string; className: string; title: string }> = {
  ai_prediction: {
    label: "AI prediction",
    className: "border-sky-500 bg-sky-500/15 text-sky-300",
    title: "Segmentation predicted by the trained model at inference time.",
  },
  ground_truth: {
    label: "Ground truth",
    className: "border-emerald-600 bg-emerald-500/15 text-emerald-300",
    title:
      "Expert reference annotation loaded from disk. NOT a model output, and must not be read as model performance.",
  },
  phantom: {
    label: "Phantom",
    className: "border-amber-600 bg-amber-500/15 text-amber-300",
    title:
      "Synthetic phantom knee with analytic ground truth. Not a patient and not a model output.",
  },
  mock: {
    label: "Mock",
    className: "border-slate-600 bg-slate-700/40 text-slate-300",
    title: "Deterministic fixture for interface development. Carries no anatomical meaning.",
  },
};

export default function SourceBadge({
  source,
  size = "md",
}: {
  source: SegmentationSource | string;
  size?: "sm" | "md";
}) {
  const style = STYLES[source as SegmentationSource] ?? {
    label: "Unknown source",
    className: "border-rose-600 bg-rose-500/20 text-rose-300",
    title:
      "The provenance of this geometry could not be determined. Do not rely on anything shown.",
  };

  return (
    <span
      title={style.title}
      aria-label={`Data source: ${style.label}`}
      className={[
        "chip",
        style.className,
        size === "md" ? "px-2 py-1 text-[10.5px]" : "",
      ].join(" ")}
    >
      {style.label}
    </span>
  );
}
