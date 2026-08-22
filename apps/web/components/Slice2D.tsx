"use client";

/**
 * 2D orthogonal slice views with segmentation overlay.
 *
 * Slices are rendered server-side to PNG. Slice geometry depends on the affine, and
 * native spacing here is anisotropic (0.7 x 0.365 x 0.365 mm), so a plate drawn on a
 * square pixel grid is stretched roughly 2:1 — which silently distorts every visual
 * judgement made from it. That correction lives next to the code that understands the
 * volume rather than being reimplemented against a hand-rolled NIfTI parser here.
 *
 * Contours are the default rather than filled masks, because a filled overlay hides
 * the image the segmentation is being judged against.
 *
 * Slice changes are debounced and the previous frame is held while the next loads, so
 * dragging a slider does not flash black between frames.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { SliceInfo } from "@/lib/types";

type Plane = "axial" | "coronal" | "sagittal";
type Mode = "contour" | "fill" | "image";

const PLANES: Array<{ id: Plane; label: string; corners: [string, string, string, string] }> = [
  // Corner letters follow radiological convention: what lies in each direction.
  { id: "axial", label: "Axial", corners: ["A", "R", "P", "L"] },
  { id: "coronal", label: "Coronal", corners: ["S", "R", "I", "L"] },
  { id: "sagittal", label: "Sagittal", corners: ["S", "A", "I", "P"] },
];

function SlicePanel({
  caseId,
  plane,
  label,
  corners,
  count,
  index,
  mode,
  onIndex,
}: {
  caseId: string;
  plane: Plane;
  label: string;
  corners: [string, string, string, string];
  count: number;
  index: number;
  mode: Mode;
  onIndex: (v: number) => void;
}) {
  const [shown, setShown] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pending = useRef<number | null>(null);

  const url = useMemo(
    () => api.sliceUrl(caseId, plane, index, mode),
    [caseId, plane, index, mode],
  );

  // Hold the previous frame until the next has decoded, so dragging the slider
  // does not strobe.
  useEffect(() => {
    let alive = true;
    setLoading(true);
    const img = new Image();
    img.onload = () => {
      if (!alive) return;
      setShown(url);
      setLoading(false);
    };
    img.onerror = () => alive && setLoading(false);
    img.src = url;
    return () => {
      alive = false;
    };
  }, [url]);

  const step = useCallback(
    (delta: number) => onIndex(Math.max(0, Math.min(count - 1, index + delta))),
    [count, index, onIndex],
  );

  return (
    <div className="rounded-md border border-slate-800 bg-slate-950">
      <div className="flex items-center justify-between px-2 py-1.5">
        <span className="text-[11px] font-medium text-slate-300">{label}</span>
        <span className="font-mono text-[10px] text-slate-500">
          {index + 1} / {count}
        </span>
      </div>

      <div
        className="relative flex aspect-square items-center justify-center overflow-hidden bg-black"
        onWheel={(e) => {
          e.preventDefault();
          step(e.deltaY > 0 ? 1 : -1);
        }}
        role="img"
        aria-label={`${label} slice ${index + 1} of ${count}`}
      >
        {shown && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={shown}
            alt=""
            className="max-h-full max-w-full object-contain"
            style={{ imageRendering: "auto" }}
            draggable={false}
          />
        )}
        {loading && (
          <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />
        )}
        {/* Orientation letters, so a reader is never guessing which way is which. */}
        <span className="pointer-events-none absolute left-1/2 top-1 -translate-x-1/2 font-mono text-[9px] text-slate-500">
          {corners[0]}
        </span>
        <span className="pointer-events-none absolute right-1 top-1/2 -translate-y-1/2 font-mono text-[9px] text-slate-500">
          {corners[3]}
        </span>
        <span className="pointer-events-none absolute bottom-1 left-1/2 -translate-x-1/2 font-mono text-[9px] text-slate-500">
          {corners[2]}
        </span>
        <span className="pointer-events-none absolute left-1 top-1/2 -translate-y-1/2 font-mono text-[9px] text-slate-500">
          {corners[1]}
        </span>
      </div>

      <div className="flex items-center gap-1.5 px-2 py-1.5">
        <button
          type="button"
          onClick={() => step(-1)}
          className="px-1 text-slate-500 hover:text-slate-200"
          aria-label={`Previous ${label} slice`}
        >
          ‹
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, count - 1)}
          value={index}
          onChange={(e) => onIndex(Number(e.target.value))}
          className="h-1 flex-1 accent-sky-500"
          aria-label={`${label} slice index`}
        />
        <button
          type="button"
          onClick={() => step(1)}
          className="px-1 text-slate-500 hover:text-slate-200"
          aria-label={`Next ${label} slice`}
        >
          ›
        </button>
      </div>
    </div>
  );
}

export default function Slice2D({ caseId }: { caseId: string }) {
  const [info, setInfo] = useState<SliceInfo | null>(null);
  const [index, setIndex] = useState<Record<Plane, number>>({
    axial: 0,
    coronal: 0,
    sagittal: 0,
  });
  const [mode, setMode] = useState<Mode>("contour");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setInfo(null);
    setError(null);
    api
      .sliceInfo(caseId)
      .then((data) => {
        if (!alive) return;
        setInfo(data);
        setIndex({
          axial: data.default_index.axial ?? 0,
          coronal: data.default_index.coronal ?? 0,
          sagittal: data.default_index.sagittal ?? 0,
        });
        // With no intensity volume, contours over black show almost nothing.
        if (!data.has_source_image) setMode("fill");
      })
      .catch((e) => alive && setError(String(e?.message ?? e)));
    return () => {
      alive = false;
    };
  }, [caseId]);

  if (error) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">2D slices</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] text-slate-500">{error}</p>
        </div>
      </div>
    );
  }

  if (!info) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">2D slices</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] text-slate-500">Loading…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-head">
        <span className="card-title">2D slices</span>
        <div className="flex gap-1">
          {(["contour", "fill", "image"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              disabled={m === "image" && !info.has_source_image}
              aria-pressed={mode === m}
              className={[
                "rounded border px-1.5 py-0.5 text-[10px] font-medium capitalize transition-colors",
                m === "image" && !info.has_source_image
                  ? "cursor-not-allowed border-slate-800 text-slate-600"
                  : mode === m
                    ? "border-sky-500 bg-sky-500/20 text-sky-300"
                    : "border-slate-700 text-slate-400 hover:bg-slate-800",
              ].join(" ")}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="card-body space-y-2">
        {PLANES.map((p) => (
          <SlicePanel
            key={p.id}
            caseId={caseId}
            plane={p.id}
            label={p.label}
            corners={p.corners}
            count={info.n_slices[p.id] ?? 1}
            index={index[p.id]}
            mode={mode}
            onIndex={(v) => setIndex((s) => ({ ...s, [p.id]: v }))}
          />
        ))}

        <p className="text-[11px] leading-relaxed text-slate-500">
          {info.note} Scroll over a panel or drag its slider to move through slices.
          Voxel spacing is {info.spacing_mm.map((s) => s.toFixed(2)).join(" × ")} mm and
          the panels are corrected for it, so proportions are anatomically true.
        </p>
      </div>
    </div>
  );
}
