"use client";

/**
 * Screenshot gallery: thumbnail rail plus a large frame.
 *
 * The images are captures of the running product taken by `scripts/capture-shots.mjs`
 * against a real OAIZIB-CM case — not mockups and not renders of a design file. Each has a
 * caption, because a screenshot of a dense clinical interface is otherwise just texture.
 *
 * Keyboard: left/right arrows move between screens while the tablist has focus. If a capture
 * is missing (a fresh clone has no `public/shots`), the frame degrades to a labelled note
 * instead of a broken image.
 */

import { useCallback, useState } from "react";

import { SHOTS } from "@/content/project";

/** Static export under a subpath needs the prefix added by hand for a plain <img>. */
const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export default function Screens() {
  const [active, setActive] = useState(0);
  const [failed, setFailed] = useState<Record<string, boolean>>({});

  const move = useCallback((delta: number) => {
    setActive((i) => (i + delta + SHOTS.length) % SHOTS.length);
  }, []);

  const shot = SHOTS[active];
  const broken = failed[shot.src];

  return (
    <div className="grid gap-8 lg:grid-cols-[0.28fr_0.72fr] lg:gap-10">
      {/* --- Rail --- */}
      <div
        role="tablist"
        aria-label="Product screens"
        onKeyDown={(e) => {
          if (e.key === "ArrowRight" || e.key === "ArrowDown") move(1);
          if (e.key === "ArrowLeft" || e.key === "ArrowUp") move(-1);
        }}
        className="flex gap-2 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0"
      >
        {SHOTS.map((s, i) => (
          <button
            key={s.src}
            role="tab"
            aria-selected={i === active}
            tabIndex={i === active ? 0 : -1}
            onClick={() => setActive(i)}
            className={`shrink-0 rounded-xl border px-4 py-3 text-left transition-all duration-150 lg:w-full ${
              i === active
                ? "border-brand-300 bg-brand-50 shadow-card"
                : "border-ink-200/70 bg-white hover:border-brand-200"
            }`}
          >
            <span className="block font-mono text-[10px] uppercase tracking-[0.14em] text-ink-400">
              {String(i + 1).padStart(2, "0")}
            </span>
            <span
              className={`mt-1 block whitespace-nowrap text-[13px] font-semibold lg:whitespace-normal ${
                i === active ? "text-brand-700" : "text-ink-800"
              }`}
            >
              {s.label}
            </span>
          </button>
        ))}
      </div>

      {/* --- Frame --- */}
      <figure>
        <div className="frame">
          <div className="frame-bar">
            <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-rose-400/60" />
            <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-amber-400/60" />
            <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-emerald-400/60" />
            <span className="ml-2 truncate font-mono text-[10.5px] text-white/45">
              morp · {shot.label.toLowerCase()} · case oaizib_405
            </span>
            <span className="ml-auto shrink-0 font-mono text-[10.5px] text-white/30">
              {active + 1}/{SHOTS.length}
            </span>
          </div>

          <div className="relative aspect-[16/9] w-full">
            {broken ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 px-8 text-center">
                <p className="text-[13px] text-white/70">{shot.label}</p>
                <p className="max-w-md text-[11.5px] leading-relaxed text-white/40">
                  Capture not present in this checkout. With the API and the workspace running,{" "}
                  <span className="font-mono">npm run shots</span> regenerates{" "}
                  <span className="font-mono">public/shots/{shot.src}</span>.
                </p>
              </div>
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={`${BASE}/shots/${shot.src}`}
                alt={shot.caption}
                loading={active === 0 ? "eager" : "lazy"}
                className="h-full w-full object-cover object-top"
                onError={() => setFailed((f) => ({ ...f, [shot.src]: true }))}
              />
            )}
          </div>
        </div>

        <figcaption className="mt-5 flex flex-wrap items-start justify-between gap-5">
          <p className="body max-w-reading">{shot.caption}</p>
          <div className="flex shrink-0 gap-2">
            <button
              type="button"
              onClick={() => move(-1)}
              aria-label="Previous screen"
              className="flex h-9 w-9 items-center justify-center rounded-full border border-ink-200 bg-white text-ink-500 transition-colors hover:border-brand-300 hover:text-brand-600"
            >
              <span aria-hidden>←</span>
            </button>
            <button
              type="button"
              onClick={() => move(1)}
              aria-label="Next screen"
              className="flex h-9 w-9 items-center justify-center rounded-full border border-ink-200 bg-white text-ink-500 transition-colors hover:border-brand-300 hover:text-brand-600"
            >
              <span aria-hidden>→</span>
            </button>
          </div>
        </figcaption>
      </figure>
    </div>
  );
}
