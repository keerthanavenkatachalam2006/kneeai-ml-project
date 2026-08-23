"use client";

/**
 * Switch between the two windows of the application, with the matching capture.
 *
 * A product site usually puts an edition comparison here. There is only one application, so
 * the honest version of that control is the two windows it opens: the planning workspace and
 * the robot cell.
 */

import { useState } from "react";

import Icon from "@/components/Icon";
import { MODES } from "@/content/project";

const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export default function ProductSwitcher() {
  const [active, setActive] = useState(0);
  const [failed, setFailed] = useState<Record<string, boolean>>({});
  const mode = MODES[active];

  return (
    <div>
      {/* --- Switch --- */}
      <div className="mb-10 flex flex-wrap items-center gap-3">
        <span className="text-[12.5px] font-medium text-ink-500">Switch between windows:</span>
        <div
          role="tablist"
          aria-label="Application windows"
          className="inline-flex rounded-full border border-ink-200 bg-white p-1 shadow-card"
        >
          {MODES.map((m, i) => (
            <button
              key={m.id}
              role="tab"
              aria-selected={i === active}
              onClick={() => setActive(i)}
              className={`rounded-full px-4 py-1.5 text-[12.5px] font-semibold transition-colors ${
                i === active ? "bg-brand-600 text-white" : "text-ink-600 hover:text-brand-600"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid items-start gap-10 lg:grid-cols-[1.15fr_0.85fr] lg:gap-14">
        {/* --- Capture --- */}
        <figure className="frame">
          <div className="frame-bar">
            <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-rose-400/60" />
            <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-amber-400/60" />
            <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-emerald-400/60" />
            <span className="ml-2 truncate font-mono text-[10.5px] text-white/45">
              morp · {mode.label.toLowerCase()}
            </span>
          </div>
          <div className="relative aspect-[16/9] w-full">
            {failed[mode.shot] ? (
              <div className="flex h-full items-center justify-center px-8 text-center">
                <p className="text-[12px] leading-relaxed text-white/45">
                  Capture not present in this checkout. Run <span className="font-mono">npm run shots</span>{" "}
                  with the API and the workspace running.
                </p>
              </div>
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={`${BASE}/shots/${mode.shot}`}
                alt={`${mode.label}: ${mode.blurb}`}
                className="h-full w-full object-cover object-top"
                onError={() => setFailed((f) => ({ ...f, [mode.shot]: true }))}
              />
            )}
          </div>
        </figure>

        {/* --- Points --- */}
        <div>
          <p className="body mb-7 max-w-reading">{mode.blurb}</p>
          <ul className="space-y-6">
            {mode.points.map((p) => (
              <li key={p.title} className="flex gap-3.5">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-50 text-brand-600">
                  <Icon name="check" className="h-3.5 w-3.5" />
                </span>
                <div>
                  <h3 className="h3 text-[14.5px]">{p.title}</h3>
                  <p className="body mt-1">{p.body}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
