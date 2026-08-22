"use client";

/**
 * Implant recommendation, with the reasoning kept visible.
 *
 * This is the panel that distinguishes the system from a 3D viewer, so the runner-up
 * and the reason it lost carry the same weight as the winner. A recommendation a
 * surgeon cannot interrogate is one they must either take on faith or discard, and
 * PRD §2.2 requires the evidence to travel with the number.
 *
 * The catalog disclaimer renders unconditionally: this is reference geometry, not
 * manufacturer data, and the fit numbers are only as meaningful as the shapes behind
 * them.
 */

import { useState } from "react";
import { fmt } from "@/lib/api";
import type { ImplantCandidate, ImplantResult } from "@/lib/types";

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function Metrics({ c }: { c: ImplantCandidate }) {
  const score = num(c.fit_score);
  const rows: Array<[string, string]> = [
    ["Fit score", score === null ? "—" : `${score.toFixed(1)} / 100`],
    ["Coverage", fmt(num(c.coverage_pct), "%", 1)],
    ["Max overhang", fmt(num(c.max_overhang_mm), "mm", 2)],
    ["Undercoverage", fmt(num(c.undercoverage_mm), "mm", 2)],
  ];
  return (
    <div>
      {rows.map(([k, v]) => (
        <div key={k} className="row">
          <span className="row-key">{k}</span>
          <span className="row-val">{v}</span>
        </div>
      ))}
    </div>
  );
}

export default function ImplantPanel({ data }: { data: ImplantResult | null }) {
  const [showAll, setShowAll] = useState(false);

  if (!data) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">Implant</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] text-slate-500">Not available for this case.</p>
        </div>
      </div>
    );
  }

  const selected = data.selected;
  const runnerUp = data.runner_up;
  const rationale = Array.isArray(data.rationale)
    ? data.rationale
    : data.rationale
      ? [data.rationale]
      : [];
  const gap =
    num(selected?.fit_score) !== null && num(runnerUp?.fit_score) !== null
      ? num(selected!.fit_score)! - num(runnerUp!.fit_score)!
      : null;

  return (
    <div className="card">
      <div className="card-head">
        <span className="card-title">Implant recommendation</span>
        <span className="card-sub">{data.candidates?.length ?? 0} ranked</span>
      </div>

      <div className="card-body">
        {selected ? (
          <>
            <div className="mb-2 flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate text-[14px] font-semibold text-slate-50">
                  {selected.configuration_label ?? selected.configuration_id ?? "Selected"}
                </div>
                <div className="mt-0.5 font-mono text-[11px] text-slate-500">
                  femoral {selected.femoral_size ?? "—"} · tibial {selected.tibial_size ?? "—"}
                </div>
              </div>
              <span className="chip shrink-0 border-sky-500 bg-sky-500/15 text-sky-300">
                recommended
              </span>
            </div>

            <Metrics c={selected} />

            {selected.criteria && Object.keys(selected.criteria).length > 0 && (
              <div className="mt-3">
                <h4 className="section-label">Why this size</h4>
                {Object.entries(selected.criteria).map(([key, value]) => (
                  <p key={key} className="mb-1 text-[11px] leading-relaxed text-slate-400">
                    <span className="font-mono text-slate-300">{key}</span> — {String(value)}
                  </p>
                ))}
              </div>
            )}

            {rationale.length > 0 && (
              <div className="mt-2 space-y-1">
                {rationale.map((r, i) => (
                  <p key={i} className="text-[11px] leading-relaxed text-slate-400">
                    {r}
                  </p>
                ))}
              </div>
            )}
          </>
        ) : (
          <p className="text-[12px] text-slate-500">No candidate met the fit criteria.</p>
        )}

        {runnerUp && (
          <div className="mt-4 border-t border-slate-800 pt-3">
            <div className="mb-2 flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate text-[12px] font-medium text-slate-300">
                  {runnerUp.configuration_label ?? runnerUp.configuration_id ?? "Runner-up"}
                </div>
                <div className="mt-0.5 font-mono text-[11px] text-slate-500">
                  femoral {runnerUp.femoral_size ?? "—"} · tibial {runnerUp.tibial_size ?? "—"}
                </div>
              </div>
              <span className="chip shrink-0 border-slate-600 bg-slate-800 text-slate-400">
                runner-up
              </span>
            </div>
            <Metrics c={runnerUp} />
            {gap !== null && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-slate-400">
                Lost by <span className="font-mono text-slate-200">{gap.toFixed(1)}</span> points.
                {runnerUp.criteria &&
                  ` Weakest criterion: ${Object.keys(runnerUp.criteria)[0] ?? "unspecified"}.`}
              </p>
            )}
          </div>
        )}

        {data.candidates && data.candidates.length > 2 && (
          <>
            <button
              type="button"
              className="btn-ghost mt-3 w-full"
              onClick={() => setShowAll(!showAll)}
              aria-expanded={showAll}
            >
              {showAll ? "Hide" : `Show all ${data.candidates.length}`} candidates
            </button>
            {showAll && (
              <div className="mt-2 max-h-48 overflow-y-auto">
                {data.candidates.map((c, i) => (
                  <div key={i} className="row">
                    <span className="row-key">
                      {c.configuration_label ?? c.configuration_id ?? `candidate ${i + 1}`}
                    </span>
                    <span className="row-val">
                      {num(c.fit_score) === null ? "—" : num(c.fit_score)!.toFixed(1)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        <p className="mt-3 border-t border-slate-800 pt-2.5 text-[11px] leading-relaxed text-amber-400">
          {data.disclaimer ?? "Reference (non-clinical) implant geometry, not manufacturer data."}{" "}
          This is not a clinically approved implant selection.
        </p>
      </div>
    </div>
  );
}
