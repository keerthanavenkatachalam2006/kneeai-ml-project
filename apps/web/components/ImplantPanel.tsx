"use client";

/**
 * Implant recommendation, with the reasoning kept visible.
 *
 * This is the panel that distinguishes the system from a 3D viewer, so the runner-up
 * and the reason it lost carry the same weight as the winner. A recommendation a
 * surgeon cannot interrogate is one they must either take on faith or discard, and
 * PRD §2.2 requires the evidence to travel with the number.
 *
 * The femoral and tibial components are shown separately because they are scored
 * differently and for a reason: a tibial baseplate seats on one flat cut, so coverage
 * and overhang of that footprint are the whole story, while a femoral component is
 * captured between five cuts and never seats on a single plane, so its fit is governed
 * by mediolateral overhang and anteroposterior envelope instead. Presenting one set of
 * numbers for both would mean showing blanks for half of them.
 *
 * The catalog disclaimer renders unconditionally: this is reference geometry, not
 * manufacturer data, and the fit numbers are only as meaningful as the shapes behind
 * them.
 */

import { useState } from "react";
import { fmt } from "@/lib/api";
import type { ComponentFit, ImplantCandidate, ImplantResult } from "@/lib/types";

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function label(c: ImplantCandidate | null | undefined): string {
  return c?.configuration?.label ?? c?.configuration?.configuration_id ?? "—";
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="row">
      <span className="row-key">{k}</span>
      <span className="row-val">{v}</span>
    </div>
  );
}

function FitBlock({ title, fit }: { title: string; fit: ComponentFit | undefined }) {
  if (!fit) return null;
  const score = num(fit.score);
  const isFootprint = num(fit.coverage_pct) !== null;

  return (
    <div className="mt-2.5">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <h4 className="section-label !mb-0">
          {title} {fit.size_label ? `· size ${fit.size_label}` : ""}
        </h4>
        <span
          className={
            "chip shrink-0 " +
            (fit.acceptable
              ? "border-emerald-600 bg-emerald-500/10 text-emerald-300"
              : "border-amber-600 bg-amber-500/10 text-amber-300")
          }
        >
          {score === null ? "—" : score.toFixed(1)}
          {fit.acceptable ? "" : " · constrained"}
        </span>
      </div>

      {isFootprint ? (
        <>
          <Row k="Coverage" v={fmt(num(fit.coverage_pct), "%", 1)} />
          <Row k="Max overhang" v={fmt(num(fit.overhang_max_mm), "mm", 2)} />
          <Row k="Uncovered rim" v={fmt(num(fit.undercoverage_max_mm), "mm", 2)} />
        </>
      ) : (
        <>
          <Row k="Mediolateral vs bone" v={fmt(num(fit.ml_mismatch_mm), "mm", 1)} />
          <Row k="Anteroposterior vs bone" v={fmt(num(fit.ap_mismatch_mm), "mm", 1)} />
        </>
      )}

      {(fit.violations ?? []).map((v, i) => (
        <p key={i} className="mt-1 text-[11px] leading-relaxed text-amber-400">
          {v}
        </p>
      ))}
    </div>
  );
}

/** The scoring terms, which are where the trade-off actually becomes visible. */
function Terms({ fit }: { fit: ComponentFit | undefined }) {
  const terms = fit?.terms ?? [];
  if (terms.length === 0) return null;
  return (
    <div className="mt-1.5 space-y-1">
      {terms.map((t, i) => (
        <p key={i} className="text-[11px] leading-relaxed text-slate-400">
          <span className="font-mono text-slate-300">{t.name.replace(/_/g, " ")}</span>
          {t.explanation ? ` — ${t.explanation}` : ""}
        </p>
      ))}
    </div>
  );
}

export default function ImplantPanel({ data }: { data: ImplantResult | null }) {
  const [showAll, setShowAll] = useState(false);
  const [showTerms, setShowTerms] = useState(false);

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
    num(selected?.score) !== null && num(runnerUp?.score) !== null
      ? num(selected!.score)! - num(runnerUp!.score)!
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
            <div className="mb-1.5 flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate text-[14px] font-semibold text-slate-50">
                  {label(selected)}
                </div>
                <div className="mt-0.5 font-mono text-[11px] text-slate-500">
                  {selected.configuration?.bearing?.replace(/_/g, " ") ?? "—"} ·{" "}
                  {selected.configuration?.fixation ?? "—"} ·{" "}
                  {fmt(num(selected.configuration?.total_construct_thickness_mm), "mm", 1)}{" "}
                  construct
                </div>
              </div>
              <span
                className={
                  "chip shrink-0 " +
                  (selected.acceptable
                    ? "border-sky-500 bg-sky-500/15 text-sky-300"
                    : "border-amber-500 bg-amber-500/15 text-amber-300")
                }
              >
                {selected.acceptable ? "recommended" : "best available"}
              </span>
            </div>

            <Row
              k="Combined fit"
              v={num(selected.score) === null ? "—" : `${num(selected.score)!.toFixed(1)} / 100`}
            />

            {/* When nothing clears the thresholds the panel has to say so plainly,
                rather than presenting the least-bad option as a recommendation. */}
            {!selected.acceptable && (
              <p className="mt-2 rounded border border-amber-700/60 bg-amber-500/10 p-2 text-[11px] leading-relaxed text-amber-300">
                No configuration in this catalog met every fit constraint. This is the
                highest-scoring one; the unmet constraints are listed below and must be
                resolved before approval.
              </p>
            )}

            <FitBlock title="Femoral" fit={selected.femoral_fit} />
            <FitBlock title="Tibial baseplate" fit={selected.tibial_fit} />

            <button
              type="button"
              className="btn-ghost mt-2.5 w-full"
              onClick={() => setShowTerms(!showTerms)}
              aria-expanded={showTerms}
            >
              {showTerms ? "Hide scoring detail" : "Why this size"}
            </button>
            {showTerms && (
              <div className="mt-1.5">
                <Terms fit={selected.femoral_fit} />
                <Terms fit={selected.tibial_fit} />
                {rationale.length > 0 && (
                  <pre className="mt-2 max-h-56 overflow-y-auto whitespace-pre-wrap font-mono text-[10.5px] leading-relaxed text-slate-500">
                    {rationale.join("\n")}
                  </pre>
                )}
              </div>
            )}
          </>
        ) : (
          <p className="text-[12px] text-slate-500">
            No candidate could be scored for this case.
          </p>
        )}

        {runnerUp && (
          <div className="mt-4 border-t border-slate-800 pt-3">
            <div className="mb-1 flex items-start justify-between gap-3">
              <div className="min-w-0 truncate text-[12px] font-medium text-slate-300">
                {label(runnerUp)}
              </div>
              <span className="chip shrink-0 border-slate-600 bg-slate-800 text-slate-400">
                runner-up
              </span>
            </div>
            <Row
              k="Combined fit"
              v={num(runnerUp.score) === null ? "—" : `${num(runnerUp.score)!.toFixed(1)} / 100`}
            />
            {gap !== null && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-slate-400">
                {Math.abs(gap) < 0.05
                  ? "Scores equally: these differ only in a dimension the fit model does not measure, such as insert thickness or bearing type."
                  : `Lost by ${gap.toFixed(1)} points.`}
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
                    <span className="row-key">{label(c)}</span>
                    <span className="row-val">
                      {num(c.score) === null ? "—" : num(c.score)!.toFixed(1)}
                      {c.acceptable ? "" : " ⚠"}
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
