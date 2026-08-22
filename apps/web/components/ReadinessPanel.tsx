"use client";

/**
 * Robotic execution readiness as a percentage, with its arithmetic on display.
 *
 * A single percentage is the most quotable thing this platform produces, which
 * makes it the most dangerous. Two design choices push back on that:
 *
 * **Blockers are shown before the breakdown.** When something genuinely prevents
 * execution — unknown operative side, a failed geometric simulation — it caps the
 * score, and the cap is stated with the raw weighted figure next to it so the
 * reader can see the number was held down rather than earned.
 *
 * **Every criterion is listed with its weight and its own score.** The total is
 * recomputable by hand from the table. A reader who distrusts the weighting can see
 * exactly what it was and disagree with it specifically.
 *
 * The wording throughout says "readiness for simulation", never "suitability for
 * robotic surgery". The score measures artefact completeness and internal
 * consistency. It has never seen a patient.
 */

import type { ReadinessCriterion, RoboticReadiness } from "@/lib/types";

const BAND_STYLE: Record<string, string> = {
  ready: "border-emerald-600 bg-emerald-500/15 text-emerald-300",
  conditional: "border-amber-600 bg-amber-500/15 text-amber-300",
  not_ready: "border-rose-600 bg-rose-500/15 text-rose-300",
};

const BAND_RING: Record<string, string> = {
  ready: "text-emerald-400",
  conditional: "text-amber-400",
  not_ready: "text-rose-400",
};

const BAND_BAR: Record<string, string> = {
  ready: "bg-emerald-500",
  conditional: "bg-amber-500",
  not_ready: "bg-rose-500",
};

/** A ring gauge. SVG rather than a bar: the headline number needs to read at a glance. */
function Gauge({ percentage, band }: { percentage: number; band: string }) {
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const filled = (Math.max(0, Math.min(100, percentage)) / 100) * circumference;
  return (
    <svg viewBox="0 0 80 80" className="h-20 w-20 shrink-0 -rotate-90" aria-hidden="true">
      <circle cx="40" cy="40" r={radius} className="fill-none stroke-slate-800" strokeWidth="7" />
      <circle
        cx="40"
        cy="40"
        r={radius}
        className={`fill-none ${BAND_RING[band] ?? "text-slate-400"}`}
        stroke="currentColor"
        strokeWidth="7"
        strokeLinecap="round"
        strokeDasharray={`${filled} ${circumference}`}
      />
    </svg>
  );
}

function CriterionRow({ criterion }: { criterion: ReadinessCriterion }) {
  const pct = criterion.score_pct;
  const tone = pct >= 80 ? "bg-emerald-500" : pct >= 50 ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className="border-b border-slate-800/60 py-2 last:border-0">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] font-medium text-slate-200">{criterion.display_name}</span>
        <span className="shrink-0 font-mono text-[12px] text-slate-100">{pct.toFixed(0)}%</span>
      </div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-slate-800">
        <div className={`h-full ${tone}`} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
      </div>
      <div className="mt-1 flex items-baseline justify-between gap-2 font-mono text-[10px] text-slate-600">
        <span>weight {(criterion.weight * 100).toFixed(0)}%</span>
        <span>contributes {criterion.weighted_contribution_pct.toFixed(1)} pts</span>
      </div>
      <p className="mt-1 text-[11px] leading-snug text-slate-500">{criterion.detail}</p>
    </div>
  );
}

export default function ReadinessPanel({
  readiness,
  reason,
}: {
  readiness?: RoboticReadiness | null;
  /** Why readiness is absent, when the backend supplied an explanation. */
  reason?: string | null;
}) {
  if (!readiness) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">Robotic readiness</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] font-medium text-slate-300">Not available</p>
          <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
            {reason ??
              "No readiness score for this case. It is computed at the end of the pipeline, so re-run the pipeline to generate one."}
          </p>
        </div>
      </div>
    );
  }

  const bandStyle = BAND_STYLE[readiness.band] ?? "border-slate-600 bg-slate-700/40 text-slate-300";

  return (
    <>
      <div className="card">
        <div className="card-head">
          <span className="card-title">Robotic readiness</span>
          <span className={`chip ${bandStyle}`}>{readiness.band.replace(/_/g, " ")}</span>
        </div>

        <div className="card-body">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Gauge percentage={readiness.percentage} band={readiness.band} />
              <span
                className={`absolute inset-0 flex items-center justify-center font-mono text-[17px] font-semibold ${
                  BAND_RING[readiness.band] ?? "text-slate-300"
                }`}
              >
                {readiness.percentage.toFixed(0)}%
              </span>
            </div>
            <div className="min-w-0">
              <p className="text-[12px] leading-relaxed text-slate-300">{readiness.band_note}</p>
              {readiness.capped && (
                <p className="mt-1 text-[11px] leading-snug text-amber-400">
                  Capped at {readiness.ceiling_pct.toFixed(0)}% by a blocker. The weighted
                  criteria alone would have read{" "}
                  {readiness.raw_weighted_percentage.toFixed(0)}%.
                </p>
              )}
            </div>
          </div>

          <p className="mt-3 text-[11px] leading-relaxed text-slate-400">
            This scores whether the case <strong>artefacts</strong> are complete and
            self-consistent enough to simulate a constrained robotic execution of the
            plan. It is not a clinical judgement that this knee should be operated on
            robotically.
          </p>
        </div>
      </div>

      {/* Blockers come before the breakdown: they are the reason the number is what it is. */}
      {readiness.blockers.length > 0 && (
        <div className="card border-rose-800">
          <div className="card-head">
            <span className="card-title text-rose-300">
              Blockers ({readiness.blockers.length})
            </span>
          </div>
          <div className="card-body">
            {readiness.blockers.map((blocker) => (
              <div key={blocker.code} className="border-b border-slate-800/60 py-2 last:border-0">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-mono text-[11px] text-rose-300">{blocker.code}</span>
                  <span className="shrink-0 font-mono text-[10px] text-slate-500">
                    caps at {blocker.ceiling_pct.toFixed(0)}%
                  </span>
                </div>
                <p className="mt-0.5 text-[11px] leading-relaxed text-slate-400">
                  {blocker.message}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <span className="card-title">Score breakdown</span>
          <span className="card-sub">{readiness.criteria.length} criteria</span>
        </div>
        <div className="card-body">
          {readiness.criteria.map((criterion) => (
            <CriterionRow key={criterion.name} criterion={criterion} />
          ))}
          <div className="mt-2 flex items-baseline justify-between border-t border-slate-800 pt-2">
            <span className="text-[12px] font-medium text-slate-300">
              Weighted total
            </span>
            <span className="font-mono text-[12px] text-slate-100">
              {readiness.raw_weighted_percentage.toFixed(1)}%
            </span>
          </div>
          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">{readiness.method}</p>
        </div>
      </div>

      {readiness.warnings.length > 0 && (
        <div className="card border-amber-800">
          <div className="card-head">
            <span className="card-title text-amber-300">Caveats</span>
          </div>
          <div className="card-body">
            <ul className="space-y-1.5 text-[11px] leading-relaxed text-amber-400">
              {readiness.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <p className="px-1 text-[10px] leading-relaxed text-slate-500">{readiness.disclaimer}</p>
    </>
  );
}
