"use client";

/**
 * Implant size explorer.
 *
 * Steps through the catalogue's real size range and previews each one on the model.
 *
 * The matcher shortlists only sizes that are dimensionally plausible, so on most cases
 * its candidate list is a single femoral/tibial size varying by insert thickness and
 * CR/PS — nothing to explore. The catalogue itself carries eight femoral and seven
 * tibial sizes, so the ladder here is built from the catalogue and the matcher's choice
 * is marked on it. That way "what would one size up look like" is answerable, which is
 * the actual question in front of a surgeon holding trials.
 *
 * The overhang figure is measured, not scored: it is the component's mediolateral width
 * against the bone's, halved, which is what hangs over each side. The planned size's
 * own fit metrics come from the engine and are shown alongside.
 */

import { useEffect, useMemo, useState } from "react";

import type { ImplantCandidate, ImplantResult } from "@/lib/types";

/** Reference catalogue, mirroring ``kneeai.implants.reference``. */
const FEMORAL_SIZES = [
  { label: "1", ml: 55.0, ap: 27.5 },
  { label: "2", ml: 58.5, ap: 29.2 },
  { label: "3", ml: 62.0, ap: 30.8 },
  { label: "4", ml: 66.0, ap: 32.5 },
  { label: "5", ml: 70.0, ap: 34.1 },
  { label: "6", ml: 74.0, ap: 35.8 },
  { label: "7", ml: 78.0, ap: 37.4 },
  { label: "8", ml: 82.0, ap: 39.1 },
];

const TIBIAL_SIZES = [
  { label: "1", ml: 58.0, ap: 39.0 },
  { label: "2", ml: 62.0, ap: 41.5 },
  { label: "3", ml: 66.0, ap: 44.0 },
  { label: "4", ml: 70.0, ap: 46.5 },
  { label: "5", ml: 74.0, ap: 49.0 },
  { label: "6", ml: 78.0, ap: 51.5 },
  { label: "7", ml: 82.0, ap: 54.0 },
];

/** Engine scores are 0-100; treat anything above 1 as already a percentage. */
function asPercent(score: number | null | undefined): number | null {
  if (score == null || !Number.isFinite(score)) return null;
  return score > 1 ? score : score * 100;
}

function toneFor(pct: number | null): { text: string; bar: string } {
  if (pct == null) return { text: "text-slate-400", bar: "bg-slate-700" };
  if (pct >= 85) return { text: "text-emerald-400", bar: "bg-emerald-500" };
  if (pct >= 70) return { text: "text-sky-400", bar: "bg-sky-500" };
  if (pct >= 50) return { text: "text-amber-400", bar: "bg-amber-500" };
  return { text: "text-rose-400", bar: "bg-rose-500" };
}

function Row({
  label,
  value,
  suffix = "",
  warn = false,
}: {
  label: string;
  value: string;
  suffix?: string;
  warn?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5">
      <span className="text-[11px] text-slate-500">{label}</span>
      <span className={`font-mono text-[11px] ${warn ? "text-amber-400" : "text-slate-300"}`}>
        {value}
        {suffix}
      </span>
    </div>
  );
}

export default function ImplantSizeSelector({
  data,
  onSelect,
}: {
  data: ImplantResult | null;
  /**
   * Fires with the femoral and tibial scale factors to apply to the previewed
   * component, relative to the planned size whose geometry was exported.
   */
  onSelect?: (preview: {
    femoral: { ml: number; ap: number };
    tibial: { ml: number; ap: number };
    sizeLabel: string;
    isPlanned: boolean;
  }) => void;
}) {
  const planned = data?.selected ?? data?.candidates?.[0] ?? null;

  const plannedFem = planned?.femoral_fit;
  const plannedTib = planned?.tibial_fit;

  // Index of the size the engine chose, so the ladder can mark it and scales can be
  // taken relative to the geometry that was actually exported.
  const plannedFemIndex = useMemo(() => {
    const label = plannedFem?.size_label;
    const byLabel = FEMORAL_SIZES.findIndex((s) => s.label === String(label));
    if (byLabel >= 0) return byLabel;
    const ml = plannedFem?.component_ml_mm;
    if (ml == null) return 4;
    let best = 0;
    for (let i = 1; i < FEMORAL_SIZES.length; i += 1) {
      if (Math.abs(FEMORAL_SIZES[i].ml - ml) < Math.abs(FEMORAL_SIZES[best].ml - ml)) best = i;
    }
    return best;
  }, [plannedFem?.size_label, plannedFem?.component_ml_mm]);

  const [index, setIndex] = useState(plannedFemIndex);

  // Bone dimensions, recovered from the planned component and its mismatch. The engine
  // reports mismatch as component minus bone, so bone = component - mismatch.
  const boneFemMl =
    plannedFem?.component_ml_mm != null && plannedFem?.ml_mismatch_mm != null
      ? plannedFem.component_ml_mm - plannedFem.ml_mismatch_mm
      : null;
  const boneTibMl =
    plannedTib?.component_ml_mm != null && plannedTib?.ml_mismatch_mm != null
      ? plannedTib.component_ml_mm - plannedTib.ml_mismatch_mm
      : null;

  const emit = (i: number) => {
    const fem = FEMORAL_SIZES[i];
    const refFem = FEMORAL_SIZES[plannedFemIndex];
    const tibIndex = Math.min(i, TIBIAL_SIZES.length - 1);
    const tib = TIBIAL_SIZES[tibIndex];
    const refTib = TIBIAL_SIZES[Math.min(plannedFemIndex, TIBIAL_SIZES.length - 1)];
    onSelect?.({
      femoral: { ml: fem.ml / refFem.ml, ap: fem.ap / refFem.ap },
      tibial: { ml: tib.ml / refTib.ml, ap: tib.ap / refTib.ap },
      sizeLabel: fem.label,
      isPlanned: i === plannedFemIndex,
    });
  };

  useEffect(() => {
    setIndex(plannedFemIndex);
    emit(plannedFemIndex);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plannedFemIndex, data]);

  if (!data || !planned) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">Implant sizing</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] text-slate-500">
            No implant was matched for this case.
          </p>
        </div>
      </div>
    );
  }

  const fem = FEMORAL_SIZES[index];
  const tib = TIBIAL_SIZES[Math.min(index, TIBIAL_SIZES.length - 1)];
  const isPlanned = index === plannedFemIndex;

  const step = (delta: number) => {
    const next = Math.max(0, Math.min(FEMORAL_SIZES.length - 1, index + delta));
    setIndex(next);
    emit(next);
  };

  // Overhang for the size on screen, measured against the bone.
  const femOverhang = boneFemMl != null ? (fem.ml - boneFemMl) / 2 : null;
  const tibOverhang = boneTibMl != null ? (tib.ml - boneTibMl) / 2 : null;

  const plannedScore = asPercent(plannedFem?.score);
  const tone = toneFor(plannedScore);

  const verdict = (() => {
    const worst = Math.max(femOverhang ?? 0, tibOverhang ?? 0);
    const under = Math.min(femOverhang ?? 0, tibOverhang ?? 0);
    if (worst > 3) return { text: "Overhangs the bone", tone: "text-rose-400" };
    if (under < -3) return { text: "Undercovers the bone", tone: "text-amber-400" };
    return { text: "Within tolerance", tone: "text-emerald-400" };
  })();

  return (
    <div className="card">
      <div className="card-head">
        <span className="card-title">Implant sizing</span>
        <span className="card-sub">
          {FEMORAL_SIZES.length} catalogue sizes
        </span>
      </div>

      <div className="card-body">
        {/* ---------- Stepper ---------- */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => step(-1)}
            disabled={index === 0}
            aria-label="Smaller size"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-slate-700 text-[18px] text-slate-300 transition-colors hover:border-sky-600 hover:bg-sky-950/40 disabled:opacity-25"
          >
            −
          </button>

          <div
            className={`min-w-0 flex-1 rounded-md border px-3 py-2 text-center ${
              isPlanned ? "border-emerald-700 bg-emerald-950/25" : "border-sky-800 bg-sky-950/25"
            }`}
          >
            <p className="text-[15px] font-semibold text-slate-50">Size {fem.label}</p>
            <p className="mt-0.5 font-mono text-[10px] text-slate-400">
              fem {fem.ml.toFixed(0)}×{fem.ap.toFixed(0)} · tib {tib.ml.toFixed(0)}×
              {tib.ap.toFixed(0)} mm
            </p>
          </div>

          <button
            type="button"
            onClick={() => step(1)}
            disabled={index >= FEMORAL_SIZES.length - 1}
            aria-label="Larger size"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-slate-700 text-[18px] text-slate-300 transition-colors hover:border-sky-600 hover:bg-sky-950/40 disabled:opacity-25"
          >
            +
          </button>
        </div>

        {/* ---------- Size ladder ---------- */}
        <div className="mt-2 flex gap-1">
          {FEMORAL_SIZES.map((s, i) => (
            <button
              key={s.label}
              type="button"
              onClick={() => {
                setIndex(i);
                emit(i);
              }}
              aria-label={`Size ${s.label}`}
              title={`Size ${s.label} — ${s.ml.toFixed(0)} mm ML${
                i === plannedFemIndex ? " (planned)" : ""
              }`}
              className={`h-7 flex-1 rounded text-[10px] font-mono transition-colors ${
                i === index
                  ? "bg-sky-500 text-white"
                  : i === plannedFemIndex
                    ? "bg-emerald-600/40 text-emerald-200 hover:bg-emerald-600/60"
                    : "bg-slate-800 text-slate-500 hover:bg-slate-700"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div className="mt-1 flex items-center justify-between text-[10px] text-slate-600">
          <span>smaller</span>
          <span className="text-emerald-500">green = engine&apos;s choice</span>
          <span>larger</span>
        </div>

        {/* ---------- Status ---------- */}
        <div className="mt-3 flex items-center justify-between gap-2">
          <span
            className={`chip ${
              isPlanned
                ? "border-emerald-600 bg-emerald-500/15 text-emerald-300"
                : "border-sky-600 bg-sky-500/15 text-sky-300"
            }`}
          >
            {isPlanned ? "planned size" : `preview: ${index > plannedFemIndex ? "larger" : "smaller"}`}
          </span>
          <span className={`text-[11px] font-medium ${verdict.tone}`}>{verdict.text}</span>
        </div>

        {/* ---------- Measured fit for the size on screen ---------- */}
        <div className="mt-3 grid grid-cols-2 gap-3">
          <div>
            <h4 className="section-label">Femoral</h4>
            <Row label="Component ML" value={fem.ml.toFixed(1)} suffix=" mm" />
            <Row label="Component AP" value={fem.ap.toFixed(1)} suffix=" mm" />
            {boneFemMl != null && (
              <Row label="Bone ML" value={boneFemMl.toFixed(1)} suffix=" mm" />
            )}
            {femOverhang != null && (
              <Row
                label={femOverhang >= 0 ? "Overhang / side" : "Undercover / side"}
                value={Math.abs(femOverhang).toFixed(1)}
                suffix=" mm"
                warn={Math.abs(femOverhang) > 3}
              />
            )}
          </div>

          <div>
            <h4 className="section-label">Tibial</h4>
            <Row label="Component ML" value={tib.ml.toFixed(1)} suffix=" mm" />
            <Row label="Component AP" value={tib.ap.toFixed(1)} suffix=" mm" />
            {boneTibMl != null && (
              <Row label="Bone ML" value={boneTibMl.toFixed(1)} suffix=" mm" />
            )}
            {tibOverhang != null && (
              <Row
                label={tibOverhang >= 0 ? "Overhang / side" : "Undercover / side"}
                value={Math.abs(tibOverhang).toFixed(1)}
                suffix=" mm"
                warn={Math.abs(tibOverhang) > 3}
              />
            )}
          </div>
        </div>

        {/* ---------- Engine's score for the planned size ---------- */}
        {plannedScore != null && (
          <div className="mt-3 border-t border-slate-800 pt-2">
            <div className="flex items-baseline justify-between">
              <span className="section-label !mb-0">
                Engine fit score · size {plannedFem?.size_label ?? "—"}
              </span>
              <span className={`font-mono text-[13px] font-semibold ${tone.text}`}>
                {plannedScore.toFixed(0)}%
              </span>
            </div>
            <div className="mt-1 h-2 w-full overflow-hidden rounded bg-slate-800">
              <div
                className={`h-full ${tone.bar}`}
                style={{ width: `${Math.max(0, Math.min(100, plannedScore))}%` }}
              />
            </div>
            {asPercent(plannedTib?.score) != null && (
              <p className="mt-1 text-[10px] text-slate-500">
                Tibial component scores {asPercent(plannedTib?.score)!.toFixed(0)}% on the
                resected footprint
                {plannedTib?.coverage_pct != null &&
                  ` · ${plannedTib.coverage_pct.toFixed(0)}% coverage`}
                .
              </p>
            )}
          </div>
        )}

        {/* ---------- Violations on the planned size ---------- */}
        {(planned.violations?.length ?? 0) > 0 && (
          <div className="mt-3 rounded-md border border-rose-900/70 bg-rose-950/30 px-2.5 py-2">
            <h4 className="section-label !mb-1 text-rose-300">Engine flagged</h4>
            <ul className="space-y-1">
              {planned.violations!.map((v, i) => (
                <li key={i} className="text-[11px] leading-snug text-rose-200">
                  {v}
                </li>
              ))}
            </ul>
          </div>
        )}

        <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
          {isPlanned
            ? "This is the size the matcher chose. Step through the ladder to see how a larger or smaller component sits."
            : "Preview only. The component in the viewer is the planned shell scaled to this size's catalogue dimensions — extents are right, internal geometry is stretched. Overhang is measured against this knee's bone; the engine only scored the planned size."}
        </p>
      </div>
    </div>
  );
}
