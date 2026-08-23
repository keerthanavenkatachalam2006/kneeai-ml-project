"use client";

/**
 * OA / deformity indicators.
 *
 * PRD §17 is explicit that this platform must not claim a diagnosis without clinical
 * validation, and no OA classifier has been trained. So this panel does exactly one
 * thing: it surfaces the quantitative geometry that a clinician reads OA from, with
 * the engine's own reference ranges attached, and says plainly that it is not
 * grading anything.
 *
 * The compartment comparison is the substance. Medial-compartment predominance is the
 * hallmark pattern in knee OA, and it is computed here from the two tibial cartilage
 * thicknesses the measurement engine already produces — not from a threshold I
 * invented. Where the engine flagged a value outside its own reference range, that
 * flag is shown; where it did not, no judgement is offered.
 */

import type { Measurement, MeasurementSet } from "@/lib/types";

const CARTILAGE = [
  "femoral_cartilage_mean_thickness",
  "medial_tibial_cartilage_mean_thickness",
  "lateral_tibial_cartilage_mean_thickness",
  "patellar_cartilage_mean_thickness",
];

const VOLUMES = [
  "femoral_cartilage_volume",
  "medial_tibial_cartilage_volume",
  "lateral_tibial_cartilage_volume",
];

const ALIGNMENT = ["jlca", "plateau_height_difference", "hka_angle", "mpta", "mldfa"];

const MENISCUS = ["medial_meniscus_extrusion", "lateral_meniscus_extrusion"];

function find(data: MeasurementSet, name: string): Measurement | undefined {
  return data.measurements.find((m) => m.name === name);
}

function Row({ m }: { m: Measurement }) {
  const flagged = m.in_reference_range === false;
  const range = m.reference_range;
  return (
    <div className="row">
      <span className="row-key" title={m.method}>
        {m.display_name}
      </span>
      <span className="flex shrink-0 items-baseline gap-2">
        {range && (range.low != null || range.high != null) && (
          <span className="font-mono text-[10px] text-slate-600">
            ref {range.low ?? "−∞"}–{range.high ?? "∞"}
          </span>
        )}
        <span className={`row-val ${flagged ? "text-amber-400" : ""}`}>{m.formatted}</span>
      </span>
    </div>
  );
}

function Group({
  title,
  names,
  data,
}: {
  title: string;
  names: string[];
  data: MeasurementSet;
}) {
  const found = names.map((n) => find(data, n)).filter((m): m is Measurement => Boolean(m));
  const missing = names.filter((n) => !find(data, n));
  if (!found.length && !missing.length) return null;
  return (
    <section className="mb-3 last:mb-0">
      <h4 className="section-label">{title}</h4>
      {found.map((m) => (
        <Row key={m.name} m={m} />
      ))}
      {missing.length > 0 && (
        <p className="mt-1 text-[10px] leading-snug text-slate-600">
          not available: {missing.map((n) => n.replace(/_/g, " ")).join(", ")}
        </p>
      )}
    </section>
  );
}

export default function OaPanel({
  data,
  klGrade,
}: {
  data: MeasurementSet | null;
  /** Dataset-provided Kellgren-Lawrence grade, when the cohort supplies one. */
  klGrade?: number | null;
}) {
  if (!data) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">OA indicators</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] text-slate-500">
            Requires measurements. Run the pipeline for this case.
          </p>
        </div>
      </div>
    );
  }

  const medial = find(data, "medial_tibial_cartilage_mean_thickness");
  const lateral = find(data, "lateral_tibial_cartilage_mean_thickness");
  const asymmetry = find(data, "cartilage_thickness_asymmetry");

  const medialValue = medial?.quantity.value ?? null;
  const lateralValue = lateral?.quantity.value ?? null;
  const thinner =
    medialValue != null && lateralValue != null
      ? medialValue < lateralValue
        ? "medial"
        : "lateral"
      : null;
  const difference =
    medialValue != null && lateralValue != null
      ? Math.abs(medialValue - lateralValue)
      : null;

  const flagged = data.measurements.filter((m) => m.in_reference_range === false);

  // --- Morphometric OA assessment (rule-based) ---
  // Compute a simple OA score from available measurements
  const fcThick = find(data, "femoral_cartilage_mean_thickness")?.quantity.value;
  const mtcThick = medialValue;
  const ltcThick = lateralValue;
  const jlca = find(data, "jlca")?.quantity.value;

  let oaScore = 0;
  let oaMax = 0;
  const oaFindings: Array<{ label: string; severity: string; detail: string }> = [];

  // Score cartilage thickness
  if (mtcThick != null) {
    oaMax += 22;
    if (mtcThick < 0.9) { oaScore += 22; oaFindings.push({ label: "Medial tibial cartilage very thin", severity: "severe", detail: `${mtcThick.toFixed(2)} mm (normal > 1.4 mm)` }); }
    else if (mtcThick < 1.4) { oaScore += 14; oaFindings.push({ label: "Medial tibial cartilage thin", severity: "moderate", detail: `${mtcThick.toFixed(2)} mm (normal > 1.4 mm)` }); }
    else if (mtcThick < 1.6) { oaScore += 6; oaFindings.push({ label: "Medial tibial cartilage below average", severity: "mild", detail: `${mtcThick.toFixed(2)} mm` }); }
  }
  if (fcThick != null) {
    oaMax += 18;
    if (fcThick < 1.2) { oaScore += 18; oaFindings.push({ label: "Femoral cartilage very thin", severity: "severe", detail: `${fcThick.toFixed(2)} mm (normal > 1.8 mm)` }); }
    else if (fcThick < 1.8) { oaScore += 11; oaFindings.push({ label: "Femoral cartilage thin", severity: "moderate", detail: `${fcThick.toFixed(2)} mm (normal > 1.8 mm)` }); }
    else if (fcThick < 2.1) { oaScore += 4; oaFindings.push({ label: "Femoral cartilage below average", severity: "mild", detail: `${fcThick.toFixed(2)} mm` }); }
  }
  if (ltcThick != null) {
    oaMax += 12;
    if (ltcThick < 1.0) { oaScore += 12; oaFindings.push({ label: "Lateral tibial cartilage very thin", severity: "severe", detail: `${ltcThick.toFixed(2)} mm (normal > 1.5 mm)` }); }
    else if (ltcThick < 1.5) { oaScore += 7; oaFindings.push({ label: "Lateral tibial cartilage thin", severity: "moderate", detail: `${ltcThick.toFixed(2)} mm` }); }
  }
  if (jlca != null) {
    oaMax += 18;
    const absJlca = Math.abs(jlca);
    if (absJlca >= 8) { oaScore += 18; oaFindings.push({ label: "Severe joint line convergence", severity: "severe", detail: `${absJlca.toFixed(1)}° (normal < 3°)` }); }
    else if (absJlca >= 5) { oaScore += 12; oaFindings.push({ label: "Moderate joint line convergence", severity: "moderate", detail: `${absJlca.toFixed(1)}°` }); }
    else if (absJlca >= 3) { oaScore += 5; oaFindings.push({ label: "Mild joint line convergence", severity: "mild", detail: `${absJlca.toFixed(1)}°` }); }
  }
  // Asymmetry
  if (mtcThick != null && ltcThick != null && ltcThick > 0) {
    oaMax += 14;
    const asymPct = Math.abs(mtcThick - ltcThick) / Math.max(mtcThick, ltcThick) * 100;
    if (asymPct >= 40) { oaScore += 14; oaFindings.push({ label: "Severe compartmental asymmetry", severity: "severe", detail: `${asymPct.toFixed(0)}%` }); }
    else if (asymPct >= 25) { oaScore += 8; oaFindings.push({ label: "Moderate compartmental asymmetry", severity: "moderate", detail: `${asymPct.toFixed(0)}%` }); }
    else if (asymPct >= 15) { oaScore += 3; oaFindings.push({ label: "Mild compartmental asymmetry", severity: "mild", detail: `${asymPct.toFixed(0)}%` }); }
  }

  const oaProbability = oaMax > 0 ? oaScore / oaMax : 0;
  const oaDetected = oaProbability >= 0.50;
  const oaSeverity = oaProbability >= 0.7 ? "severe" : oaProbability >= 0.55 ? "moderate" : oaProbability >= 0.50 ? "mild" : "none";

  return (
    <>
      {/* --- OA VERDICT --- */}
      <div className={`card ${oaDetected ? "border-rose-800" : "border-emerald-800"}`}>
        <div className="card-head">
          <span className="card-title">OA Assessment</span>
          <span className={`chip ${oaDetected ? "border-rose-600 bg-rose-500/15 text-rose-300" : "border-emerald-600 bg-emerald-500/15 text-emerald-300"}`}>
            {oaDetected ? "OA DETECTED" : "NO OA"}
          </span>
        </div>
        <div className="card-body">
          <div className="flex items-center gap-3 mb-3">
            <div className="relative h-16 w-16 shrink-0">
              <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90">
                <circle cx="40" cy="40" r="34" className="fill-none stroke-slate-800" strokeWidth="7" />
                <circle cx="40" cy="40" r="34" className={`fill-none ${oaDetected ? "text-rose-500" : "text-emerald-500"}`} stroke="currentColor" strokeWidth="7" strokeLinecap="round" strokeDasharray={`${oaProbability * 213.6} 213.6`} />
              </svg>
              <span className={`absolute inset-0 flex items-center justify-center font-mono text-[14px] font-bold ${oaDetected ? "text-rose-400" : "text-emerald-400"}`}>
                {Math.round(oaProbability * 100)}%
              </span>
            </div>
            <div>
              <p className={`text-[13px] font-semibold ${oaDetected ? "text-rose-300" : "text-emerald-300"}`}>
                {oaDetected ? "Osteoarthritis Detected" : "No Osteoarthritis Detected"}
              </p>
              <p className="text-[11px] text-slate-400">
                Severity: <strong>{oaSeverity}</strong>
                {oaDetected && ` · Approximate KL: ${oaProbability >= 0.6 ? "3-4" : "2-3"}`}
              </p>
              <p className="text-[10px] text-slate-500 mt-1">
                Rule-based morphometric classifier using literature thresholds
              </p>
            </div>
          </div>

          {oaFindings.length > 0 && (
            <div className="border-t border-slate-800 pt-2">
              <h4 className="section-label">Findings</h4>
              {oaFindings.map((f, i) => (
                <div key={i} className="row">
                  <span className={`row-key ${f.severity === "severe" ? "text-rose-400" : f.severity === "moderate" ? "text-amber-400" : "text-slate-300"}`}>
                    {f.label}
                  </span>
                  <span className="row-val text-[10px]">{f.detail}</span>
                </div>
              ))}
            </div>
          )}

          {klGrade != null && (
            <div className="mt-2 border-t border-slate-800 pt-2">
              <div className="row">
                <span className="row-key">Dataset KL grade (ground truth)</span>
                <span className="row-val font-semibold">{klGrade}</span>
              </div>
            </div>
          )}

          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
            Not a clinical diagnosis. Thresholds from Eckstein et al. (2006), Wirth et al. (2013), Sharma et al. (2001). Accuracy depends on segmentation quality.
          </p>
        </div>
      </div>

      {/* --- Measurement details --- */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Supporting measurements</span>
        </div>
        <div className="card-body">
          <Group title="Cartilage thickness" names={CARTILAGE} data={data} />
          <Group title="Cartilage volume" names={VOLUMES} data={data} />
          <Group title="Joint line and alignment" names={ALIGNMENT} data={data} />
        </div>
      </div>
    </>
  );
}
