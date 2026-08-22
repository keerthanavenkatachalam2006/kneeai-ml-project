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

  return (
    <>
      <div className="card">
        <div className="card-head">
          <span className="card-title">OA indicators</span>
          <span className="chip border-amber-600 bg-amber-500/15 text-amber-300">
            not a diagnosis
          </span>
        </div>

        <div className="card-body">
          <p className="mb-3 text-[11px] leading-relaxed text-slate-400">
            Quantitative geometry only. No OA grade is predicted — that needs a trained
            classifier and clinical validation, and this platform has neither. The
            numbers below are the inputs a clinician grades from, not the grade.
          </p>

          {klGrade != null && (
            <div className="mb-3 rounded-md border border-slate-700 bg-slate-900 px-2.5 py-2">
              <div className="row !border-0">
                <span className="row-key">Kellgren-Lawrence grade</span>
                <span className="row-val">{klGrade}</span>
              </div>
              <p className="mt-0.5 text-[10px] leading-snug text-slate-500">
                From the dataset&apos;s own radiographic reading. A <strong>label</strong>,
                not a model output — nothing here predicted it.
              </p>
            </div>
          )}

          {/* Compartment comparison: the pattern that matters most in knee OA. */}
          {thinner && difference != null && (
            <div className="mb-3">
              <h4 className="section-label">Compartment comparison</h4>
              <div className="flex items-end gap-2">
                {(
                  [
                    ["Medial", medialValue!, thinner === "medial"],
                    ["Lateral", lateralValue!, thinner === "lateral"],
                  ] as Array<[string, number, boolean]>
                ).map(([label, value, isThinner]) => {
                  const maxValue = Math.max(medialValue!, lateralValue!, 0.001);
                  return (
                    <div key={label} className="flex-1">
                      <div className="mb-1 flex items-baseline justify-between">
                        <span className="text-[11px] text-slate-400">{label}</span>
                        <span
                          className={`font-mono text-[11px] ${isThinner ? "text-amber-400" : "text-slate-200"}`}
                        >
                          {value.toFixed(2)} mm
                        </span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded bg-slate-800">
                        <div
                          className={isThinner ? "h-full bg-amber-500" : "h-full bg-sky-500"}
                          style={{ width: `${(value / maxValue) * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="mt-1.5 text-[11px] leading-relaxed text-slate-400">
                Tibial cartilage is <strong>{difference.toFixed(2)} mm thinner</strong> in
                the {thinner} compartment
                {asymmetry && ` (${asymmetry.formatted} asymmetry)`}. Medial predominance
                is the usual pattern in knee OA, but thickness alone does not establish
                it, and this measurement carries the segmentation&apos;s error with it.
              </p>
            </div>
          )}

          <Group title="Cartilage thickness" names={CARTILAGE} data={data} />
          <Group title="Cartilage volume" names={VOLUMES} data={data} />
          <Group title="Joint line and alignment" names={ALIGNMENT} data={data} />
          <Group title="Meniscus extrusion" names={MENISCUS} data={data} />

          {flagged.length > 0 && (
            <div className="mt-3 border-t border-slate-800 pt-2">
              <h4 className="section-label text-amber-400">
                Outside reference range ({flagged.length})
              </h4>
              {flagged.map((m) => (
                <div key={m.name} className="row">
                  <span className="row-key">{m.display_name}</span>
                  <span className="row-val text-amber-400">{m.formatted}</span>
                </div>
              ))}
              <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                Reference ranges here are deliberately wide population figures, not
                diagnostic thresholds. Falling outside one is a prompt to look, not a
                finding.
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="card border-slate-700">
        <div className="card-head">
          <span className="card-title">What would make this a real OA assessment</span>
        </div>
        <div className="card-body">
          <ul className="space-y-1.5 text-[11px] leading-relaxed text-slate-400">
            <li>
              A trained pathology classifier. The dataset carries KL grades, so the labels
              exist — the model does not.
            </li>
            <li>
              Joint-space width from a weight-bearing radiograph. MRI is unloaded, so it
              systematically understates narrowing.
            </li>
            <li>
              Cartilage segmentation good enough to trust a thickness to a tenth of a
              millimetre. Current validation Dice on the tibial cartilages is around 0.45.
            </li>
            <li>Clinical validation against radiologist reads on an independent cohort.</li>
          </ul>
        </div>
      </div>
    </>
  );
}
