"use client";

/**
 * The OA verdict, and the geometry behind it.
 *
 * Two things are shown, in this order, and the order is the point:
 *
 * 1. **The answer.** A trained classifier (`kneeai.oa`) says osteoarthritis or not,
 *    with a probability and a severity band.
 * 2. **Why, and how much to trust it.** The per-feature contributions to the
 *    decision, and the classifier's own held-out performance, rendered at the same
 *    visual weight as the answer — not buried in a tooltip.
 *
 * Point 2 is not decoration. The classifier was fitted to 48 knees and validated on
 * 12. A probability from a model that small is a weak signal, and a panel that shows
 * "87% — osteoarthritis detected" without also showing the validation set size is
 * actively misleading. So `reference_metrics` renders next to the verdict, and where
 * the dataset supplies its own KL grade the two sit side by side so the reader can
 * see the model agree or disagree on this very case.
 */

import type { Measurement, MeasurementSet, OaAssessment, OaIndicator } from "@/lib/types";

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

const SEVERITY_STYLE: Record<string, string> = {
  none: "border-emerald-600 bg-emerald-500/15 text-emerald-300",
  doubtful: "border-sky-600 bg-sky-500/15 text-sky-300",
  mild: "border-amber-600 bg-amber-500/15 text-amber-300",
  moderate: "border-orange-600 bg-orange-500/15 text-orange-300",
  severe: "border-rose-600 bg-rose-500/15 text-rose-300",
};

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

function Group({ title, names, data }: { title: string; names: string[]; data: MeasurementSet }) {
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

/** One feature's push towards or away from an OA call. */
function IndicatorBar({ indicator, max }: { indicator: OaIndicator; max: number }) {
  const magnitude = Math.abs(indicator.log_odds_contribution);
  const width = max > 0 ? (magnitude / max) * 100 : 0;
  const towards = indicator.direction === "towards_oa";
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-[11px] text-slate-300" title={indicator.name}>
          {indicator.display_name}
        </span>
        <span className="shrink-0 font-mono text-[11px] text-slate-100">
          {indicator.value == null ? "—" : indicator.value.toFixed(2)}
          {indicator.unit === "mm" ? " mm" : ""}
          {indicator.imputed && (
            <span className="ml-1 text-[9px] text-amber-500" title="not computable; training median used">
              imputed
            </span>
          )}
        </span>
      </div>
      {/* Centre line: bars grow right for "towards OA", left for "away". */}
      <div className="mt-1 flex h-1.5 items-center">
        <div className="flex h-full w-1/2 justify-end overflow-hidden">
          {!towards && (
            <div className="h-full rounded-l bg-emerald-500" style={{ width: `${width}%` }} />
          )}
        </div>
        <div className="h-full w-px shrink-0 bg-slate-600" />
        <div className="flex h-full w-1/2 overflow-hidden">
          {towards && (
            <div className="h-full rounded-r bg-rose-500" style={{ width: `${width}%` }} />
          )}
        </div>
      </div>
    </div>
  );
}

function MetricsTable({ metrics, label }: { metrics: Record<string, any>; label: string }) {
  if (!metrics) return null;
  const rows: Array<[string, string]> = [
    ["Cases", `${metrics.n_cases} (${metrics.n_positive} with OA)`],
    ["Balanced accuracy", `${(metrics.balanced_accuracy * 100).toFixed(0)}%`],
    ["Sensitivity", `${(metrics.sensitivity * 100).toFixed(0)}%`],
    ["Specificity", `${(metrics.specificity * 100).toFixed(0)}%`],
    ["AUROC", Number.isFinite(metrics.auroc) ? metrics.auroc.toFixed(2) : "—"],
  ];
  return (
    <div className="mb-2">
      <h4 className="section-label">{label}</h4>
      {rows.map(([k, v]) => (
        <div key={k} className="row">
          <span className="row-key">{k}</span>
          <span className="row-val">{v}</span>
        </div>
      ))}
    </div>
  );
}

export default function OaPanel({
  data,
  oa,
  klGrade,
}: {
  data: MeasurementSet | null;
  /** The trained classifier's assessment. */
  oa?: OaAssessment | null;
  /** Dataset-provided Kellgren-Lawrence grade, when the cohort supplies one. */
  klGrade?: number | null;
}) {
  const maxContribution = oa?.indicators?.length
    ? Math.max(...oa.indicators.map((i) => Math.abs(i.log_odds_contribution)))
    : 0;

  const detected = oa?.oa_detected;
  const severityStyle =
    (oa?.severity && SEVERITY_STYLE[oa.severity]) ??
    "border-slate-600 bg-slate-700/40 text-slate-300";

  return (
    <>
      {/* ---------- The verdict ---------- */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Osteoarthritis assessment</span>
          <span className="chip border-amber-600 bg-amber-500/15 text-amber-300">
            not a diagnosis
          </span>
        </div>

        <div className="card-body">
          {!oa || !oa.available ? (
            <>
              <p className="text-[12px] font-medium text-slate-300">Not available</p>
              <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
                {oa?.unavailable_reason ??
                  "No OA assessment was produced for this case. Re-run the pipeline to generate one."}
              </p>
            </>
          ) : (
            <>
              <div className="flex items-center gap-3">
                <div
                  className={`flex h-14 w-14 shrink-0 flex-col items-center justify-center rounded-full border-2 ${
                    detected
                      ? "border-rose-500 bg-rose-500/10"
                      : "border-emerald-500 bg-emerald-500/10"
                  }`}
                >
                  <span
                    className={`font-mono text-[15px] font-semibold ${
                      detected ? "text-rose-300" : "text-emerald-300"
                    }`}
                  >
                    {oa.probability == null ? "—" : `${Math.round(oa.probability * 100)}%`}
                  </span>
                </div>
                <div className="min-w-0">
                  <p
                    className={`text-[14px] font-semibold ${
                      detected ? "text-rose-300" : "text-emerald-300"
                    }`}
                  >
                    {detected ? "Osteoarthritis detected" : "No osteoarthritis detected"}
                  </p>
                  <p className="mt-0.5 text-[11px] leading-snug text-slate-400">
                    Probability of KL&nbsp;≥&nbsp;{oa.kl_threshold_for_oa} (definite
                    radiographic OA), at a decision threshold of{" "}
                    {oa.threshold.toFixed(2)}.
                  </p>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                {oa.severity && (
                  <span className={`chip ${severityStyle}`}>severity: {oa.severity}</span>
                )}
                {oa.predicted_kl_grade != null && (
                  <span className="chip border-slate-600 bg-slate-700/40 text-slate-300">
                    predicted KL {oa.predicted_kl_grade.toFixed(1)}
                  </span>
                )}
                {klGrade != null && (
                  <span className="chip border-sky-700 bg-sky-500/10 text-sky-300">
                    dataset KL {klGrade}
                  </span>
                )}
              </div>

              {klGrade != null && oa.predicted_kl_grade != null && (
                <p className="mt-2 text-[11px] leading-relaxed text-slate-400">
                  The dataset&apos;s own radiographic grade for this knee is{" "}
                  <strong>{klGrade}</strong>; the model predicted{" "}
                  <strong>{oa.predicted_kl_grade.toFixed(1)}</strong>. The dataset grade is
                  a label and was never an input — on a case from the training split this
                  agreement is not evidence of anything.
                </p>
              )}

              {oa.features?.full_thickness_loss?.length > 0 && (
                <div className="mt-2 rounded-md border border-rose-800 bg-rose-950/30 px-2.5 py-2">
                  <p className="text-[11px] leading-relaxed text-rose-300">
                    Full-thickness cartilage loss recorded in:{" "}
                    {oa.features.full_thickness_loss
                      .map((s) => s.replace(/_/g, " "))
                      .join(", ")}
                    .
                  </p>
                </div>
              )}

              {oa.warnings?.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {oa.warnings.map((w, i) => (
                    <li key={i} className="text-[11px] leading-relaxed text-amber-400">
                      {w}
                    </li>
                  ))}
                </ul>
              )}

              <p className="mt-3 border-t border-slate-800 pt-2 text-[10px] leading-relaxed text-slate-500">
                {oa.disclaimer}
              </p>
            </>
          )}
        </div>
      </div>

      {/* ---------- Why ---------- */}
      {oa?.available && oa.indicators?.length > 0 && (
        <div className="card">
          <div className="card-head">
            <span className="card-title">What drove the answer</span>
            <span className="card-sub">log-odds contribution</span>
          </div>
          <div className="card-body">
            <div className="mb-2 flex items-center justify-between text-[10px] text-slate-500">
              <span>← away from OA</span>
              <span>towards OA →</span>
            </div>
            {oa.indicators.map((indicator) => (
              <IndicatorBar key={indicator.name} indicator={indicator} max={maxContribution} />
            ))}
            <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
              This is a linear model, so each bar is exactly the feature&apos;s
              standardised value times its coefficient. The bars sum to the decision;
              nothing is approximated.
            </p>
          </div>
        </div>
      )}

      {/* ---------- How much to trust it ---------- */}
      {oa?.available && oa.reference_metrics && (
        <div className="card border-slate-700">
          <div className="card-head">
            <span className="card-title">Classifier performance</span>
            <span className="card-sub">measured, held out</span>
          </div>
          <div className="card-body">
            {oa.reference_metrics.validation && (
              <MetricsTable
                metrics={oa.reference_metrics.validation}
                label="Validation split — ground-truth masks"
              />
            )}
            {oa.reference_metrics.validation_on_predicted_masks && (
              <MetricsTable
                metrics={oa.reference_metrics.validation_on_predicted_masks}
                label="Validation split — model's own masks"
              />
            )}
            <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
              The second table is the one that matters in production: it measures the
              classifier on segmentations the model produced, not on expert annotation.
              The gap between the two is the cost of imperfect cartilage segmentation.
            </p>
            <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
              These describe the validation cohort, not the case on screen. With a
              validation set this small the confidence interval on every figure above is
              wide and unquantified.
            </p>
          </div>
        </div>
      )}

      {/* ---------- The geometry ---------- */}
      {data ? (
        <div className="card">
          <div className="card-head">
            <span className="card-title">Underlying geometry</span>
            <span className="card-sub">{data.n_measurements} measured</span>
          </div>
          <div className="card-body">
            <p className="mb-3 text-[11px] leading-relaxed text-slate-400">
              The quantities a clinician grades OA from, measured by the deterministic
              engine. These are the classifier&apos;s inputs, shown here in their own
              units with the engine&apos;s reference ranges attached.
            </p>
            <Group title="Cartilage thickness" names={CARTILAGE} data={data} />
            <Group title="Cartilage volume" names={VOLUMES} data={data} />
            <Group title="Joint line and alignment" names={ALIGNMENT} data={data} />
          </div>
        </div>
      ) : (
        <div className="card">
          <div className="card-head">
            <span className="card-title">Underlying geometry</span>
          </div>
          <div className="card-body">
            <p className="text-[12px] text-slate-500">
              Requires measurements. Run the pipeline for this case.
            </p>
          </div>
        </div>
      )}

      {/* ---------- Known gaps ---------- */}
      <div className="card border-slate-700">
        <div className="card-head">
          <span className="card-title">What this assessment still lacks</span>
        </div>
        <div className="card-body">
          <ul className="space-y-1.5 text-[11px] leading-relaxed text-slate-400">
            <li>
              <strong>Scale.</strong> Fitted on 48 knees, validated on 12. That is a
              feasibility result, not a validated classifier.
            </li>
            <li>
              <strong>Label proxy.</strong> It predicts a radiographic KL grade from MR
              cartilage geometry. KL is graded on osteophytes and joint space narrowing
              from a projection — a correlated but different measurement.
            </li>
            <li>
              <strong>Joint-space width under load.</strong> MRI is unloaded, so it
              systematically understates narrowing.
            </li>
            <li>
              <strong>Cartilage segmentation quality.</strong> Tibial cartilage Dice is
              the weakest part of the segmentation, and it carries most of the OA signal.
            </li>
            <li>
              <strong>No external cohort.</strong> Never tested against radiologist reads
              outside this dataset.
            </li>
          </ul>
        </div>
      </div>
    </>
  );
}
