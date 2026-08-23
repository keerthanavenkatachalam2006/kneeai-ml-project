"use client";

/**
 * Surgeon dashboard.
 *
 * Layout: the 3D knee is permanently on the left at roughly two-thirds width, and a
 * single context panel on the right changes with the workflow step. The previous
 * three-column version showed six panels at once, which meant a viewer had to work
 * out for themselves what to look at; here the step decides, and the anatomy is
 * always in view because it is the thing every stage is talking about.
 *
 * The viewer is dynamically imported with SSR off: three.js touches `window` at
 * module scope and would fail the server render.
 */

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import FloatingPanel from "@/components/FloatingPanel";
import ImplantPanel from "@/components/ImplantPanel";
import ImplantPreview3D from "@/components/ImplantPreview3D";
import ImplantSizeSelector from "@/components/ImplantSizeSelector";
import MeasurementsPanel from "@/components/MeasurementsPanel";
import OaPanel from "@/components/OaPanel";
import ReadinessPanel from "@/components/ReadinessPanel";
import RobotPanel from "@/components/RobotPanel";
import Slice2D from "@/components/Slice2D";
import SourceBadge from "@/components/SourceBadge";
import Stepper, { type Step } from "@/components/Stepper";
import UploadPanel from "@/components/UploadPanel";
import ViewerControls from "@/components/ViewerControls";
import type { PickInfo, ViewerSettings } from "@/components/Viewer3D";
import { api, ApiError, fmt } from "@/lib/api";
import type {
  CaseListEntry,
  CaseSummary,
  ModelInfo,
  RobotCommandSequence,
  SimulationStateInfo,
} from "@/lib/types";

const Viewer3D = dynamic(() => import("@/components/Viewer3D"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-[12px] text-slate-500">
      Loading 3D viewer…
    </div>
  ),
});

const ANIMATION_MS = 10000;

/** Panel heading per workflow step. */
const STEP_TITLES: Record<string, string> = {
  upload: "Upload scan",
  case: "Patient case",
  segment: "AI segmentation",
  slices: "2D slices",
  anatomy: "3D anatomy",
  measure: "Measurements",
  oa: "OA analysis",
  plan: "Implant & plan",
  readiness: "Robotic readiness",
  simulate: "Approve & simulate",
};

/**
 * Open the robot cell as a real second window. Named so repeated clicks focus the
 * existing one instead of stacking copies.
 */
function openRobotWindow(caseId: string) {
  window.open(
    `/robot?case=${encodeURIComponent(caseId)}`,
    "morp-robot-cell",
    "popup=yes,width=1320,height=880",
  );
}

function baseSettings(): ViewerSettings {
  return {
    visible: {},
    opacity: {},
    showLandmarks: false,
    showCutPlanes: false,
    showImplant: false,
    showBoundary: false,
    showTrajectory: false,
    clipEnabled: false,
    clipPosition: 0,
    clipAxis: 0,
    resectedView: false,
  };
}

export default function Dashboard() {
  const [cases, setCases] = useState<CaseListEntry[]>([]);
  const [caseId, setCaseId] = useState<string | null>(null);
  const [summary, setSummary] = useState<CaseSummary | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [simState, setSimState] = useState<SimulationStateInfo | null>(null);
  const [commands, setCommands] = useState<RobotCommandSequence | null>(null);
  const [settings, setSettings] = useState<ViewerSettings>(baseSettings);
  const [progress, setProgress] = useState<number | null>(null);
  const [step, setStep] = useState("case");
  const [pick, setPick] = useState<PickInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  /**
   * Size being previewed in the standalone implant viewer. Kept out of
   * `ViewerSettings` on purpose: the main scene always shows the planned component, so
   * the geometry on the bone stays the geometry the plan describes.
   */
  const [sizePreview, setSizePreview] = useState({
    femoral: { ml: 1, ap: 1 },
    tibial: { ml: 1, ap: 1 },
    sizeLabel: "—",
    isPlanned: true,
  });
  const [loading, setLoading] = useState(true);
  const [viewerKey, setViewerKey] = useState(0);

  const glRef = useRef<{ gl: any; scene: any; camera: any } | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [list, modelInfo] = await Promise.all([api.listCases(), api.model()]);
        if (!alive) return;
        setCases(list.cases);
        setModel(modelInfo);
        const best =
          list.cases.find((c) => c.is_ai && c.has_plan && c.has_meshes) ??
          list.cases.find((c) => c.has_plan && c.has_meshes) ??
          list.cases[0];
        setCaseId(best?.case_id ?? null);
      } catch (e) {
        if (alive) setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!caseId) return;
    let alive = true;
    setProgress(null);
    setCommands(null);
    setStep("case");
    (async () => {
      try {
        const [detail, state] = await Promise.all([
          api.case(caseId),
          api.simulationState(caseId).catch(() => null),
        ]);
        if (!alive) return;
        setSummary(detail);
        setSimState(state);
        // Keyed by path: the manifest has three entries called `femur`
        // (anatomy / resected / component), so `structure` is not unique.
        const next = baseSettings();
        for (const mesh of detail.meshes?.meshes ?? []) {
          // Only show anatomy by default. Resected bone and implant components
          // are shown when the user navigates to the plan/simulate steps.
          const isAnatomy = mesh.category === "anatomy";
          next.visible[mesh.path] = isAnatomy;
          next.opacity[mesh.path] = /cartilage/.test(mesh.structure)
            ? 0.7
            : /meniscus/.test(mesh.structure)
              ? 0.78
              : 1;
        }
        setSettings(next);
        if (detail.plan) {
          api.robotCommands(caseId).then((c) => alive && setCommands(c)).catch(() => undefined);
        }
        setError(null);
      } catch (e) {
        if (alive) setError(e instanceof ApiError ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [caseId]);

  // --- OA gate ---------------------------------------------------------------
  // A total knee replacement is indicated for osteoarthritis. Everything
  // downstream of that decision — implant sizing, resection planes, simulation —
  // is gated on it, so the score is computed here once and read everywhere.
  const oaStatus = useMemo(() => {
    const meas = summary?.measurements;
    if (!meas) return { detected: false, probability: 0, known: false };

    const findVal = (name: string) =>
      meas.measurements.find((m) => m.name === name)?.quantity.value ?? null;

    const fc = findVal("femoral_cartilage_mean_thickness");
    const mtc = findVal("medial_tibial_cartilage_mean_thickness");
    const ltc = findVal("lateral_tibial_cartilage_mean_thickness");
    const jlca = findVal("jlca");

    let score = 0;
    let max = 0;
    if (mtc != null) {
      max += 22;
      if (mtc < 0.9) score += 22;
      else if (mtc < 1.4) score += 14;
      else if (mtc < 1.6) score += 6;
    }
    if (fc != null) {
      max += 18;
      if (fc < 1.2) score += 18;
      else if (fc < 1.8) score += 11;
      else if (fc < 2.1) score += 4;
    }
    if (ltc != null) {
      max += 12;
      if (ltc < 1.0) score += 12;
      else if (ltc < 1.5) score += 7;
    }
    if (jlca != null) {
      max += 18;
      const a = Math.abs(jlca);
      if (a >= 8) score += 18;
      else if (a >= 5) score += 12;
      else if (a >= 3) score += 5;
    }
    if (mtc != null && ltc != null && ltc > 0) {
      max += 14;
      const asym = (Math.abs(mtc - ltc) / Math.max(mtc, ltc)) * 100;
      if (asym >= 40) score += 14;
      else if (asym >= 25) score += 8;
      else if (asym >= 15) score += 3;
    }

    const probability = max > 0 ? score / max : 0;
    return { detected: probability >= 0.5, probability, known: max > 0 };
  }, [summary?.measurements]);

  // Each step turns on the overlays it is about, and off the ones it isn't. The
  // viewer should always be showing what the current panel is discussing.
  useEffect(() => {
    setSettings((s) => {
      // Implant/resection views are gated on OA: a knee replacement is only
      // indicated for an osteoarthritic knee, so showing the hardware on a
      // healthy knee is clinically meaningless.
      const showPlan = (step === "plan" || step === "simulate") && oaStatus.detected;
      const updated = { ...s };



      if (summary?.meshes) {
        for (const mesh of summary.meshes.meshes) {
          const isSoft = /cartilage|meniscus/.test(mesh.structure);
          if (mesh.category === "anatomy") {
            // In the plan view the cartilage is gone: a TKA resects it. Keeping it on
            // screen would also drown the component in segmentation fragments, since
            // the cartilage masks scatter over the full volume.
            updated.visible[mesh.path] = !showPlan;
          } else if (mesh.category === "resected" || mesh.category === "component") {
            // Never visible without OA.
            updated.visible[mesh.path] = showPlan;
          }
        }
      }

      return {
        ...updated,
        showLandmarks: step === "anatomy" || step === "measure",
        showCutPlanes: showPlan,
        showImplant: showPlan,
        showBoundary: step === "simulate" && Boolean(commands) && oaStatus.detected,
        showTrajectory: step === "simulate" && Boolean(commands) && oaStatus.detected,
        resectedView: showPlan,
      };
    });
  }, [step, commands, summary?.meshes, oaStatus.detected]);

  const animate = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ANIMATION_MS);
      setProgress(t);
      if (t > 0.85) setSettings((s) => (s.resectedView ? s : { ...s, resectedView: true }));
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  const activeCommand = useMemo(() => {
    if (progress === null || !commands) return null;
    const posed = commands.commands.filter((c) => c.pose);
    return posed.length ? Math.min(posed.length - 1, Math.floor(progress * posed.length)) : null;
  }, [progress, commands]);

  const screenshot = useCallback(() => {
    const ctx = glRef.current;
    if (!ctx) return;
    ctx.gl.render(ctx.scene, ctx.camera);
    const a = document.createElement("a");
    a.href = ctx.gl.domElement.toDataURL("image/png");
    a.download = `${caseId ?? "case"}.png`;
    a.click();
  }, [caseId]);

  const result = summary?.result;
  const dice = model?.reference_metrics?.dice_per_structure ?? null;

  const steps: Step[] = useMemo(
    () => [
      { id: "upload", label: "Upload scan", hint: "Upload MRI for analysis", ready: true },
      { id: "case", label: "Patient case", hint: "Study, model, provenance", ready: true },
      {
        id: "segment",
        label: "AI segmentation",
        hint: `${result?.segmentation.structures_present.length ?? 0} structures identified`,
        ready: Boolean(summary),
      },
      {
        id: "slices",
        label: "2D slices",
        hint: "Axial, coronal and sagittal with contours",
        ready: Boolean(summary),
      },
      {
        id: "anatomy",
        label: "3D anatomy",
        hint: "Reconstructed surfaces and landmarks",
        ready: Boolean(summary?.meshes),
        blockedReason: "No meshes — run the reconstruction pipeline",
      },
      {
        id: "measure",
        label: "Measurements",
        hint: `${summary?.measurements?.n_measurements ?? 0} computed`,
        ready: Boolean(summary?.measurements),
        blockedReason: "No measurements for this case",
      },
      {
        id: "oa",
        label: "OA analysis",
        hint: "Trained classifier + morphometric indicators",
        ready: Boolean(summary?.measurements),
        blockedReason: "Requires measurements",
      },
      {
        id: "plan",
        label: "Implant & plan",
        hint: oaStatus.detected
          ? `${summary?.plan?.cuts.length ?? 0} resection cuts`
          : "Not indicated — no OA detected",
        ready: Boolean(summary?.plan) && oaStatus.detected,
        blockedReason: !oaStatus.detected
          ? "Knee replacement is only indicated for osteoarthritic knees. This knee shows no OA."
          : "No surgical plan for this case",
      },
      {
        id: "readiness",
        label: "Robotic readiness",
        hint: oaStatus.detected ? "Execution readiness score" : "Not indicated — no OA",
        ready: Boolean(summary) && oaStatus.detected,
        blockedReason: !oaStatus.detected
          ? "Readiness scoring applies to a surgical plan, which is not indicated without OA."
          : "No case loaded",
      },
      {
        id: "simulate",
        label: "Approve & simulate",
        hint: oaStatus.detected ? "Surgeon gate, then robot simulation" : "Not indicated — no OA",
        ready: Boolean(summary?.plan) && oaStatus.detected,
        blockedReason: !oaStatus.detected
          ? "Simulation requires a surgical plan, which is not indicated without OA."
          : "Requires a surgical plan",
      },
    ],
    [summary, result, oaStatus],
  );

  return (
    <div className="flex h-[calc(100vh-1.75rem)] flex-col">
      {/* Header */}
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="flex h-8 w-8 items-center justify-center rounded-full bg-teal-400/15 ring-1 ring-inset ring-teal-400/30"
          >
            <span className="h-3 w-3 rounded-full bg-teal-400" />
          </span>
          <div>
            <h1 className="text-[15px] font-semibold leading-tight tracking-tight text-white">
              MORP
              <span className="ml-2 font-normal text-teal-200/45">Knee planning workspace</span>
            </h1>
            <span className="hidden text-[10px] leading-tight text-teal-200/40 md:block">
              MRI → segmentation → 3D → measurement → plan → simulation
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2.5">
          {summary && <SourceBadge source={summary.source} />}
          <select
            aria-label="Select case"
            value={caseId ?? ""}
            onChange={(e) => setCaseId(e.target.value)}
            className="rounded-full border border-white/10 bg-clinic-900/80 px-3.5 py-1.5 text-[12px] text-teal-50 transition-colors hover:border-white/20 focus:border-teal-400"
          >
            {cases.map((c) => (
              <option key={c.case_id} value={c.case_id} className="bg-clinic-900">
                {c.case_id} — {c.badge}
                {c.has_plan ? "" : " (no plan)"}
              </option>
            ))}
          </select>
        </div>
      </header>

      <Stepper steps={steps} current={step} onSelect={setStep} />

      {loading && <p className="px-4 py-6 text-[12px] text-slate-500">Loading…</p>}

      {error && (
        <div role="alert" className="mx-4 mt-3 rounded-md border border-rose-800 bg-rose-950/40 px-3 py-2.5">
          <p className="text-[12px] text-rose-300">{error}</p>
          <p className="mt-1 text-[11px] text-slate-400">
            Is the API running? <code className="font-mono">make api</code> starts it on
            127.0.0.1:8000.
          </p>
        </div>
      )}

      {(summary || step === "upload") && (
        <div className="relative min-h-0 flex-1 p-3">
          {/* ---------- Viewer: full bleed, panels float over it ---------- */}
          <section className="relative h-full min-h-[26rem] overflow-hidden rounded-panel border border-white/5 bg-clinic-950 shadow-panel">
            {step === "upload" && !summary?.meshes ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
                <p className="text-[14px] font-medium text-white">Upload a knee MRI</p>
                <p className="max-w-sm text-[11px] leading-relaxed text-teal-100/50">
                  Drop a NIfTI (.nii.gz), DICOM zip, or single DICOM file into the upload
                  panel. Segmentation, measurement and planning run automatically.
                </p>
              </div>
            ) : summary?.meshes ? (
              <>
                <Viewer3D
                  key={viewerKey}
                  caseId={summary.case_id}
                  manifest={summary.meshes}
                  landmarks={summary.landmarks?.landmarks ?? []}
                  cuts={summary.plan?.cuts ?? []}
                  commands={commands}
                  settings={settings}
                  animationProgress={progress}
                  onPick={setPick}
                  onCanvasReady={(gl, scene, camera) => {
                    glRef.current = { gl, scene, camera };
                  }}
                />

                <div className="pointer-events-none absolute inset-x-3 top-3 flex justify-center">
                  <div className="pointer-events-auto">
                    <ViewerControls
                      manifest={summary.meshes}
                      settings={settings}
                      onChange={setSettings}
                      onScreenshot={screenshot}
                      onReset={() => setViewerKey((k) => k + 1)}
                      hasCuts={Boolean(summary.plan?.cuts.length)}
                      hasCommands={Boolean(commands)}
                    />
                  </div>
                </div>

                <div className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-slate-950/70 px-2 py-1 text-[10px] text-slate-500 backdrop-blur">
                  {summary.meshes.meshes.length} meshes ·{" "}
                  {summary.meshes.total_faces.toLocaleString()} faces · millimetres, image frame
                </div>

                {progress !== null && (
                  <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-slate-950/90 px-4 py-2 backdrop-blur">
                    <div className="mb-1.5 flex items-center justify-between text-[11px]">
                      <span className="font-mono text-sky-400">
                        {activeCommand !== null && commands
                          ? commands.commands.filter((c) => c.pose)[activeCommand]?.action
                          : "…"}
                      </span>
                      <span className="text-slate-500">{Math.round(progress * 100)}%</span>
                    </div>
                    <div className="h-1 w-full overflow-hidden rounded bg-slate-800">
                      <div className="h-full bg-sky-500" style={{ width: `${progress * 100}%` }} />
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-2 px-8 text-center">
                <p className="text-[13px] text-slate-300">No 3D geometry for this case</p>
                <p className="max-w-sm text-[11px] leading-relaxed text-slate-500">
                  The segmentation exists but has not been reconstructed. Run{" "}
                  <code className="font-mono text-slate-400">
                    kneeai infer --case-id {summary?.case_id ?? "..."}
                  </code>{" "}
                  to build meshes, measurements and a plan.
                </p>
              </div>
            )}
          </section>

          {/* ---------- Floating context panels ----------
              A pointer-events-none layer so the viewer stays draggable everywhere the
              panels are not; each panel re-enables pointer events for itself. */}
          <div className="pointer-events-none absolute inset-3 z-10">
            <FloatingPanel
              title={STEP_TITLES[step] ?? "Details"}
              subtitle={summary?.case_id}
              accent={step === "simulate" ? "amber" : "teal"}
              initial={{ x: 16, y: 16 }}
              width={step === "simulate" ? 420 : 360}
            >
              <div className="space-y-3 p-3">
            {/* Upload panel: always accessible from the upload step */}
            {step === "upload" && (
              <UploadPanel
                onComplete={(job) => {
                  if (job.status === "complete") {
                    setCaseId(job.case_id);
                    setStep("case");
                  }
                }}
                onCaseCreated={(newCaseId) => {
                  // Case created, but still processing
                }}
              />
            )}

            {step === "case" && summary && (
              <>
                <div className="card">
                  <div className="card-head">
                    <span className="card-title">Study</span>
                    <span className="card-sub">{result?.geometry.side} knee</span>
                  </div>
                  <div className="card-body">
                    <div className="row">
                      <span className="row-key">Case</span>
                      <span className="row-val">{summary.case_id}</span>
                    </div>
                    <div className="row">
                      <span className="row-key">Voxel spacing</span>
                      <span className="row-val">
                        {result?.geometry.spacing_mm.map((s) => s.toFixed(2)).join(" × ")} mm
                      </span>
                    </div>
                    <div className="row">
                      <span className="row-key">Volume</span>
                      <span className="row-val">{result?.geometry.image_shape.join(" × ")}</span>
                    </div>
                    <div className="row">
                      <span className="row-key">Orientation</span>
                      <span className="row-val">{result?.geometry.orientation}</span>
                    </div>
                    <div className="row">
                      <span className="row-key">Contract</span>
                      <span
                        className={`row-val ${summary.contract_valid ? "text-emerald-400" : "text-rose-400"}`}
                      >
                        {summary.contract_valid ? "valid" : "INVALID"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-head">
                    <span className="card-title">Where this geometry came from</span>
                    <SourceBadge source={summary.source} size="sm" />
                  </div>
                  <div className="card-body">
                    {summary.is_ai ? (
                      <>
                        <div className="row">
                          <span className="row-key">Model</span>
                          <span className="row-val">
                            {result?.model.model_id} {result?.model.version}
                          </span>
                        </div>
                        <div className="row">
                          <span className="row-key">Architecture</span>
                          <span className="row-val">{result?.model.architecture}</span>
                        </div>
                      </>
                    ) : (
                      <p className="text-[11px] leading-relaxed text-amber-400">
                        Not produced by a model. Nothing on this screen represents model
                        performance.
                      </p>
                    )}
                  </div>
                </div>

                {model?.available && dice && (
                  <div className="card">
                    <div className="card-head">
                      <span className="card-title">Model performance</span>
                      <span className="card-sub">
                        validation set, n={model.reference_metrics?.n_patients ?? "?"}
                      </span>
                    </div>
                    <div className="card-body">
                      <p className="mb-2 text-[11px] leading-relaxed text-amber-400">
                        Describes the model, <strong>not this patient&apos;s</strong> segmentation.
                      </p>
                      {Object.entries(dice).map(([s, v]) => (
                        <div key={s} className="row">
                          <span className="row-key">{s.replace(/_/g, " ")}</span>
                          <span
                            className={`row-val ${v < 0.6 ? "text-amber-400" : v > 0.85 ? "text-emerald-400" : ""}`}
                          >
                            {v.toFixed(3)}
                          </span>
                        </div>
                      ))}
                      <div className="row">
                        <span className="row-key font-semibold text-slate-300">Mean Dice</span>
                        <span className="row-val font-semibold">
                          {model.reference_metrics?.mean_dice_excl_background?.toFixed(3) ?? "—"}
                        </span>
                      </div>
                      {model.approved_for_clinical_use === false && (
                        <p className="mt-2 text-[11px] text-rose-400">
                          Not approved for clinical use.
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}

            {step === "segment" && (
              <>
                <div className="card">
                  <div className="card-head">
                    <span className="card-title">Structures identified</span>
                    <span className="card-sub">
                      {result?.segmentation.structures_present.length ?? 0}
                    </span>
                  </div>
                  <div className="card-body">
                    {result?.segmentation.structures_present.map((s) => {
                      const vol = result.confidence.volume_mm3[s];
                      const conf = result.confidence.per_structure[s];
                      return (
                        <div key={s} className="row">
                          <span className="row-key">{s.replace(/_/g, " ")}</span>
                          <span className="row-val">
                            {vol != null ? `${(vol / 1000).toFixed(1)} cm³` : "—"}
                            {conf != null && (
                              <span className="text-slate-500"> · {conf.toFixed(2)}</span>
                            )}
                          </span>
                        </div>
                      );
                    })}
                    <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                      Second figure is a confidence proxy, not accuracy.{" "}
                      {result?.confidence.method}
                    </p>
                  </div>
                </div>

                {(result?.segmentation.structures_expected_but_missing?.length ?? 0) > 0 && (
                  <div className="card border-amber-900/60">
                    <div className="card-head">
                      <span className="card-title text-amber-400">Not produced</span>
                    </div>
                    <div className="card-body">
                      <p className="text-[11px] leading-relaxed text-slate-300">
                        {result!.segmentation.structures_expected_but_missing
                          .map((s) => s.replace(/_/g, " "))
                          .join(", ")}
                      </p>
                      <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
                        The training dataset contains no labels for these, so the model
                        cannot output them. Shown rather than omitted.
                      </p>
                    </div>
                  </div>
                )}

                {result?.warnings && result.warnings.length > 0 && (
                  <div className="card">
                    <div className="card-head">
                      <span className="card-title">Pipeline warnings</span>
                      <span className="card-sub">{result.warnings.length}</span>
                    </div>
                    <div className="card-body space-y-2">
                      {result.warnings.map((w, i) => (
                        <p key={i} className="text-[11px] leading-relaxed text-amber-400">
                          {w}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {step === "anatomy" && summary?.meshes && (
              <>
                <div className="card">
                  <div className="card-head">
                    <span className="card-title">Reconstruction quality</span>
                    <span className="card-sub">{summary.meshes.meshes.length} surfaces</span>
                  </div>
                  <div className="card-body">
                    {summary.meshes.meshes
                      .filter((m) => m.category === "anatomy")
                      .map((m) => (
                        <div key={m.path} className="row">
                          <span className="row-key">{m.structure.replace(/_/g, " ")}</span>
                          <span className="row-val">
                            {m.n_faces.toLocaleString()} f
                            <span
                              className={m.is_watertight ? "text-emerald-400" : "text-amber-400"}
                            >
                              {" "}
                              {m.is_watertight ? "✓" : "!"}
                            </span>
                          </span>
                        </div>
                      ))}
                    <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                      A watertight surface encloses a well-defined volume; a non-watertight
                      one cannot be measured reliably.
                    </p>
                  </div>
                </div>

                {summary.landmarks && (
                  <div className="card">
                    <div className="card-head">
                      <span className="card-title">Landmarks</span>
                      <span className="card-sub">
                        {summary.landmarks.n_landmarks} · mean conf{" "}
                        {summary.landmarks.mean_confidence.toFixed(2)}
                      </span>
                    </div>
                    <div className="card-body">
                      <p className="mb-2 text-[11px] leading-relaxed text-slate-500">
                        Hover a marker in the viewer for its name and confidence. Green is
                        confident, red is not.
                      </p>
                      <div className="row">
                        <span className="row-key">Reference frame</span>
                        <span className="row-val">
                          {String((summary.frame as any)?.method ?? "—")}
                        </span>
                      </div>
                      <div className="row">
                        <span className="row-key">Surgeon-edited</span>
                        <span className="row-val">{summary.landmarks.n_surgeon_edited}</span>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            {step === "slices" && summary && <Slice2D caseId={summary.case_id} />}

            {step === "measure" && summary && <MeasurementsPanel data={summary.measurements} />}

            {step === "oa" && summary && (
              <OaPanel data={summary.measurements} klGrade={summary.kl_grade ?? null} />
            )}

            {step === "readiness" && (
              <ReadinessPanel
                readiness={oaStatus.detected ? ((summary as any)?.readiness ?? null) : null}
                reason={
                  oaStatus.detected
                    ? null
                    : "Readiness scores a surgical plan. No plan is generated for a knee without osteoarthritis, so there is nothing to score."
                }
              />
            )}

            {step === "plan" && !oaStatus.detected && (
              <div className="card border-emerald-800">
                <div className="card-head">
                  <span className="card-title text-emerald-300">
                    Implant planning not indicated
                  </span>
                  <span className="chip border-emerald-600 bg-emerald-500/15 text-emerald-300">
                    no OA
                  </span>
                </div>
                <div className="card-body">
                  <p className="text-[12px] leading-relaxed text-slate-300">
                    This knee shows no osteoarthritis
                    {oaStatus.known && ` (${Math.round(oaStatus.probability * 100)}% on the morphometric score)`}
                    , so a total knee replacement is not indicated. The implant, resection
                    planes and simulation are withheld rather than shown on a knee that
                    does not need them.
                  </p>
                  <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                    Review the OA analysis step for the cartilage geometry behind this
                    conclusion, and the measurements step for the full morphometry.
                  </p>
                </div>
              </div>
            )}

            {step === "plan" && oaStatus.detected && summary && (
              <>
                <ImplantSizeSelector
                  data={summary.implants}
                  onSelect={setSizePreview}
                />
                <ImplantPreview3D
                  caseId={summary.case_id}
                  manifest={summary.meshes}
                  scale={{ femoral: sizePreview.femoral, tibial: sizePreview.tibial }}
                  sizeLabel={sizePreview.sizeLabel}
                  isPlanned={sizePreview.isPlanned}
                />
                <ImplantPanel data={summary.implants} />
                {summary.plan && (
                  <div className="card">
                    <div className="card-head">
                      <span className="card-title">Resection plan</span>
                      <span className="card-sub">
                        v{summary.plan.version} · {summary.plan.strategy.replace(/_/g, " ")}
                      </span>
                    </div>
                    <div className="card-body">
                      {summary.plan.cuts.map((cut) => (
                        <div key={cut.name} className="border-b border-slate-800/60 py-2 last:border-0">
                          <div className="flex items-baseline justify-between gap-3">
                            <span className="text-[12px] font-medium text-slate-200">
                              {cut.name.replace(/_/g, " ")}
                            </span>
                            <span className="font-mono text-[12px] text-slate-100">
                              {fmt(cut.depth_mm, "mm", 1)}
                            </span>
                          </div>
                          <div className="mt-0.5 flex flex-wrap gap-x-3 font-mono text-[10px] text-slate-500">
                            {Object.entries(cut.angles_deg ?? {}).map(([k, v]) =>
                              v == null ? null : (
                                <span key={k}>
                                  {k.replace(/_/g, " ")} {v.toFixed(2)}°
                                </span>
                              ),
                            )}
                          </div>
                          {cut.reference && (
                            <p className="mt-1 text-[10px] leading-snug text-slate-600">
                              {cut.reference}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {summary.simulation && (
                  <div className="card">
                    <div className="card-head">
                      <span className="card-title">Geometric simulation</span>
                      <span
                        className={`chip ${
                          summary.simulation.passed
                            ? "border-emerald-600 bg-emerald-500/15 text-emerald-400"
                            : "border-rose-600 bg-rose-500/15 text-rose-400"
                        }`}
                      >
                        {summary.simulation.passed ? "pass" : "fail"}
                      </span>
                    </div>
                    <div className="card-body">
                      <div className="row">
                        <span className="row-key">Failures / warnings</span>
                        <span className="row-val">
                          {summary.simulation.n_failures} / {summary.simulation.n_warnings}
                        </span>
                      </div>
                      <div className="row">
                        <span className="row-key">Bone resected</span>
                        <span className="row-val">
                          {summary.simulation.resected_volume_mm3 != null
                            ? `${(summary.simulation.resected_volume_mm3 / 1000).toFixed(1)} cm³`
                            : "—"}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            {step === "simulate" && !oaStatus.detected && (
              <div className="card border-emerald-800">
                <div className="card-head">
                  <span className="card-title text-emerald-300">Simulation not indicated</span>
                  <span className="chip border-emerald-600 bg-emerald-500/15 text-emerald-300">
                    no OA
                  </span>
                </div>
                <div className="card-body">
                  <p className="text-[12px] leading-relaxed text-slate-300">
                    Robotic simulation animates a resection plan. Since no replacement is
                    indicated for this knee, there is no plan to approve or simulate.
                  </p>
                </div>
              </div>
            )}

            {step === "simulate" && oaStatus.detected && summary && (
              <>
                {/* The cell opens in its own window. It used to float over the anatomy,
                    which put machine motion and the knee it is cutting on the same
                    pixels; they are two different things to look at. */}
                <div className="card border-amber-900/50">
                  <div className="card-head">
                    <span className="card-title text-amber-300">Robot cell</span>
                    <span className="card-sub">separate window</span>
                  </div>
                  <div className="card-body">
                    <p className="mb-2.5 text-[11px] leading-relaxed text-teal-100/55">
                      The 3-axis gantry runs in its own window so the machine and the
                      anatomy are not fighting for the same canvas. Approval, safety checks
                      and the command list are mirrored there.
                    </p>
                    <button
                      type="button"
                      className="btn-primary w-full"
                      onClick={() => openRobotWindow(summary.case_id)}
                    >
                      Open robot cell ↗
                    </button>
                  </div>
                </div>

                <RobotPanel
                  caseId={summary.case_id}
                  hasPlan={Boolean(summary.plan)}
                  state={simState}
                  commands={commands}
                  animating={progress !== null && progress < 1}
                  activeCommand={activeCommand}
                  onStateChange={setSimState}
                  onCommandsChange={setCommands}
                  onAnimate={animate}
                />
              </>
            )}
              </div>
            </FloatingPanel>

            {/* Surface probe. Its own panel, because it is a direct response to a click
                and should not push the step content around when it appears. */}
            {pick && (
              <FloatingPanel
                key="probe"
                title={pick.structure.replace(/_/g, " ")}
                subtitle={`${pick.category} · surface probe`}
                accent="emerald"
                initial={{ x: 420, y: 16 }}
                width={300}
                onClose={() => setPick(null)}
              >
                <div className="px-3 py-2.5">
                  <div className="row">
                    <span className="row-key">Surface point</span>
                    <span className="row-val">
                      {pick.point.map((v) => v.toFixed(1)).join(", ")} mm
                    </span>
                  </div>
                  {pick.entry.volume_mm3 != null && (
                    <div className="row">
                      <span className="row-key">Structure volume</span>
                      <span className="row-val">
                        {(pick.entry.volume_mm3 / 1000).toFixed(1)} cm³
                      </span>
                    </div>
                  )}
                  <div className="row">
                    <span className="row-key">Triangles</span>
                    <span className="row-val">{pick.entry.n_faces.toLocaleString()}</span>
                  </div>
                  <div className="row">
                    <span className="row-key">Watertight</span>
                    <span
                      className={`row-val ${
                        pick.entry.is_watertight ? "text-emerald-400" : "text-amber-400"
                      }`}
                    >
                      {pick.entry.is_watertight ? "yes" : "no"}
                    </span>
                  </div>
                  {pick.nearestLandmark && (
                    <div className="row">
                      <span className="row-key">Nearest landmark</span>
                      <span className="row-val">
                        {pick.nearestLandmark.name.replace(/_/g, " ")} ·{" "}
                        {pick.nearestLandmark.distance_mm.toFixed(1)} mm
                      </span>
                    </div>
                  )}
                  <p className="mt-2 text-[10px] leading-relaxed text-teal-100/40">
                    Image frame, millimetres. Click another surface to move the probe.
                  </p>
                </div>
              </FloatingPanel>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
