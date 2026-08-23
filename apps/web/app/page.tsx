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

import ImplantPanel from "@/components/ImplantPanel";
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
  UploadJob,
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
  const [loading, setLoading] = useState(true);
  const [viewerKey, setViewerKey] = useState(0);
  const [pendingUpload, setPendingUpload] = useState<string | null>(null);

  /** Lets a surgeon open the plan on a knee the classifier called non-arthritic. */
  const [indicationOverride, setIndicationOverride] = useState(false);

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
        // With nothing on disk, upload is the only useful thing to show.
        if (!best) setStep("upload");
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
    // Must reset per case: an override carried from a previous knee would silently
    // unlock the plan on a knee nobody has assessed.
    setIndicationOverride(false);
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
          next.visible[mesh.path] = true;
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

  // Each step turns on the overlays it is about, and off the ones it isn't. The
  // viewer should always be showing what the current panel is discussing.
  useEffect(() => {
    setSettings((s) => ({
      ...s,
      showLandmarks: step === "anatomy" || step === "measure",
      showCutPlanes: step === "plan" || step === "simulate",
      showImplant: step === "plan" || step === "simulate",
      showBoundary: step === "simulate" && Boolean(commands),
      showTrajectory: step === "simulate" && Boolean(commands),
      // The implant only makes sense seated in resected bone, so the plan step
      // shows the after-state by default.
      resectedView: step === "plan" || step === "simulate" ? true : s.resectedView,
    }));
  }, [step, commands]);

  // An upload creates its case directory before the pipeline finishes, so the case
  // id is only selected once the job is terminal. Selecting it earlier would fetch a
  // case whose result.json does not exist yet and show a 404 mid-upload.
  const handleUploadCreated = useCallback((newCaseId: string) => {
    setPendingUpload(newCaseId);
  }, []);

  const handleUploadComplete = useCallback(async (job: UploadJob) => {
    setPendingUpload(null);
    try {
      const list = await api.listCases();
      setCases(list.cases);
      if (list.cases.some((c) => c.case_id === job.case_id)) {
        setCaseId(job.case_id);
        // Land on the answer the user came for rather than back at the top.
        setStep(job.outcome?.oa_available ? "oa" : "case");
      }
    } catch {
      // A failed refresh is not worth surfacing: the job panel already shows the
      // outcome, and the case selector can be reloaded by hand.
    }
  }, []);

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

  /**
   * True when the OA classifier ran and found no osteoarthritis.
   *
   * `oa_detected === false` specifically, not falsy: `null` means the assessment
   * could not be produced, and an absent verdict is not the same as a negative one.
   * Gating on the absent case would block every pre-OA case in the library.
   */
  const notIndicated = summary?.oa?.available === true && summary.oa.oa_detected === false;

  const steps: Step[] = useMemo(
    () => [
      {
        id: "upload",
        label: "Upload scan",
        hint: "MRI, CT or X-ray",
        // Always available: it is the entry point, and it must not depend on a
        // case already being loaded.
        ready: true,
      },
      { id: "case", label: "Patient case", hint: "Study, model, provenance", ready: true },
      {
        id: "segment",
        label: "AI segmentation",
        hint: `${result?.segmentation.structures_present.length ?? 0} structures identified`,
        ready: Boolean(summary),
      },
      {
        id: "oa",
        label: "OA assessment",
        hint:
          summary?.oa?.available && summary.oa.probability != null
            ? `${summary.oa.oa_detected ? "OA detected" : "No OA"} · ${Math.round(
                summary.oa.probability * 100,
              )}%`
            : "Osteoarthritis: yes or no",
        ready: Boolean(summary),
        blockedReason: "Load or upload a case first",
      },
      {
        id: "slices",
        label: "3 views",
        hint: "Axial, coronal and sagittal with contours",
        ready: Boolean(summary),
        blockedReason: "Load or upload a case first",
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
        id: "plan",
        label: "Implant & plan",
        hint: notIndicated
          ? "Not indicated — no osteoarthritis"
          : `${summary?.plan?.cuts.length ?? 0} resection cuts`,
        // Arthroplasty treats osteoarthritis. With no OA detected there is no
        // indication, so the plan is withheld by default rather than presented as
        // a recommendation. Overridable: the classifier is weak (AUROC 0.567) and
        // must not be able to veto a surgeon.
        ready: Boolean(summary?.plan) && (!notIndicated || indicationOverride),
        blockedReason: notIndicated
          ? "No osteoarthritis detected — arthroplasty is not indicated. Override below to inspect the geometry anyway."
          : "No surgical plan for this case",
      },
      {
        id: "readiness",
        label: "Robotic readiness",
        hint:
          summary?.readiness?.percentage != null
            ? `${Math.round(summary.readiness.percentage)}% · ${(
                summary.readiness.band ?? ""
              ).replace(/_/g, " ")}`
            : "Percentage, with the breakdown",
        ready: Boolean(summary?.readiness),
        blockedReason: "Scored at the end of the pipeline; none for this case yet",
      },
      {
        id: "simulate",
        label: "Approve & simulate",
        hint: notIndicated ? "Not indicated — no osteoarthritis" : "Surgeon gate, then robot simulation",
        ready: Boolean(summary?.plan) && (!notIndicated || indicationOverride),
        blockedReason: notIndicated
          ? "No osteoarthritis detected — nothing to operate on. Override below if you disagree with the classifier."
          : "Requires a surgical plan",
      },
    ],
    [summary, result, notIndicated, indicationOverride],
  );

  return (
    <div className="flex h-[calc(100vh-1.75rem)] flex-col">
      {/* Header */}
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-2.5">
        <div className="flex items-baseline gap-3">
          <h1 className="text-[15px] font-semibold tracking-tight text-slate-50">
            Knee Intelligence Platform
          </h1>
          <span className="hidden text-[11px] text-slate-500 md:inline">
            MRI → segmentation → 3D → measurement → plan → simulation
          </span>
        </div>
        <div className="flex items-center gap-2.5">
          {pendingUpload && (
            <span
              className="chip border-sky-600 bg-sky-500/15 text-sky-300"
              title={`Processing upload ${pendingUpload}`}
            >
              processing upload…
            </span>
          )}
          {summary && <SourceBadge source={summary.source} />}
          <select
            aria-label="Select case"
            value={caseId ?? ""}
            onChange={(e) => setCaseId(e.target.value)}
            className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-[12px]"
          >
            {cases.map((c) => (
              <option key={c.case_id} value={c.case_id}>
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

      {/* Upload is reachable with no case loaded — it is how the first case gets
          here, so it cannot be nested inside the `summary` guard below. */}
      {!summary && step === "upload" && !loading && (
        <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 p-3 xl:grid-cols-[1fr_400px]">
          <section className="flex min-h-[26rem] items-center justify-center rounded-lg border border-slate-800 bg-slate-950">
            <div className="max-w-md px-6 text-center">
              <p className="text-[13px] font-medium text-slate-300">
                Upload a scan to begin
              </p>
              <p className="mt-2 text-[12px] leading-relaxed text-slate-500">
                A knee MR volume runs the full pipeline: AI segmentation, 3D
                reconstruction, measurement, an osteoarthritis assessment, implant
                matching, a resection plan and a robotic readiness score. The
                reconstructed knee appears here when it is ready.
              </p>
            </div>
          </section>
          <aside className="min-h-0 space-y-3 overflow-y-auto pr-0.5">
            <UploadPanel
              onCaseCreated={handleUploadCreated}
              onComplete={handleUploadComplete}
            />
          </aside>
        </div>
      )}

      {summary && (
        <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 p-3 xl:grid-cols-[1fr_400px]">
          {/* ---------- Viewer ---------- */}
          <section className="relative min-h-[26rem] overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
            {summary.meshes ? (
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
                    kneeai infer --case-id {summary.case_id}
                  </code>{" "}
                  to build meshes, measurements and a plan.
                </p>
              </div>
            )}
          </section>

          {/* ---------- Context panel ---------- */}
          <aside className="min-h-0 space-y-3 overflow-y-auto pr-0.5">
            {/* Surface pick. Pinned above the step content because it is a direct
                response to something the user just clicked. */}
            {pick && (
              <div className="card border-sky-800/70">
                <div className="card-head">
                  <span className="card-title">
                    {pick.structure.replace(/_/g, " ")}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="chip border-sky-600 bg-sky-500/15 text-sky-300">
                      {pick.category}
                    </span>
                    <button
                      type="button"
                      onClick={() => setPick(null)}
                      className="text-slate-500 hover:text-slate-200"
                      aria-label="Clear selection"
                    >
                      ×
                    </button>
                  </div>
                </div>
                <div className="card-body">
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
                      className={`row-val ${pick.entry.is_watertight ? "text-emerald-400" : "text-amber-400"}`}
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
                  <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                    Coordinates are in the image frame, millimetres. Click elsewhere on a
                    surface to move the probe, or on the background to clear it.
                  </p>
                </div>
              </div>
            )}

            {step === "case" && (
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

            {step === "anatomy" && summary.meshes && (
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

            {step === "upload" && (
              <UploadPanel
                onCaseCreated={handleUploadCreated}
                onComplete={handleUploadComplete}
              />
            )}

            {step === "slices" && <Slice2D caseId={summary.case_id} />}

            {step === "measure" && <MeasurementsPanel data={summary.measurements} />}

            {step === "oa" && (
              <>
                <OaPanel
                  data={summary.measurements}
                  oa={summary.oa}
                  klGrade={summary.kl_grade ?? null}
                />
                {notIndicated && summary.plan && (
                  <div className="card border-amber-700">
                    <div className="card-head">
                      <span className="card-title text-amber-300">
                        Arthroplasty not indicated
                      </span>
                    </div>
                    <div className="card-body">
                      <p className="text-[11px] leading-relaxed text-slate-300">
                        A total knee replacement treats osteoarthritis. No
                        osteoarthritis was detected here, so the implant, resection
                        plan and robot simulation steps are{" "}
                        <strong>withheld</strong> — presenting them would imply a
                        recommendation that does not follow from the assessment.
                      </p>
                      <p className="mt-2 text-[11px] leading-relaxed text-slate-400">
                        The plan was still computed and is retained as reference
                        geometry. This classifier is weak (validation AUROC 0.567 on
                        11 cases), so it must not be able to veto a clinician.
                      </p>
                      <label className="mt-3 flex cursor-pointer items-start gap-2">
                        <input
                          type="checkbox"
                          checked={indicationOverride}
                          onChange={(e) => setIndicationOverride(e.target.checked)}
                          className="mt-0.5 h-3.5 w-3.5 shrink-0 accent-amber-500"
                        />
                        <span className="text-[11px] leading-snug text-amber-300">
                          I disagree with the assessment — unlock the plan and
                          simulation steps for this case.
                        </span>
                      </label>
                    </div>
                  </div>
                )}
              </>
            )}

            {step === "readiness" && <ReadinessPanel readiness={summary.readiness} />}

            {step === "plan" && (
              <>
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

            {step === "simulate" && (
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
            )}
          </aside>
        </div>
      )}
    </div>
  );
}
