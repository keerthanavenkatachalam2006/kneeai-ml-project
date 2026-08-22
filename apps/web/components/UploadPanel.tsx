"use client";

/**
 * Scan upload: pick a file, choose the side, watch it process.
 *
 * Three things this panel refuses to do:
 *
 * **Guess the operative side.** It is a required choice with no default beyond
 * "unknown", and choosing "unknown" is allowed but visibly withholds planning. The
 * platform treats a guessed side as a wrong-site-surgery class error, so the UI does
 * not quietly pick one to make the flow smoother.
 *
 * **Pretend progress is linear.** The stage label from the pipeline is shown verbatim
 * alongside the bar, because segmentation alone is most of the wall clock and a bar
 * that sits at 12% for forty seconds needs to say what it is doing.
 *
 * **Hide a rejection.** A file that fails validation, or a radiograph that no model
 * here can read, terminates with the backend's own explanation rendered in full.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api } from "@/lib/api";
import type { UploadJob } from "@/lib/types";

const POLL_MS = 1200;

const ACCEPT = ".nii,.nii.gz,.zip,.dcm,.dicom,.ima,.png,.jpg,.jpeg";

const SIDES: Array<[string, string]> = [
  ["right", "Right"],
  ["left", "Left"],
  ["unknown", "Unknown"],
];

/**
 * A bare NIfTI carries no modality tag. Left blank the server assumes MR and says
 * so in a warning; declaring it here removes the assumption from the record.
 */
const MODALITIES: Array<[string, string]> = [
  ["", "Auto-detect"],
  ["MR", "MR"],
  ["CT", "CT"],
];

const STATUS_STYLE: Record<string, string> = {
  queued: "border-slate-600 bg-slate-700/40 text-slate-300",
  validating: "border-sky-600 bg-sky-500/15 text-sky-300",
  segmenting: "border-sky-600 bg-sky-500/15 text-sky-300",
  processing: "border-sky-600 bg-sky-500/15 text-sky-300",
  complete: "border-emerald-600 bg-emerald-500/15 text-emerald-300",
  failed: "border-rose-600 bg-rose-500/15 text-rose-300",
  rejected: "border-rose-600 bg-rose-500/15 text-rose-300",
  radiograph_only: "border-amber-600 bg-amber-500/15 text-amber-300",
};

function humanBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export default function UploadPanel({
  onComplete,
  onCaseCreated,
}: {
  /** Called when a job reaches a terminal state, with the finished job. */
  onComplete?: (job: UploadJob) => void;
  /** Called as soon as the case id exists, so the shell can select it. */
  onCaseCreated?: (caseId: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [side, setSide] = useState("right");
  const [modality, setModality] = useState("");
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState<UploadJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const notifiedRef = useRef<string | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => clearTimer, [clearTimer]);

  // Poll while the job is live. Recursive setTimeout rather than setInterval so a
  // slow response cannot stack requests on a machine already busy segmenting.
  const poll = useCallback(
    (caseId: string) => {
      clearTimer();
      timerRef.current = setTimeout(async () => {
        try {
          const next = await api.uploadStatus(caseId);
          setJob(next);
          if (next.terminal) {
            if (notifiedRef.current !== caseId) {
              notifiedRef.current = caseId;
              onComplete?.(next);
            }
            return;
          }
          poll(caseId);
        } catch (e) {
          setError(e instanceof ApiError ? e.message : String(e));
        }
      }, POLL_MS);
    },
    [clearTimer, onComplete],
  );

  const submit = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    notifiedRef.current = null;
    try {
      const created = await api.uploadScan(file, {
        side,
        modality: modality || undefined,
      });
      setJob(created);
      onCaseCreated?.(created.case_id);
      poll(created.case_id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [file, side, modality, poll, onCaseCreated]);

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      setFile(dropped);
      setJob(null);
      setError(null);
    }
  }, []);

  const running = job !== null && !job.terminal;
  const percent = Math.round((job?.progress ?? 0) * 100);

  return (
    <>
      <div className="card">
        <div className="card-head">
          <span className="card-title">Upload a scan</span>
          <span className="card-sub">MRI · CT · X-ray</span>
        </div>

        <div className="card-body">
          {/* ---------- Dropzone ---------- */}
          <div
            role="button"
            tabIndex={0}
            aria-label="Choose or drop a scan file"
            onClick={() => !running && inputRef.current?.click()}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && !running) inputRef.current?.click();
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed px-3 py-5 text-center transition-colors ${
              dragging
                ? "border-sky-500 bg-sky-500/10"
                : "border-slate-700 bg-slate-900/60 hover:border-slate-600"
            } ${running ? "pointer-events-none opacity-60" : ""}`}
          >
            <p className="text-[12px] font-medium text-slate-200">
              {file ? file.name : "Drop a scan here, or click to choose"}
            </p>
            <p className="mt-1 text-[10px] leading-snug text-slate-500">
              {file
                ? humanBytes(file.size)
                : "NIfTI (.nii, .nii.gz) · DICOM series as .zip · single .dcm · radiograph (.png, .jpg)"}
            </p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => {
              const chosen = e.target.files?.[0] ?? null;
              setFile(chosen);
              setJob(null);
              setError(null);
            }}
          />

          {/* ---------- Side ---------- */}
          <div className="mt-3">
            <label className="section-label" htmlFor="upload-side">
              Operative side
            </label>
            <div id="upload-side" className="flex gap-1.5">
              {SIDES.map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  disabled={running}
                  onClick={() => setSide(value)}
                  className={`flex-1 rounded-md border px-2 py-1.5 text-[12px] transition-colors disabled:opacity-50 ${
                    side === value
                      ? "border-sky-500 bg-sky-500/15 text-sky-200"
                      : "border-slate-700 bg-slate-900 text-slate-400 hover:border-slate-600"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            {side === "unknown" && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-amber-400">
                With an unknown side, medial and lateral cannot be resolved. Segmentation,
                measurements and the OA assessment still run; implant matching, planning
                and simulation are withheld, because guessing the side is a
                wrong-site-surgery class error.
              </p>
            )}
          </div>

          {/* ---------- Modality ---------- */}
          <div className="mt-3">
            <label className="section-label" htmlFor="upload-modality">
              Modality
            </label>
            <select
              id="upload-modality"
              value={modality}
              disabled={running}
              onChange={(e) => setModality(e.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-[12px] text-slate-200 disabled:opacity-50"
            >
              {MODALITIES.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            {modality === "CT" && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-amber-400">
                The segmentation model was trained on 3D DESS MR only. A CT will be
                stored and validated, but segmentation is refused rather than run
                across modalities without validation behind it.
              </p>
            )}
          </div>

          <button
            type="button"
            disabled={!file || busy || running}
            onClick={submit}
            className="mt-3 w-full rounded-md border border-sky-600 bg-sky-600/20 px-3 py-2 text-[12px] font-medium text-sky-200 transition-colors hover:bg-sky-600/30 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? "Uploading…" : running ? "Processing…" : "Upload and analyse"}
          </button>

          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
            Segmentation runs on CPU here and takes roughly 30–90 seconds. Uploads are
            processed one at a time.
          </p>

          {error && (
            <div role="alert" className="mt-2 rounded-md border border-rose-800 bg-rose-950/40 px-2.5 py-2">
              <p className="text-[11px] leading-relaxed text-rose-300">{error}</p>
            </div>
          )}
        </div>
      </div>

      {/* ---------- Progress ---------- */}
      {job && (
        <div className="card">
          <div className="card-head">
            <span className="card-title">{job.case_id}</span>
            <span className={`chip ${STATUS_STYLE[job.status] ?? ""}`}>
              {job.status.replace(/_/g, " ")}
            </span>
          </div>
          <div className="card-body">
            <div className="h-2 w-full overflow-hidden rounded bg-slate-800">
              <div
                className={`h-full transition-all duration-500 ${
                  job.status === "failed" || job.status === "rejected"
                    ? "bg-rose-500"
                    : job.status === "complete"
                      ? "bg-emerald-500"
                      : "bg-sky-500"
                }`}
                style={{ width: `${percent}%` }}
              />
            </div>
            <div className="mt-1.5 flex items-baseline justify-between gap-2">
              <span className="text-[11px] text-slate-400">{job.stage}</span>
              <span className="shrink-0 font-mono text-[10px] text-slate-500">
                {percent}% · {job.elapsed_s.toFixed(0)}s
              </span>
            </div>

            <div className="mt-2">
              <div className="row">
                <span className="row-key">File</span>
                <span className="row-val">{humanBytes(job.bytes_size)}</span>
              </div>
              <div className="row">
                <span className="row-key">Side</span>
                <span className="row-val">{job.side}</span>
              </div>
              {job.modality && (
                <div className="row">
                  <span className="row-key">Modality</span>
                  <span className="row-val">{job.modality}</span>
                </div>
              )}
            </div>

            {job.error && (
              <div className="mt-2 rounded-md border border-rose-800 bg-rose-950/40 px-2.5 py-2">
                <p className="text-[11px] font-medium text-rose-300">
                  {job.status === "rejected" ? "Rejected" : "Failed"}
                </p>
                <p className="mt-1 text-[11px] leading-relaxed text-rose-200">{job.error}</p>
              </div>
            )}

            {/* The radiograph path: accepted, stored, and honestly unusable here. */}
            {job.status === "radiograph_only" && job.outcome.oa_reason && (
              <div className="mt-2 rounded-md border border-amber-800 bg-amber-950/30 px-2.5 py-2">
                <p className="text-[11px] font-medium text-amber-300">
                  Radiograph stored, but no model here can read it
                </p>
                <p className="mt-1 text-[11px] leading-relaxed text-amber-200">
                  {job.outcome.oa_reason}
                </p>
              </div>
            )}

            {/* ---------- Headline result ---------- */}
            {job.status === "complete" && (
              <div className="mt-3 border-t border-slate-800 pt-2">
                <h4 className="section-label">Result</h4>
                {job.outcome.oa_available ? (
                  <div className="row">
                    <span className="row-key">Osteoarthritis</span>
                    <span
                      className={`row-val ${
                        job.outcome.oa_detected ? "text-rose-400" : "text-emerald-400"
                      }`}
                    >
                      {job.outcome.oa_detected ? "detected" : "not detected"}
                      {job.outcome.oa_probability != null &&
                        ` · ${Math.round(job.outcome.oa_probability * 100)}%`}
                      {job.outcome.oa_severity && ` · ${job.outcome.oa_severity}`}
                    </span>
                  </div>
                ) : (
                  <p className="text-[11px] leading-relaxed text-slate-400">
                    OA assessment unavailable. {job.outcome.oa_reason ?? ""}
                  </p>
                )}
                {job.outcome.readiness_pct != null && (
                  <div className="row">
                    <span className="row-key">Robotic readiness</span>
                    <span className="row-val">
                      {job.outcome.readiness_pct.toFixed(0)}%
                      {job.outcome.readiness_band &&
                        ` · ${job.outcome.readiness_band.replace(/_/g, " ")}`}
                    </span>
                  </div>
                )}
                <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
                  Full detail is in the OA analysis and robotic readiness steps above.
                </p>
              </div>
            )}

            {job.warnings.length > 0 && (
              <div className="mt-2 border-t border-slate-800 pt-2">
                <h4 className="section-label text-amber-400">
                  Warnings ({job.warnings.length})
                </h4>
                <ul className="space-y-1">
                  {job.warnings.map((w, i) => (
                    <li key={i} className="text-[11px] leading-relaxed text-amber-400/90">
                      {w}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
