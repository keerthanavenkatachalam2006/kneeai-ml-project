"use client";

/**
 * Surgeon approval gate and robot simulation control.
 *
 * The gate is real, not decorative. Run is disabled until the API has recorded a
 * named approval, and the API refuses regardless of what this component sends —
 * `SimulationController.start()` raises without an approver. The UI is the second
 * line of defence, not the only one, which is why an empty approver blocks here
 * *and* returns 422 from the server.
 *
 * "Emergency stop" halts the simulation. There is no hardware attached to this
 * system, and the copy says so rather than implying otherwise.
 */

import { useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { RobotCommandSequence, SimulationStateInfo } from "@/lib/types";

const STATES: Record<string, { text: string; className: string }> = {
  planning: { text: "Planning", className: "border-slate-600 bg-slate-800 text-slate-300" },
  awaiting_surgeon_approval: {
    text: "Awaiting approval",
    className: "border-amber-600 bg-amber-500/15 text-amber-300",
  },
  approved_for_simulation: {
    text: "Approved",
    className: "border-sky-500 bg-sky-500/15 text-sky-300",
  },
  safety_check: { text: "Checks passed", className: "border-sky-500 bg-sky-500/15 text-sky-300" },
  simulation_running: {
    text: "Running",
    className: "border-emerald-600 bg-emerald-500/15 text-emerald-300",
  },
  simulation_complete: {
    text: "Complete",
    className: "border-emerald-600 bg-emerald-500/15 text-emerald-300",
  },
  emergency_stopped: { text: "Stopped", className: "border-rose-600 bg-rose-500/15 text-rose-300" },
  rejected: { text: "Rejected", className: "border-rose-600 bg-rose-500/15 text-rose-300" },
};

interface Props {
  caseId: string;
  hasPlan: boolean;
  state: SimulationStateInfo | null;
  commands: RobotCommandSequence | null;
  animating: boolean;
  activeCommand: number | null;
  onStateChange: (s: SimulationStateInfo) => void;
  onCommandsChange: (c: RobotCommandSequence) => void;
  onAnimate: () => void;
}

export default function RobotPanel({
  caseId,
  hasPlan,
  state,
  commands,
  animating,
  activeCommand,
  onStateChange,
  onCommandsChange,
  onAnimate,
}: Props) {
  const [approver, setApprover] = useState("");
  const [justification, setJustification] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const name = state?.state ?? "planning";
  const badge = STATES[name] ?? { text: name, className: "border-slate-600 bg-slate-800 text-slate-300" };
  const approved = Boolean(state?.approver);
  const checksPassed = state?.safety_report?.passed ?? false;
  const terminal = name === "simulation_complete" || name === "emergency_stopped";
  const canRun = approved && checksPassed && !terminal;

  async function act(fn: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!hasPlan) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">Approve &amp; simulate</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] text-slate-500">
            This case has no surgical plan, so there is nothing to approve.
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="card">
        <div className="card-head">
          <span className="card-title">Surgeon approval</span>
          <span className={`chip ${badge.className}`}>{badge.text}</span>
        </div>

        <div className="card-body">
          {!approved ? (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (!approver.trim()) {
                  setError("A named approver is required. Anonymous approval is refused.");
                  return;
                }
                void act(async () => {
                  onStateChange(await api.approve(caseId, approver.trim(), justification.trim()));
                });
              }}
            >
              <p className="mb-3 text-[11px] leading-relaxed text-slate-400">
                The simulation cannot start without a named approval. This is enforced by the
                API, not just this form.
              </p>
              <label className="field-label" htmlFor="approver">
                Approving surgeon <span className="text-rose-400">*</span>
              </label>
              <input
                id="approver"
                value={approver}
                onChange={(e) => setApprover(e.target.value)}
                placeholder="dr-smith"
                required
                className="field mb-2.5"
              />
              <label className="field-label" htmlFor="justification">
                Justification (recorded in the audit trail)
              </label>
              <textarea
                id="justification"
                value={justification}
                onChange={(e) => setJustification(e.target.value)}
                rows={2}
                placeholder="Reviewed cuts, alignment and implant fit."
                className="field mb-3 resize-none"
              />
              <button type="submit" className="btn-primary w-full" disabled={busy}>
                {busy ? "Recording…" : "Approve plan for simulation"}
              </button>
            </form>
          ) : (
            <>
              <div className="row">
                <span className="row-key">Approved by</span>
                <span className="row-val">{state?.approver}</span>
              </div>
              {state?.approved_at && (
                <div className="row">
                  <span className="row-key">At</span>
                  <span className="row-val">{new Date(state.approved_at).toLocaleTimeString()}</span>
                </div>
              )}
            </>
          )}

          {state?.safety_report && (
            <div className="mt-3">
              <div className="mb-1.5 flex items-center justify-between">
                <h4 className="section-label !mb-0">Safety checks</h4>
                <span
                  className={`text-[11px] font-semibold ${
                    state.safety_report.passed ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {state.safety_report.passed ? "PASS" : "BLOCKED"}
                </span>
              </div>
              {state.safety_report.checks.map((c) => (
                <div key={c.name} className="flex items-start gap-2 py-0.5">
                  <span
                    aria-hidden
                    className={`mt-px shrink-0 font-mono text-[11px] ${
                      c.passed ? "text-emerald-400" : c.blocking ? "text-rose-400" : "text-amber-400"
                    }`}
                  >
                    {c.passed ? "✓" : c.blocking ? "✕" : "!"}
                  </span>
                  <div className="min-w-0">
                    <div className="text-[11.5px] text-slate-300">{c.name.replace(/_/g, " ")}</div>
                    <div className="text-[11px] leading-snug text-slate-500">{c.detail}</div>
                  </div>
                </div>
              ))}
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                These confirm the plan artefacts are coherent enough to animate. They are not a
                clinical safety assessment.
              </p>
            </div>
          )}

          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              className="btn-primary flex-1"
              disabled={!canRun || busy || animating}
              title={
                canRun
                  ? "Generate the command sequence and animate it"
                  : "Requires a recorded approval and passing safety checks"
              }
              onClick={() =>
                void act(async () => {
                  const res = await api.runSimulation(caseId);
                  onStateChange(res.state);
                  onCommandsChange(res.commands);
                  onAnimate();
                })
              }
            >
              {animating ? "Simulating…" : "Run robot simulation"}
            </button>
            {commands && !animating && (
              <button type="button" className="btn-ghost" onClick={onAnimate}>
                Replay
              </button>
            )}
          </div>

          <div className="mt-2 flex gap-2">
            <button
              type="button"
              className="btn-danger flex-1"
              disabled={busy || terminal || !approved}
              onClick={() => void act(async () => onStateChange(await api.stopSimulation(caseId)))}
              title="Stops the simulation. No hardware is attached to this system."
            >
              Emergency stop
            </button>
            <button
              type="button"
              className="btn-ghost"
              disabled={busy}
              onClick={() =>
                void act(async () => {
                  await api.resetSimulation(caseId);
                  onStateChange({
                    case_id: caseId,
                    state: "planning",
                    approver: null,
                    safety_report: null,
                    banner: state?.banner ?? "",
                    hardware_control: false,
                  });
                })
              }
            >
              Reset
            </button>
          </div>

          {error && (
            <p
              role="alert"
              className="mt-2.5 rounded-md border border-rose-800 bg-rose-950/40 px-2.5 py-2 text-[11px] leading-relaxed text-rose-300"
            >
              {error}
            </p>
          )}
        </div>
      </div>

      {commands && (
        <div className="card">
          <div className="card-head">
            <span className="card-title">Command sequence</span>
            <span className="card-sub">{commands.n_commands} steps</span>
          </div>
          <div className="card-body">
            <div className="max-h-64 overflow-y-auto">
              {commands.commands.map((c, i) => (
                <div
                  key={c.sequence}
                  className={`flex items-baseline gap-2 rounded px-1.5 py-1 ${
                    activeCommand === i ? "bg-sky-500/15" : ""
                  }`}
                >
                  <span className="w-5 shrink-0 text-right font-mono text-[10px] text-slate-600">
                    {c.sequence}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-mono text-[11px] text-slate-300">{c.action}</div>
                    <div className="truncate text-[10px] text-slate-600">{c.target}</div>
                  </div>
                  {c.status !== "READY_FOR_SIMULATION" && (
                    <span className="shrink-0 text-[10px] text-slate-600">{c.status}</span>
                  )}
                </div>
              ))}
            </div>

            {commands.warnings.length > 0 && (
              <div className="mt-2 space-y-1">
                {commands.warnings.map((w, i) => (
                  <p key={i} className="text-[11px] leading-snug text-amber-400">
                    {w}
                  </p>
                ))}
              </div>
            )}

            {commands.safety_boundary && (
              <div className="mt-2 border-t border-slate-800 pt-2">
                <div className="row">
                  <span className="row-key">Keep-out volume</span>
                  <span className="row-val">
                    {commands.safety_boundary.size_mm.map((v) => v.toFixed(0)).join(" × ")} mm
                  </span>
                </div>
              </div>
            )}

            <p className="mt-2 text-[11px] leading-relaxed text-slate-500">{commands.note}</p>
          </div>
        </div>
      )}
    </>
  );
}
