"use client";

/**
 * Robot cell — standalone window.
 *
 * The cell used to be a floating panel over the anatomy in the dashboard, which meant the
 * machine motion was competing for the same pixels as the thing it was cutting. Watching a
 * simulation and reading a plan are two different jobs, so this is its own route: open it
 * with `window.open` and it behaves like a second monitor for the cell.
 *
 * Opened as `/robot?case=<case_id>`. It talks to the same API as the dashboard, so an
 * approval recorded in either window is visible in both after a refresh of the state.
 *
 * SIMULATION ONLY. Nothing here can drive hardware; there is no hardware attached.
 */

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import RobotPanel from "@/components/RobotPanel";
import { api, ApiError } from "@/lib/api";
import type {
  CaseListEntry,
  CaseSummary,
  RobotCommandSequence,
  SimulationStateInfo,
} from "@/lib/types";

const RobotSimulation3D = dynamic(() => import("@/components/RobotSimulation3D"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-[12px] text-teal-200/40">
      Loading robot cell…
    </div>
  ),
});

/** Full sweep duration at 1×, matching the dashboard. */
const ANIMATION_MS = 10000;
const SPEEDS = [0.5, 1, 2] as const;

export default function RobotWindow() {
  const [cases, setCases] = useState<CaseListEntry[]>([]);
  const [caseId, setCaseId] = useState<string | null>(null);
  const [summary, setSummary] = useState<CaseSummary | null>(null);
  const [state, setState] = useState<SimulationStateInfo | null>(null);
  const [commands, setCommands] = useState<RobotCommandSequence | null>(null);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [error, setError] = useState<string | null>(null);
  const [railOpen, setRailOpen] = useState(true);

  const rafRef = useRef<number | null>(null);
  const speedRef = useRef(speed);
  speedRef.current = speed;

  // --- Which case? From the query string, else the first planned case. ---------
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("case");
    let alive = true;
    (async () => {
      try {
        const list = await api.listCases();
        if (!alive) return;
        setCases(list.cases);
        const fallback =
          list.cases.find((c) => c.case_id === requested) ??
          list.cases.find((c) => c.has_plan) ??
          list.cases[0];
        setCaseId(requested ?? fallback?.case_id ?? null);
      } catch (e) {
        if (alive) setError(e instanceof ApiError ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!caseId) return;
    let alive = true;
    setProgress(0);
    setPlaying(false);
    (async () => {
      try {
        const [detail, simState, cmds] = await Promise.all([
          api.case(caseId),
          api.simulationState(caseId).catch(() => null),
          api.robotCommands(caseId).catch(() => null),
        ]);
        if (!alive) return;
        setSummary(detail);
        setState(simState);
        setCommands(cmds);
        setError(null);
      } catch (e) {
        if (alive) setError(e instanceof ApiError ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [caseId]);

  // --- Playback ---------------------------------------------------------------
  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    setPlaying(false);
  }, []);

  const play = useCallback(
    (from = 0) => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const start = performance.now() - from * ANIMATION_MS;
      setPlaying(true);
      const tick = (now: number) => {
        const t = Math.min(1, ((now - start) * speedRef.current) / ANIMATION_MS);
        setProgress(t);
        if (t < 1) {
          rafRef.current = requestAnimationFrame(tick);
        } else {
          rafRef.current = null;
          setPlaying(false);
        }
      };
      rafRef.current = requestAnimationFrame(tick);
    },
    [],
  );

  useEffect(
    () => () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    },
    [],
  );

  const activeCommand = useMemo(() => {
    if (!commands) return null;
    const posed = commands.commands.filter((c) => c.pose);
    return posed.length ? Math.min(posed.length - 1, Math.floor(progress * posed.length)) : null;
  }, [progress, commands]);

  const badge = state?.state?.replace(/_/g, " ") ?? "no state";

  return (
    <div className="flex h-[calc(100vh-1.75rem)] flex-col bg-clinic-950">
      {/* --- Header --- */}
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-white/5 px-4 py-2.5">
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-400/15 ring-1 ring-inset ring-amber-400/30"
          >
            <span className="h-2.5 w-2.5 rounded-sm bg-amber-400" />
          </span>
          <div>
            <h1 className="text-[13px] font-semibold leading-tight tracking-tight text-white">
              MORP · Robot cell
            </h1>
            <p className="text-[10px] leading-tight text-amber-200/50">
              3-axis gantry · simulation only · no hardware control
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="chip border-amber-600/60 bg-amber-500/10 text-amber-300">{badge}</span>
          <select
            aria-label="Select case"
            value={caseId ?? ""}
            onChange={(e) => setCaseId(e.target.value)}
            className="rounded-full border border-white/10 bg-clinic-900/80 px-3 py-1.5 text-[11px] text-teal-50 hover:border-white/20 focus:border-teal-400"
          >
            {cases.map((c) => (
              <option key={c.case_id} value={c.case_id} className="bg-clinic-900">
                {c.case_id}
                {c.has_plan ? "" : " (no plan)"}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setRailOpen((r) => !r)}
            className="btn-ghost !px-3 !py-1.5 !text-[11px]"
            aria-pressed={railOpen}
          >
            {railOpen ? "Hide controls" : "Show controls"}
          </button>
        </div>
      </header>

      {error && (
        <div role="alert" className="mx-4 mt-3 rounded-md border border-rose-800 bg-rose-950/40 px-3 py-2">
          <p className="text-[12px] text-rose-300">{error}</p>
          <p className="mt-1 text-[11px] text-slate-400">
            Is the API running? <code className="font-mono">make api</code> starts it on
            127.0.0.1:8000.
          </p>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        {/* --- Cell --- */}
        <section className="relative min-h-0 min-w-0 flex-1">
          <RobotSimulation3D
            commands={commands}
            running={playing}
            progress={progress}
            caseId={caseId}
            manifest={summary?.meshes ?? null}
            fill
            showNote={false}
          />
        </section>

        {/* --- Controls rail --- */}
        {railOpen && (
          <aside className="flex w-[380px] shrink-0 flex-col border-l border-white/5 bg-clinic-950">
            {/* Playback */}
            <div className="shrink-0 border-b border-white/5 px-3 py-3">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="section-label !mb-0">Playback</h2>
                <span className="font-mono text-[10px] text-teal-200/45">
                  {Math.round(progress * 100)}%
                </span>
              </div>

              <input
                type="range"
                min={0}
                max={1000}
                value={Math.round(progress * 1000)}
                aria-label="Scrub the sequence"
                onChange={(e) => {
                  stop();
                  setProgress(Number(e.target.value) / 1000);
                }}
                className="mb-2.5 h-1 w-full cursor-pointer appearance-none rounded-full bg-clinic-800 accent-teal-400"
              />

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="btn-primary flex-1 !py-1.5 !text-[11px]"
                  onClick={() => (playing ? stop() : play(progress >= 1 ? 0 : progress))}
                  disabled={!commands}
                >
                  {playing ? "Pause" : progress > 0 && progress < 1 ? "Resume" : "Play sweep"}
                </button>
                <button
                  type="button"
                  className="btn-ghost !py-1.5 !text-[11px]"
                  onClick={() => {
                    stop();
                    setProgress(0);
                  }}
                  disabled={!commands}
                >
                  Reset view
                </button>
                <div
                  className="flex overflow-hidden rounded-full border border-white/10"
                  role="group"
                  aria-label="Playback speed"
                >
                  {SPEEDS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setSpeed(s)}
                      aria-pressed={speed === s}
                      className={`px-2 py-1 font-mono text-[10px] transition-colors ${
                        speed === s
                          ? "bg-teal-400 text-clinic-950"
                          : "text-teal-100/50 hover:bg-white/5"
                      }`}
                    >
                      {s}×
                    </button>
                  ))}
                </div>
              </div>
              <p className="mt-2 text-[10px] leading-relaxed text-teal-100/40">
                Playback is a view of the sequence, not an execution of it. Scrubbing moves the
                tool along the planned poses; it sends nothing anywhere.
              </p>
            </div>

            {/* Approval + sequence, the same controls as the dashboard */}
            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
              {caseId && (
                <RobotPanel
                  caseId={caseId}
                  hasPlan={Boolean(summary?.plan)}
                  state={state}
                  commands={commands}
                  animating={playing}
                  activeCommand={activeCommand}
                  onStateChange={setState}
                  onCommandsChange={setCommands}
                  onAnimate={() => play(0)}
                />
              )}
              <p className="px-1 text-[10px] leading-relaxed text-teal-100/40">
                The tool position, axis and visit order come from{" "}
                <span className="font-mono text-teal-200/70">robot_commands</span>; the keep-out
                volume is the one the safety report computed. The gantry itself is illustrative —
                the plan carries no robot geometry, so its rails are sized from that boundary and
                its axes are Cartesian. A clinical system is a serial arm with a different
                workspace.
              </p>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
