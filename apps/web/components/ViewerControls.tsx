"use client";

/**
 * Compact viewer controls, overlaid on the canvas rather than occupying a column.
 *
 * Visibility and opacity are keyed by mesh `path`, not `structure`. The manifest has
 * three entries called `femur` (anatomy, resected, component), so keying by name
 * would make hiding the intact femur also hide the implant.
 *
 * Collapsed to one row of toggles by default. During a demo people hide cartilage to
 * look at bone; they do not adjust eleven opacity sliders. So that action is one
 * click and the rest is one click away.
 */

import { useState } from "react";
import { prettyStructure, structureColor } from "@/lib/api";
import type { MeshManifest } from "@/lib/types";
import type { ViewerSettings } from "./Viewer3D";

interface Props {
  manifest: MeshManifest;
  settings: ViewerSettings;
  onChange: (next: ViewerSettings) => void;
  onScreenshot: () => void;
  onReset: () => void;
  hasCuts: boolean;
  hasCommands: boolean;
}

const AXES = ["Sagittal", "Coronal", "Axial"] as const;

function Pill({
  label,
  active,
  onClick,
  disabled,
  hint,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  hint?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      title={hint}
      className={[
        "rounded-md border px-2.5 py-1.5 text-[11px] font-medium transition-colors",
        disabled
          ? "cursor-not-allowed border-slate-800 text-slate-600"
          : active
            ? "border-sky-500 bg-sky-500/20 text-sky-300"
            : "border-slate-700 text-slate-400 hover:bg-slate-800",
      ].join(" ")}
    >
      {label}
    </button>
  );
}

export default function ViewerControls({
  manifest,
  settings,
  onChange,
  onScreenshot,
  onReset,
  hasCuts,
  hasCommands,
}: Props) {
  const [open, setOpen] = useState(false);

  const all = manifest.meshes ?? [];
  const anatomy = all.filter((m) => m.category === "anatomy");
  const soft = anatomy.filter((m) => /cartilage|meniscus/.test(m.structure));
  const softHidden = soft.length > 0 && soft.every((m) => settings.visible[m.path] === false);

  function setPaths(paths: string[], value: boolean) {
    const visible = { ...settings.visible };
    for (const p of paths) visible[p] = value;
    onChange({ ...settings, visible });
  }

  return (
    <div className="overlay p-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <Pill
          label="Landmarks"
          active={settings.showLandmarks}
          onClick={() => onChange({ ...settings, showLandmarks: !settings.showLandmarks })}
          hint="Marker colour encodes detection confidence"
        />
        <Pill
          label="Cut planes"
          active={settings.showCutPlanes}
          disabled={!hasCuts}
          onClick={() => onChange({ ...settings, showCutPlanes: !settings.showCutPlanes })}
        />
        <Pill
          label="Implant"
          active={settings.showImplant}
          onClick={() => onChange({ ...settings, showImplant: !settings.showImplant })}
          hint="Metal components seated on the resected bone"
        />
        <Pill
          label="After resection"
          active={settings.resectedView}
          onClick={() => onChange({ ...settings, resectedView: !settings.resectedView })}
          hint="Swaps intact bone for the resected mesh"
        />
        <Pill
          label={softHidden ? "Show soft tissue" : "Bone only"}
          active={softHidden}
          onClick={() => setPaths(soft.map((m) => m.path), softHidden)}
          hint="Hide cartilage and meniscus"
        />
        <Pill
          label="Clip"
          active={settings.clipEnabled}
          onClick={() => onChange({ ...settings, clipEnabled: !settings.clipEnabled })}
        />
        {hasCommands && (
          <Pill
            label="Boundary"
            active={settings.showBoundary}
            onClick={() => onChange({ ...settings, showBoundary: !settings.showBoundary })}
            hint="Robot keep-out volume"
          />
        )}

        <span className="mx-1 h-5 w-px bg-slate-700" aria-hidden />

        <button type="button" className="btn-ghost !px-2 !py-1.5" onClick={onReset}>
          Reset view
        </button>
        <button type="button" className="btn-ghost !px-2 !py-1.5" onClick={onScreenshot}>
          Screenshot
        </button>
        <button
          type="button"
          className="btn-ghost !px-2 !py-1.5"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          {open ? "Hide structures" : "Structures"}
        </button>
      </div>

      <p className="mt-1.5 text-[10px] text-slate-500">
        Click any surface for coordinates and mesh detail · drag to orbit · scroll to zoom
      </p>

      {settings.clipEnabled && (
        <div className="mt-2 flex items-center gap-2 border-t border-slate-800 pt-2">
          <select
            aria-label="Clipping axis"
            value={settings.clipAxis}
            onChange={(e) => onChange({ ...settings, clipAxis: Number(e.target.value) as 0 | 1 | 2 })}
            className="rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-[11px]"
          >
            {AXES.map((a, i) => (
              <option key={a} value={i}>
                {a}
              </option>
            ))}
          </select>
          <input
            aria-label="Clip position in millimetres"
            type="range"
            min={-100}
            max={100}
            value={settings.clipPosition}
            onChange={(e) => onChange({ ...settings, clipPosition: Number(e.target.value) })}
            className="h-1 flex-1 accent-sky-500"
          />
          <span className="w-14 shrink-0 text-right font-mono text-[11px] text-slate-500">
            {settings.clipPosition} mm
          </span>
        </div>
      )}

      {open && (
        <div className="mt-2 max-h-56 space-y-1.5 overflow-y-auto border-t border-slate-800 pt-2">
          {anatomy.map((entry) => {
            const visible = settings.visible[entry.path] !== false;
            const opacity = settings.opacity[entry.path] ?? 1;
            return (
              <div key={entry.path} className="flex items-center gap-2">
                <span
                  aria-hidden
                  className="h-2.5 w-2.5 shrink-0 rounded-sm ring-1 ring-black/40"
                  style={{ backgroundColor: structureColor(entry.structure) }}
                />
                <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={visible}
                    onChange={(e) =>
                      onChange({
                        ...settings,
                        visible: { ...settings.visible, [entry.path]: e.target.checked },
                      })
                    }
                    className="h-3.5 w-3.5 accent-sky-500"
                  />
                  <span className="truncate text-[11px] text-slate-300">
                    {prettyStructure(entry.structure)}
                  </span>
                </label>
                <input
                  aria-label={`${prettyStructure(entry.structure)} opacity`}
                  type="range"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={opacity}
                  disabled={!visible}
                  onChange={(e) =>
                    onChange({
                      ...settings,
                      opacity: { ...settings.opacity, [entry.path]: Number(e.target.value) },
                    })
                  }
                  className="h-1 w-20 shrink-0 accent-sky-500 disabled:opacity-30"
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
