"use client";

/**
 * Measurement table.
 *
 * The unavailable list is rendered as prominently as the values, with the engine's
 * own reason attached. On a real knee MRI thirteen measurements cannot be computed —
 * HKA, MLDFA and MPTA among them — because the field of view contains neither the
 * femoral head nor the ankle. Omitting them would imply they were never wanted;
 * showing them as unavailable with the reason is the difference between a stated gap
 * and a silent hole.
 *
 * Grouped by anatomical region rather than presented as one 42-row list, because a
 * flat list of forty numbers is data rather than information.
 */

import { useMemo, useState } from "react";
import type { MeasurementSet } from "@/lib/types";

const GROUPS: Array<{ title: string; match: (name: string) => boolean }> = [
  { title: "Femur — sizing", match: (n) => n.startsWith("femoral_") && !n.includes("cartilage") },
  { title: "Femur — rotation", match: (n) => n.includes("condylar") || n.includes("twist") },
  { title: "Tibia — sizing", match: (n) => n.startsWith("tibial_") && !n.includes("cartilage") },
  { title: "Tibia — geometry", match: (n) => n.includes("plateau") || n.includes("slope") },
  { title: "Alignment", match: (n) => ["hka", "mldfa", "mpta", "jlca"].some((k) => n.startsWith(k)) },
  { title: "Cartilage", match: (n) => n.includes("cartilage") },
  { title: "Other", match: () => true },
];

export default function MeasurementsPanel({ data }: { data: MeasurementSet | null }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const grouped = useMemo(() => {
    if (!data) return [];
    const pool = [...(data.measurements ?? [])];
    const out: Array<{ title: string; items: typeof pool }> = [];
    for (const group of GROUPS) {
      const items = pool.filter((m) => group.match(m.name));
      for (const item of items) pool.splice(pool.indexOf(item), 1);
      if (items.length) out.push({ title: group.title, items });
    }
    return out;
  }, [data]);

  if (!data) {
    return (
      <div className="card">
        <div className="card-head">
          <span className="card-title">Measurements</span>
        </div>
        <div className="card-body">
          <p className="text-[12px] text-slate-500">Not available for this case.</p>
        </div>
      </div>
    );
  }

  const unavailable = Object.entries(data.unavailable ?? {});

  return (
    <>
      <div className="card">
        <div className="card-head">
          <span className="card-title">Measurements</span>
          <span className="card-sub">{data.n_measurements} computed</span>
        </div>
        <div className="card-body">
          <p className="mb-3 text-[11px] leading-relaxed text-slate-500">
            Click any name to see how it is defined.
          </p>
          {grouped.map((group) => (
            <section key={group.title} className="mb-3 last:mb-0">
              <h4 className="section-label">{group.title}</h4>
              {group.items.map((m) => (
                <div key={m.name}>
                  <div className="row">
                    <button
                      type="button"
                      className="row-key text-left hover:text-slate-200"
                      onClick={() => setExpanded(expanded === m.name ? null : m.name)}
                      aria-expanded={expanded === m.name}
                    >
                      {m.display_name}
                      {m.caveat && (
                        <span className="ml-1 text-amber-400" title={m.caveat}>
                          !
                        </span>
                      )}
                    </button>
                    <span
                      className={`row-val ${m.in_reference_range === false ? "text-amber-400" : ""}`}
                      title={
                        m.in_reference_range === false
                          ? "Outside the published reference range"
                          : m.in_reference_range == null
                            ? "No reference range is published for this measurement"
                            : undefined
                      }
                    >
                      {m.formatted}
                      {m.quantity.uncertainty != null && (
                        <span className="text-slate-500"> ±{m.quantity.uncertainty.toFixed(1)}</span>
                      )}
                    </span>
                  </div>
                  {expanded === m.name && (
                    <p className="mb-2 mt-1 rounded bg-slate-800/60 px-2.5 py-2 text-[11px] leading-relaxed text-slate-400">
                      {m.method}
                      {m.sign_convention && (
                        <>
                          <br />
                          <span className="text-slate-300">Sign: </span>
                          {m.sign_convention}
                        </>
                      )}
                      {m.caveat && (
                        <>
                          <br />
                          <span className="text-amber-400">Caveat: </span>
                          {m.caveat}
                        </>
                      )}
                    </p>
                  )}
                </div>
              ))}
            </section>
          ))}
        </div>
      </div>

      {unavailable.length > 0 && (
        <div className="card border-amber-900/60">
          <div className="card-head">
            <span className="card-title text-amber-400">Could not be measured</span>
            <span className="card-sub">{unavailable.length}</span>
          </div>
          <div className="card-body">
            <p className="mb-2.5 text-[11px] leading-relaxed text-slate-400">
              Shown rather than omitted, so a missing value is never mistaken for a normal
              one.
            </p>
            {unavailable.map(([name, reason]) => (
              <div key={name} className="mb-2 last:mb-0">
                <div className="text-[12px] text-slate-300">{name}</div>
                <p className="text-[11px] leading-snug text-slate-500">{reason}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
