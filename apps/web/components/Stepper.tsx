"use client";

/**
 * Workflow stepper.
 *
 * The clinical workflow is linear, so the interface is too. Showing every panel at
 * once forces a viewer to work out for themselves which stage a case has reached;
 * one step at a time makes the progression the primary information.
 *
 * A step that cannot be entered is disabled with a reason on hover, rather than
 * hidden. "Plan is not ready because the case has no plan" is useful; a missing
 * button is not.
 */

export interface Step {
  id: string;
  label: string;
  hint: string;
  ready: boolean;
  blockedReason?: string;
}

export default function Stepper({
  steps,
  current,
  onSelect,
}: {
  steps: Step[];
  current: string;
  onSelect: (id: string) => void;
}) {
  const currentIndex = steps.findIndex((s) => s.id === current);

  return (
    <nav aria-label="Workflow" className="border-b border-slate-800 bg-slate-900/60">
      <ol className="mx-auto flex max-w-[1700px] items-stretch gap-px px-4">
        {steps.map((step, i) => {
          const isCurrent = step.id === current;
          const isPast = i < currentIndex;
          const disabled = !step.ready;

          return (
            <li key={step.id} className="min-w-0 flex-1">
              <button
                type="button"
                disabled={disabled}
                aria-current={isCurrent ? "step" : undefined}
                title={disabled ? step.blockedReason : step.hint}
                onClick={() => onSelect(step.id)}
                className={[
                  "group relative flex w-full items-center gap-2.5 px-3 py-3 text-left transition-colors",
                  disabled
                    ? "cursor-not-allowed opacity-40"
                    : isCurrent
                      ? "bg-slate-800/80"
                      : "hover:bg-slate-800/40",
                ].join(" ")}
              >
                <span
                  aria-hidden
                  className={[
                    "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold tabular-nums",
                    isCurrent
                      ? "bg-sky-500 text-white"
                      : isPast
                        ? "bg-emerald-600/80 text-white"
                        : "border border-slate-600 text-slate-400",
                  ].join(" ")}
                >
                  {isPast ? "✓" : i + 1}
                </span>
                <span className="min-w-0">
                  <span
                    className={[
                      "block truncate text-[13px] font-medium leading-tight",
                      isCurrent ? "text-slate-50" : "text-slate-300",
                    ].join(" ")}
                  >
                    {step.label}
                  </span>
                  <span className="hidden truncate text-[11px] leading-tight text-slate-500 lg:block">
                    {disabled ? step.blockedReason : step.hint}
                  </span>
                </span>
                {isCurrent && (
                  <span aria-hidden className="absolute inset-x-0 bottom-0 h-0.5 bg-sky-500" />
                )}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
