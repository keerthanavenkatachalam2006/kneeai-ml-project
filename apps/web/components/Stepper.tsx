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
    <nav aria-label="Workflow" className="px-3 pt-3">
      <ol className="mx-auto flex max-w-[1700px] items-stretch gap-1.5 overflow-x-auto rounded-panel border border-white/5 bg-clinic-900/70 p-1.5 shadow-panel backdrop-blur-sm">
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
                  "group flex w-full items-center gap-2 rounded-[10px] px-2.5 py-2 text-left transition-all duration-150",
                  disabled
                    ? "cursor-not-allowed opacity-30"
                    : isCurrent
                      ? "bg-teal-400/15 ring-1 ring-inset ring-teal-400/40"
                      : "hover:bg-white/[0.05]",
                ].join(" ")}
              >
                <span
                  aria-hidden
                  className={[
                    "flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-full text-[10px] font-bold tabular-nums transition-colors",
                    isCurrent
                      ? "bg-teal-400 text-clinic-950"
                      : isPast
                        ? "bg-teal-400/25 text-teal-200"
                        : "border border-white/15 text-teal-100/40",
                  ].join(" ")}
                >
                  {isPast ? "✓" : i + 1}
                </span>
                <span className="min-w-0">
                  <span
                    className={[
                      "block truncate text-[12px] font-medium leading-tight tracking-tight",
                      isCurrent ? "text-white" : "text-teal-50/70",
                    ].join(" ")}
                  >
                    {step.label}
                  </span>
                  <span className="hidden truncate text-[10px] leading-tight text-teal-200/35 xl:block">
                    {disabled ? step.blockedReason : step.hint}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
