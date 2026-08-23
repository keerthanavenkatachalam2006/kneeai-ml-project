"use client";

/**
 * A draggable, collapsible panel that floats over the viewer.
 *
 * The previous layout put every panel in a fixed sidebar, which had two costs: the
 * viewer was squeezed into two thirds of the width, and anything past the second card
 * was below the fold — the robot cell simulation was being missed entirely for that
 * reason. Floating the panels lets the anatomy use the full canvas and lets the reader
 * put the information where it does not cover the thing it describes.
 *
 * Dragging is deliberately clamped to the viewport. A panel dragged off-screen cannot be
 * recovered without a reload, so the header is always kept reachable.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface PanelPlacement {
  x: number;
  y: number;
}

export default function FloatingPanel({
  title,
  subtitle,
  accent = "teal",
  initial,
  width = 340,
  collapsible = true,
  onClose,
  children,
}: {
  title: string;
  subtitle?: string;
  /** Colours the header rule and the collapse affordance. */
  accent?: "teal" | "amber" | "rose" | "emerald";
  initial: PanelPlacement;
  width?: number;
  collapsible?: boolean;
  onClose?: () => void;
  children: React.ReactNode;
}) {
  const [pos, setPos] = useState<PanelPlacement>(initial);
  const [collapsed, setCollapsed] = useState(false);
  const [dragging, setDragging] = useState(false);

  const ref = useRef<HTMLDivElement>(null);
  const grab = useRef({ dx: 0, dy: 0 });

  const accentRing: Record<string, string> = {
    teal: "ring-teal-400/25",
    amber: "ring-amber-400/30",
    rose: "ring-rose-400/30",
    emerald: "ring-emerald-400/30",
  };
  const accentDot: Record<string, string> = {
    teal: "bg-teal-400",
    amber: "bg-amber-400",
    rose: "bg-rose-400",
    emerald: "bg-emerald-400",
  };

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    // Ignore drags that start on a control inside the header.
    if ((e.target as HTMLElement).closest("button")) return;
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    grab.current = { dx: e.clientX - rect.left, dy: e.clientY - rect.top };
    setDragging(true);
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  }, []);

  useEffect(() => {
    if (!dragging) return;

    const move = (e: PointerEvent) => {
      const w = ref.current?.offsetWidth ?? width;
      // Clamp so at least the header stays on screen and reachable.
      const maxX = window.innerWidth - 60;
      const maxY = window.innerHeight - 40;
      setPos({
        x: Math.max(-w + 60, Math.min(maxX, e.clientX - grab.current.dx)),
        y: Math.max(0, Math.min(maxY, e.clientY - grab.current.dy)),
      });
    };
    const up = () => setDragging(false);

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [dragging, width]);

  return (
    <div
      ref={ref}
      style={{ left: pos.x, top: pos.y, width }}
      className={`pointer-events-auto absolute z-20 overflow-hidden rounded-panel border border-white/10 bg-clinic-900/90 shadow-float ring-1 ring-inset backdrop-blur-md ${
        accentRing[accent]
      } ${dragging ? "cursor-grabbing select-none" : ""}`}
    >
      <div
        onPointerDown={onPointerDown}
        className="flex cursor-grab items-center justify-between gap-2 border-b border-white/5 px-3 py-2 active:cursor-grabbing"
      >
        <div className="flex min-w-0 items-center gap-2">
          <span aria-hidden className={`h-1.5 w-1.5 shrink-0 rounded-full ${accentDot[accent]}`} />
          <div className="min-w-0">
            <p className="truncate text-[12px] font-semibold leading-tight tracking-tight text-white">
              {title}
            </p>
            {subtitle && (
              <p className="truncate text-[10px] leading-tight text-teal-200/40">{subtitle}</p>
            )}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-0.5">
          {collapsible && (
            <button
              type="button"
              onClick={() => setCollapsed((c) => !c)}
              aria-label={collapsed ? `Expand ${title}` : `Collapse ${title}`}
              aria-expanded={!collapsed}
              className="flex h-6 w-6 items-center justify-center rounded text-teal-200/50 transition-colors hover:bg-white/10 hover:text-white"
            >
              <span aria-hidden className="text-[13px] leading-none">
                {collapsed ? "+" : "−"}
              </span>
            </button>
          )}
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label={`Close ${title}`}
              className="flex h-6 w-6 items-center justify-center rounded text-teal-200/50 transition-colors hover:bg-rose-500/20 hover:text-rose-300"
            >
              <span aria-hidden className="text-[13px] leading-none">
                ×
              </span>
            </button>
          )}
        </div>
      </div>

      {!collapsed && (
        <div className="max-h-[calc(100vh-13rem)] overflow-y-auto overscroll-contain">
          {children}
        </div>
      )}
    </div>
  );
}
