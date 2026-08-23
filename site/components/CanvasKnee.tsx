"use client";

/**
 * Hero background: a point-cloud knee on a bare canvas.
 *
 * Generated, not loaded. Condyle spheres, a tibial plateau and two shafts are sampled as
 * surface points, rotated slowly and projected by hand, while a scan plane sweeps up the
 * volume and brightens the points it crosses. That sweep is the one idea the whole project
 * rests on: a stack of 2D slices becomes a 3D surface you can measure.
 *
 * Two palettes. `light` draws navy ink on a white page for this site; `dark` keeps the
 * product's teal-on-near-black. Both are the same geometry.
 *
 * Deliberately cheap — one 2D canvas, no three.js, ~1400 points, one fill per point — and
 * completely idle when the tab is hidden, the element is off-screen, or the reader has
 * asked for reduced motion, in which case it draws a single static frame.
 */

import { useEffect, useRef } from "react";

interface P3 {
  x: number;
  y: number;
  z: number;
  /** 0 = bone, 1 = cartilage. Only changes the colour. */
  soft: number;
}

type Variant = "light" | "dark";

const PALETTE: Record<
  Variant,
  { grid: string; bone: (a: number) => string; soft: (a: number) => string; lit: (a: number) => string; rule: string; halo: string; label: string }
> = {
  light: {
    grid: "rgba(38,90,112,0.07)",
    bone: (a) => `rgba(19,66,86,${a})`,
    soft: (a) => `rgba(18,137,186,${a})`,
    lit: (a) => `rgba(10,111,155,${a})`,
    rule: "rgba(18,137,186,0.45)",
    halo: "rgba(18,137,186,",
    label: "rgba(74,115,135,0.75)",
  },
  dark: {
    grid: "rgba(45,212,191,0.035)",
    bone: (a) => `rgba(45,212,191,${a})`,
    soft: (a) => `rgba(96,190,232,${a})`,
    lit: (a) => `rgba(190,247,240,${a})`,
    rule: "rgba(45,212,191,0.35)",
    halo: "rgba(45,212,191,",
    label: "rgba(153,246,228,0.5)",
  },
};

/** Deterministic PRNG, so the cloud is identical on every render and reload. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function buildKnee(): P3[] {
  const rand = rng(20260823);
  const pts: P3[] = [];

  const onSphere = (cx: number, cy: number, cz: number, r: number, n: number, soft: number) => {
    for (let i = 0; i < n; i++) {
      const u = rand() * 2 - 1;
      const t = rand() * Math.PI * 2;
      const s = Math.sqrt(1 - u * u);
      pts.push({
        x: cx + r * s * Math.cos(t),
        y: cy + r * u * 0.82,
        z: cz + r * s * Math.sin(t),
        soft,
      });
    }
  };

  const onTube = (
    cx: number,
    y0: number,
    y1: number,
    r0: number,
    r1: number,
    n: number,
    soft: number,
  ) => {
    for (let i = 0; i < n; i++) {
      const f = rand();
      const r = r0 + (r1 - r0) * f;
      pts.push({
        x: cx + r * Math.cos(rand() * Math.PI * 2),
        y: y0 + (y1 - y0) * f,
        z: r * 0.8 * Math.sin(rand() * Math.PI * 2),
        soft,
      });
    }
  };

  // Femoral condyles: the two lobes that make a knee read as a knee.
  onSphere(-0.3, -0.16, 0, 0.29, 210, 0);
  onSphere(0.3, -0.16, 0, 0.29, 210, 0);
  // The cartilage shell over them.
  onSphere(-0.3, -0.18, 0, 0.32, 70, 1);
  onSphere(0.3, -0.18, 0, 0.32, 70, 1);
  // Femoral shaft, rising out of frame.
  onTube(0, 0.05, 1.25, 0.3, 0.19, 260, 0);
  // Tibial plateau and shaft.
  onSphere(0, -0.62, 0, 0.36, 150, 0);
  onSphere(0, -0.55, 0, 0.38, 60, 1);
  onTube(0, -0.7, -1.5, 0.26, 0.17, 230, 0);

  return pts;
}

export default function CanvasKnee({
  className = "",
  variant = "light",
}: {
  className?: string;
  variant?: Variant;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const colours = PALETTE[variant];
    const points = buildKnee();
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let raf = 0;
    let onScreen = true;
    let t = reduced ? 1.15 : 0;

    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = () => {
      if (width === 0 || height === 0) return;
      ctx.clearRect(0, 0, width, height);

      const cx = width * 0.5;
      const cy = height * 0.52;
      const scale = Math.min(width, height) * 0.42;

      // --- Grid, very low contrast, behind everything. ---
      const cell = 34;
      ctx.lineWidth = 1;
      ctx.strokeStyle = colours.grid;
      for (let x = ((cx % cell) + cell) % cell; x < width; x += cell) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = ((cy % cell) + cell) % cell; y < height; y += cell) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      const angle = t * 0.22;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      // The scan plane, sweeping bottom to top in model units.
      const sweep = -1.5 + ((t * 0.42) % 2.9);

      // Painter's order, so nearer points sit on top.
      const projected = points
        .map((p) => {
          const x = p.x * cos - p.z * sin;
          const z = p.x * sin + p.z * cos;
          const persp = 1 / (1 + z * 0.28);
          return {
            sx: cx + x * scale * persp,
            sy: cy - p.y * scale * persp,
            depth: z,
            persp,
            soft: p.soft,
            near: Math.abs(p.y - sweep),
          };
        })
        .sort((a, b) => b.depth - a.depth);

      for (const p of projected) {
        const fade = 0.32 + 0.68 * ((1.4 - p.depth) / 2.4);
        const lit = Math.max(0, 1 - p.near / 0.075);
        const r = (p.soft ? 1.1 : 1.5) * p.persp + lit * 1.5;
        const alpha = Math.min(0.9, (p.soft ? 0.28 : 0.42) * fade + lit * 0.5);

        ctx.beginPath();
        ctx.arc(p.sx, p.sy, Math.max(0.4, r), 0, Math.PI * 2);
        ctx.fillStyle = p.soft
          ? colours.soft(alpha)
          : lit > 0.15
            ? colours.lit(alpha)
            : colours.bone(alpha);
        ctx.fill();
      }

      // --- The plane itself: a thin rule with a soft halo and a slice index. ---
      const sy = cy - sweep * scale;
      const grad = ctx.createLinearGradient(0, sy - 26, 0, sy + 26);
      grad.addColorStop(0, `${colours.halo}0)`);
      grad.addColorStop(0.5, `${colours.halo}0.1)`);
      grad.addColorStop(1, `${colours.halo}0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(cx - scale * 1.5, sy - 26, scale * 3, 52);

      ctx.strokeStyle = colours.rule;
      ctx.beginPath();
      ctx.moveTo(cx - scale * 1.35, sy);
      ctx.lineTo(cx + scale * 1.35, sy);
      ctx.stroke();

      ctx.fillStyle = colours.label;
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillText(
        `slice ${String(Math.round(((sweep + 1.5) / 2.9) * 159)).padStart(3, "0")} / 160`,
        cx + scale * 1.35 - 92,
        sy - 8,
      );
    };

    const frame = () => {
      t += 1 / 60;
      draw();
      raf = requestAnimationFrame(frame);
    };
    const start = () => {
      if (raf || reduced || !onScreen || document.hidden) return;
      raf = requestAnimationFrame(frame);
    };
    const stop = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    };

    resize();
    draw();
    start();

    const onResize = () => {
      resize();
      draw();
    };
    const onVisibility = () => (document.hidden ? stop() : start());

    const io = new IntersectionObserver(
      ([entry]) => {
        onScreen = entry.isIntersecting;
        if (onScreen) start();
        else stop();
      },
      { threshold: 0 },
    );
    io.observe(canvas);

    window.addEventListener("resize", onResize);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      stop();
      io.disconnect();
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [variant]);

  return <canvas ref={ref} aria-hidden className={className} />;
}
