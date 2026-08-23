"use client";

/**
 * Fades a block in the first time it scrolls into view, then stops observing.
 *
 * One effect, no library, and nothing at all under `prefers-reduced-motion` — the content
 * is rendered visible from the start in that case, rather than depending on JavaScript to
 * reveal it. If the observer never fires (no JS), the fallback is also visible, so the page
 * can never end up blank.
 */

import { useEffect, useRef, useState } from "react";

export default function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  /** Milliseconds, for staggering siblings. */
  delay?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(true);
      return;
    }
    const node = ref.current;
    if (!node) return;

    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        io.disconnect();
        window.setTimeout(() => setShown(true), delay);
      },
      { rootMargin: "-6% 0px -6% 0px" },
    );
    io.observe(node);
    return () => io.disconnect();
  }, [delay]);

  return (
    <div
      ref={ref}
      /* `min-w-0` matters: as a grid child the default `min-width: auto` lets a wide
         child (the terminal block) stretch its column and push the whole page sideways
         on a phone. */
      className={`${className} min-w-0 transition-[opacity,transform] duration-700 ease-out ${
        shown ? "translate-y-0 opacity-100" : "translate-y-3 opacity-0"
      }`}
    >
      {children}
    </div>
  );
}
