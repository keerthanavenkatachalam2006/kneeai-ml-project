"use client";

/**
 * Sticky navigation.
 *
 * Transparent over the hero and solid once the page scrolls, which is the pattern a product
 * site uses so the hero reads as full-bleed. The current-section highlight is driven by an
 * IntersectionObserver over the section elements rather than scroll arithmetic, so it stays
 * correct when sections differ wildly in height.
 *
 * The mobile menu is a plain disclosure, not a drawer: five links do not need an overlay.
 */

import { useEffect, useState } from "react";

import Icon from "@/components/Icon";

const LINKS = [
  { id: "product", label: "Product" },
  { id: "how", label: "Technology" },
  { id: "screens", label: "Screens" },
  { id: "evidence", label: "Evidence" },
  { id: "faq", label: "FAQ" },
];

export default function Nav() {
  const [solid, setSolid] = useState(false);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState("");

  useEffect(() => {
    const onScroll = () => setSolid(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (visible) setActive(visible.target.id);
      },
      { rootMargin: "-45% 0px -50% 0px" },
    );
    LINKS.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    });

    return () => {
      window.removeEventListener("scroll", onScroll);
      io.disconnect();
    };
  }, []);

  return (
    <header
      className={`sticky top-0 z-40 transition-all duration-200 ${
        solid ? "border-b border-ink-200/70 bg-white/90 backdrop-blur-md" : "bg-transparent"
      }`}
    >
      <div className="shell flex items-center justify-between gap-4 py-3.5">
        <a href="#top" className="flex items-center gap-2.5" aria-label="MORP home">
          <span
            aria-hidden
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white"
          >
            <span className="h-2.5 w-2.5 rounded-full bg-white" />
          </span>
          <span className="text-[17px] font-bold tracking-[0.14em] text-ink-900">MORP</span>
        </a>

        <nav aria-label="Sections" className="hidden items-center gap-8 lg:flex">
          {LINKS.map((l) => (
            <a
              key={l.id}
              href={`#${l.id}`}
              aria-current={active === l.id ? "true" : undefined}
              className={`text-[13px] font-medium transition-colors ${
                active === l.id ? "text-brand-600" : "text-ink-600 hover:text-brand-600"
              }`}
            >
              {l.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <a href="#run" className="hidden btn-outline !px-4 !py-2 sm:inline-flex">
            Run it locally
          </a>
          <a href="#screens" className="btn-primary !px-4 !py-2">
            See it working
          </a>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            aria-label="Toggle navigation"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-ink-200 text-ink-600 lg:hidden"
          >
            <Icon name="chevron" className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
        </div>
      </div>

      {open && (
        <nav
          aria-label="Sections"
          className="border-t border-ink-200/70 bg-white px-6 py-3 lg:hidden"
        >
          {LINKS.map((l) => (
            <a
              key={l.id}
              href={`#${l.id}`}
              onClick={() => setOpen(false)}
              className="block py-2 text-[13.5px] font-medium text-ink-700"
            >
              {l.label}
            </a>
          ))}
          <a href="#run" onClick={() => setOpen(false)} className="block py-2 text-[13.5px] font-medium text-brand-600">
            Run it locally
          </a>
        </nav>
      )}
    </header>
  );
}
