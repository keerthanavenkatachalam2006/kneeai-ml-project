"use client";

/**
 * FAQ accordion.
 *
 * Native `details`/`summary` would be fewer lines, but the animated chevron and the
 * single-open behaviour are easier to control explicitly, and the ARIA is the same either
 * way. Answers stay in the DOM only while open, so nothing hidden is read out.
 */

import { useState } from "react";

import Icon from "@/components/Icon";
import { FAQ } from "@/content/project";

export default function Faq() {
  const [open, setOpen] = useState<number | null>(0);

  return (
    <div className="divide-y divide-ink-200/70 overflow-hidden rounded-card border border-ink-200/70 bg-white shadow-card">
      {FAQ.map((item, i) => {
        const isOpen = open === i;
        return (
          <div key={item.q}>
            <h3>
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : i)}
                aria-expanded={isOpen}
                className="flex w-full items-center justify-between gap-6 px-5 py-4 text-left md:px-7 md:py-5"
              >
                <span
                  className={`text-[14.5px] font-semibold tracking-[-0.01em] transition-colors ${
                    isOpen ? "text-brand-700" : "text-ink-900"
                  }`}
                >
                  {item.q}
                </span>
                <span
                  className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border transition-all duration-200 ${
                    isOpen
                      ? "rotate-180 border-brand-200 bg-brand-50 text-brand-600"
                      : "border-ink-200 text-ink-400"
                  }`}
                >
                  <Icon name="chevron" className="h-4 w-4" />
                </span>
              </button>
            </h3>
            {isOpen && (
              <div className="px-5 pb-5 md:px-7 md:pb-6">
                <p className="body max-w-reading">{item.a}</p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
