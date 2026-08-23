/**
 * MORP showcase site — one page that explains the project.
 *
 * This is not the product. It makes no API calls, ships no model and contains no patient
 * data: the only imagery is a set of captures of the running workspace. The product lives in
 * `apps/web` and is run locally, which the last section explains.
 *
 * The page is laid out in the register a commercial orthopaedic-planning site uses — framed
 * product shots, a window switcher, capability cards, a figures band, an FAQ — because that
 * is the shape a reader expects and it makes the work legible quickly. Two deliberate
 * departures from that genre: the regulatory status sits above the navigation rather than in
 * the footer, and the slot where such a site puts a surgeon testimonial holds a quote from
 * our own design brief, since no clinician has endorsed this prototype.
 *
 * Every figure comes from `content/project.ts`, which names the artefact it was copied from.
 */

import CanvasKnee from "@/components/CanvasKnee";
import Faq from "@/components/Faq";
import Icon, { type IconName } from "@/components/Icon";
import Nav from "@/components/Nav";
import ProductSwitcher from "@/components/ProductSwitcher";
import Reveal from "@/components/Reveal";
import Screens from "@/components/Screens";
import {
  DATASET,
  DICE,
  FEATURES,
  FOOTER_LINKS,
  IMPACT,
  LIMITS,
  MAE,
  PIPELINE,
  PRINCIPLES,
  PROJECT,
  PULL_QUOTE,
  REQUIREMENTS,
  STACK,
  TRAINING,
} from "@/content/project";

const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const STATUS: Record<string, { label: string; className: string }> = {
  done: { label: "delivered", className: "border-emerald-200 bg-emerald-50 text-emerald-700" },
  partial: { label: "partial", className: "border-amber-200 bg-amber-50 text-amber-700" },
  gap: { label: "blocked", className: "border-rose-200 bg-rose-50 text-rose-700" },
};

/** The three claims directly under the hero, mirroring a product site's headline features. */
const HERO_POINTS: Array<{ icon: IconName; title: string; body: string }> = [
  {
    icon: "scan",
    title: "Automatic segmentation",
    body: "Femur, tibia and cartilage labelled from the MR volume in 30–40 seconds on CPU.",
  },
  {
    icon: "ruler",
    title: "Automatic morphometry",
    body: "Twenty-six measurements, each with its method, landmarks and sign convention attached.",
  },
  {
    icon: "implant",
    title: "Explainable templating",
    body: "Implant candidates ranked with the overhang and undercoverage that decided the order.",
  },
];

export default function Page() {
  return (
    <>
      <Nav />

      <main id="top">
        {/* ======================= Hero ======================= */}
        <section className="relative overflow-hidden bg-white">
          {/* Soft brand wash and the point cloud, both behind the type. */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                "radial-gradient(60% 55% at 88% 12%, #eaf6fc 0%, rgba(234,246,252,0) 70%), radial-gradient(50% 45% at 8% 90%, #f0fbf9 0%, rgba(240,251,249,0) 70%)",
            }}
          />
          <CanvasKnee
            variant="light"
            className="pointer-events-none absolute right-[-8%] top-[6%] hidden h-[560px] w-[46%] opacity-[0.5] lg:block"
          />

          <div className="shell relative grid items-center gap-14 py-16 md:py-24 lg:grid-cols-[1.02fr_0.98fr]">
            <div>
              <p className="pill mb-6">
                <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-accent-500" />
                {PROJECT.expansion}
              </p>

              <h1 className="h1">
                Patient-specific knee planning,
                <span className="text-brand-600"> explained</span>.
              </h1>

              <p className="lede mt-7">
                MORP reads one knee MR volume and returns a 3D model, twenty-six morphometric
                measurements, an osteoarthritis assessment, ranked implant candidates, a
                resection plan and a simulated robot cell. Every number arrives with the
                reasoning that produced it — and every number it cannot produce says so, with
                the reason.
              </p>

              <div className="mt-9 flex flex-wrap items-center gap-3">
                <a href="#screens" className="btn-primary">
                  See it working
                </a>
                <a href="#run" className="btn-outline">
                  Run it locally
                </a>
              </div>

              <p className="mt-7 max-w-reading text-[12px] leading-relaxed text-ink-400">
                Runs entirely on your machine: a loopback API over a directory of artefacts and
                a local workspace. Nothing is uploaded anywhere.
              </p>
            </div>

            {/* Hero capture, tilted very slightly so it reads as an object on the page. */}
            <Reveal>
              <figure className="frame lg:translate-y-2">
                <div className="frame-bar">
                  <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-rose-400/60" />
                  <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-amber-400/60" />
                  <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-emerald-400/60" />
                  <span className="ml-2 truncate font-mono text-[10.5px] text-white/45">
                    morp · implant &amp; plan · case oaizib_405
                  </span>
                </div>
                <div className="relative aspect-[16/10] w-full bg-ink-900">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${BASE}/shots/plan.png`}
                    alt="The planning workspace showing ranked implant candidates, per-side overhang and the resection plan over a 3D knee."
                    className="h-full w-full object-cover object-top"
                  />
                </div>
              </figure>
            </Reveal>
          </div>

          {/* Three headline capabilities on a tinted band, closing the hero. */}
          <div className="relative border-y border-ink-200/70 bg-paper-soft">
            <div className="shell grid gap-8 py-10 md:grid-cols-3 md:gap-12">
              {HERO_POINTS.map((p, i) => (
                <Reveal key={p.title} delay={i * 80}>
                  <div className="flex gap-4">
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white text-brand-600 shadow-card">
                      <Icon name={p.icon} />
                    </span>
                    <div>
                      <h2 className="h3 text-[14.5px]">{p.title}</h2>
                      <p className="body mt-1">{p.body}</p>
                    </div>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* ======================= Product ======================= */}
        <section id="product" className="section bg-white">
          <div className="shell">
            <Reveal>
              <div className="mb-12 max-w-2xl">
                <p className="eyebrow">The product</p>
                <h2 className="h2">Two windows, one plan</h2>
                <p className="lede mt-5">
                  The workspace walks the clinical order one step at a time, and the robot cell
                  opens beside it. Both are the same local application reading the same artefacts.
                </p>
              </div>
            </Reveal>

            <Reveal delay={100}>
              <ProductSwitcher />
            </Reveal>
          </div>
        </section>

        {/* ======================= Capabilities ======================= */}
        <section className="section section-soft">
          <div className="shell">
            <Reveal>
              <div className="mb-12 max-w-2xl">
                <p className="eyebrow">Capabilities</p>
                <h2 className="h2">What you can do with MORP</h2>
                <p className="lede mt-5">
                  Six things, each of which writes an artefact you can open rather than a number
                  you have to trust.
                </p>
              </div>
            </Reveal>

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {FEATURES.map((f, i) => (
                <Reveal key={f.title} delay={i * 60}>
                  <article className="card card-hover h-full p-6">
                    <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand-50 text-brand-600">
                      <Icon name={f.icon as IconName} />
                    </span>
                    <h3 className="h3 mt-5">{f.title}</h3>
                    <p className="body mt-2.5">{f.body}</p>
                  </article>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* ======================= How it works ======================= */}
        <section id="how" className="section bg-white">
          <div className="shell">
            <Reveal>
              <div className="mb-12 max-w-2xl">
                <p className="eyebrow">Technology</p>
                <h2 className="h2">Six stages, each one inspectable</h2>
                <p className="lede mt-5">
                  There is no single opaque step between the scan and the plan. Each stage reads
                  the artefact the previous one wrote and writes its own, so any claim on screen
                  traces back to a file on disk — and a stage can be re-run on its own.
                </p>
              </div>
            </Reveal>

            <ol className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {PIPELINE.map((s, i) => (
                <Reveal key={s.n} delay={i * 55}>
                  <li className="card h-full p-6">
                    <div className="flex items-center justify-between">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-900 font-mono text-[11px] font-bold text-white">
                        {s.n}
                      </span>
                      <Icon name="arrow" className="h-4 w-4 text-ink-300" />
                    </div>
                    <h3 className="h3 mt-5">{s.title}</h3>
                    <p className="body mt-2.5">{s.body}</p>
                    <p className="mt-4 border-t border-ink-200/70 pt-3 font-mono text-[10.5px] leading-relaxed text-brand-600">
                      {s.artefact}
                    </p>
                  </li>
                </Reveal>
              ))}
            </ol>

            {/* The brief, answered row by row. */}
            <Reveal delay={80}>
              <div className="mt-20 max-w-2xl">
                <p className="eyebrow">Against the brief</p>
                <h2 className="h2">Nine requirements, plainly scored</h2>
                <p className="lede mt-5">
                  Seven delivered, one partial, two blocked. The blocked pair are the same missing
                  thing: the open dataset we can legally use carries no meniscus labels, so the
                  model has never been shown one.
                </p>
              </div>
            </Reveal>

            <Reveal delay={120}>
              <ul className="mt-10 divide-y divide-ink-200/70 overflow-hidden rounded-card border border-ink-200/70 bg-white shadow-card">
                {REQUIREMENTS.map((r) => {
                  const status = STATUS[r.status];
                  return (
                    <li
                      key={r.ask}
                      className="grid gap-2 px-5 py-4 md:grid-cols-[0.9fr_1.7fr_auto] md:items-baseline md:gap-6 md:px-7"
                    >
                      <span className="text-[13.5px] font-semibold text-ink-900">{r.ask}</span>
                      <span className="body">{r.answer}</span>
                      <span
                        className={`justify-self-start rounded-full border px-2.5 py-0.5 font-mono text-[9.5px] font-semibold uppercase tracking-[0.12em] md:justify-self-end ${status.className}`}
                      >
                        {status.label}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </Reveal>
          </div>
        </section>

        {/* ======================= Screens ======================= */}
        <section id="screens" className="section section-soft">
          <div className="shell">
            <Reveal>
              <div className="mb-12 flex flex-wrap items-end justify-between gap-6">
                <div className="max-w-2xl">
                  <p className="eyebrow">Screens</p>
                  <h2 className="h2">Captures from the running workspace</h2>
                  <p className="lede mt-5">
                    Real screens against a real case from the OAIZIB-CM dataset — not mockups. The
                    anatomy holds the canvas and the panels float over it, changing with the step.
                  </p>
                </div>
                <a href="#run" className="btn-outline shrink-0">
                  Run it yourself
                </a>
              </div>
            </Reveal>

            <Reveal delay={100}>
              <Screens />
            </Reveal>
          </div>
        </section>

        {/* ======================= Figures band ======================= */}
        <section className="bg-ink-900 py-16">
          <div className="shell">
            <Reveal>
              <div className="mb-10 max-w-2xl">
                <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-[0.2em] text-brand-300">
                  By the numbers
                </p>
                <h2 className="text-[26px] font-semibold leading-tight tracking-[-0.015em] text-white md:text-[34px]">
                  Measured, not asserted
                </h2>
                <p className="mt-4 max-w-reading text-[14px] leading-relaxed text-white/55">
                  Every figure below and in the evidence section comes from an artefact in the
                  repository. The weak ones are shown too, because a planning tool that overstates
                  itself is worse than no planning tool.
                </p>
              </div>
            </Reveal>

            <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
              {IMPACT.map((s, i) => (
                <Reveal key={s.label} delay={i * 70}>
                  <div className="border-t border-white/15 pt-5">
                    <span className="text-brand-300">
                      <Icon name={s.icon as IconName} className="h-5 w-5" />
                    </span>
                    <p className="mt-4 font-mono text-[34px] leading-none text-white">{s.value}</p>
                    <p className="mt-3 text-[12.5px] leading-snug text-white/50">{s.label}</p>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* ======================= Pull quote ======================= */}
        <section className="bg-white py-16 md:py-20">
          <div className="shell">
            <Reveal>
              <figure className="mx-auto max-w-3xl text-center">
                <span className="mx-auto mb-6 flex h-11 w-11 items-center justify-center rounded-full bg-brand-50 text-brand-600">
                  <Icon name="check" className="h-5 w-5" />
                </span>
                <blockquote className="text-[19px] font-medium leading-[1.5] tracking-[-0.01em] text-ink-900 md:text-[24px]">
                  “{PULL_QUOTE.text}”
                </blockquote>
                <figcaption className="mt-6 text-[12.5px] text-ink-500">
                  <span className="font-semibold text-ink-800">{PULL_QUOTE.attribution}</span>
                  <span className="mx-2 text-ink-300">·</span>
                  <span className="font-mono text-[11.5px]">{PULL_QUOTE.source}</span>
                </figcaption>
                <p className="mt-5 text-[11.5px] leading-relaxed text-ink-400">
                  Not a clinician testimonial. No surgeon has endorsed this prototype, and this is
                  where a commercial site would put one.
                </p>
              </figure>
            </Reveal>
          </div>
        </section>

        {/* ======================= Evidence ======================= */}
        <section id="evidence" className="section section-soft">
          <div className="shell">
            <Reveal>
              <div className="mb-14 max-w-2xl">
                <p className="eyebrow">Evidence</p>
                <h2 className="h2">Validation results in full</h2>
                <p className="lede mt-5">
                  Bone segmentation is usable. Thin cartilage is not, and femoral sizing error is
                  far too large to trust — both of which the interface reports per structure rather
                  than averaging away.
                </p>
              </div>
            </Reveal>

            <div className="grid gap-6 lg:grid-cols-2">
              {/* --- Dice --- */}
              <Reveal>
                <div className="card h-full p-7">
                  <h3 className="h3">Segmentation quality</h3>
                  <p className="mt-1 font-mono text-[11px] text-ink-400">
                    Dice, 102 held-out cases
                  </p>

                  <ul className="mt-6 space-y-4">
                    {DICE.map((d) => (
                      <li key={d.structure}>
                        <div className="mb-1.5 flex items-baseline justify-between gap-3">
                          <span className="text-[13px] text-ink-700">{d.structure}</span>
                          <span
                            className={`font-mono text-[13px] tabular-nums ${
                              d.verdict === "weak" ? "text-amber-600" : "text-ink-900"
                            }`}
                          >
                            {d.mean.toFixed(3)}
                            <span className="text-ink-300"> ±{d.sd.toFixed(3)}</span>
                          </span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-200/60">
                          <div
                            className={`h-full rounded-full ${
                              d.verdict === "weak" ? "bg-amber-400" : "bg-brand-500"
                            }`}
                            style={{ width: `${d.mean * 100}%` }}
                          />
                        </div>
                      </li>
                    ))}
                  </ul>

                  <p className="mt-6 border-t border-ink-200/70 pt-4 text-[11.5px] leading-relaxed text-ink-400">
                    Mean Dice excluding background is 0.558. Quoting that alone would hide both the
                    0.79 on the femur and the 0.16 on the medial tibial cartilage, which are two
                    completely different clinical situations.
                  </p>
                </div>
              </Reveal>

              {/* --- MAE --- */}
              <Reveal delay={70}>
                <div className="card h-full p-7">
                  <h3 className="h3">Measurement error</h3>
                  <p className="mt-1 font-mono text-[11px] text-ink-400">
                    Mean absolute error against ground-truth labels
                  </p>

                  <ul className="mt-6 divide-y divide-ink-200/70">
                    {MAE.map((m) => (
                      <li key={m.name} className="flex items-baseline justify-between gap-4 py-2.5">
                        <span className="text-[13px] text-ink-600">{m.name}</span>
                        <span
                          className={`font-mono text-[13px] tabular-nums ${
                            m.mae > 5 ? "text-amber-600" : "text-ink-900"
                          }`}
                        >
                          {m.mae.toFixed(2)} mm
                        </span>
                      </li>
                    ))}
                  </ul>

                  <p className="mt-6 border-t border-ink-200/70 pt-4 text-[11.5px] leading-relaxed text-ink-400">
                    Tibial width at 3.2 mm is within the range where a size suggestion is worth
                    reading. Femoral depth at 17.8 mm is not, which is why candidates are presented
                    as ranked options with their millimetres shown rather than as an answer.
                  </p>
                </div>
              </Reveal>

              {/* --- Dataset --- */}
              <Reveal>
                <div className="card h-full p-7">
                  <h3 className="h3">The data</h3>
                  <p className="mt-1 font-mono text-[11px] text-ink-400">
                    {DATASET.name} · {DATASET.host}
                  </p>
                  <ul className="mt-6 divide-y divide-ink-200/70">
                    {DATASET.facts.map((f) => (
                      <li key={f.k} className="flex items-baseline justify-between gap-4 py-2.5">
                        <span className="text-[13px] text-ink-600">{f.k}</span>
                        <span className="text-right font-mono text-[12px] text-ink-900">{f.v}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-6 flex gap-2.5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-[11.5px] leading-relaxed text-amber-800">
                    <span aria-hidden className="mt-[6px] h-1 w-1 shrink-0 rounded-full bg-amber-500" />
                    <span>
                      {DATASET.licence}. A model trained on it cannot ship in a commercial product,
                      and the model card records that rather than leaving it to be discovered later.
                    </span>
                  </p>
                </div>
              </Reveal>

              {/* --- Model --- */}
              <Reveal delay={70}>
                <div className="card h-full p-7">
                  <h3 className="h3">The model</h3>
                  <p className="mt-1 font-mono text-[11px] text-ink-400">
                    fine-tuned on a deliberately small budget
                  </p>
                  <ul className="mt-6 divide-y divide-ink-200/70">
                    {TRAINING.map((f) => (
                      <li key={f.k} className="flex items-baseline justify-between gap-4 py-2.5">
                        <span className="text-[13px] text-ink-600">{f.k}</span>
                        <span className="text-right font-mono text-[12px] text-ink-900">{f.v}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-6 border-t border-ink-200/70 pt-4 text-[11.5px] leading-relaxed text-ink-400">
                    Twenty epochs is a budget, not an accident. It is why cartilage Dice is where it
                    is, and the platform is built so a better model drops in behind the same
                    contract without the geometry, measurement or planning code changing.
                  </p>
                </div>
              </Reveal>
            </div>
          </div>
        </section>

        {/* ======================= Limits ======================= */}
        <section id="limits" className="section bg-white">
          <div className="shell">
            <div className="grid gap-12 lg:grid-cols-[0.8fr_1.2fr]">
              <Reveal>
                <p className="eyebrow !text-amber-600">What this is not</p>
                <h2 className="h2">The honest part</h2>
                <p className="lede mt-5">
                  This is the section a reviewer reads first and a brochure usually leaves out.
                  None of it is hedging: each line is a measured limit or a stated architectural
                  boundary.
                </p>
              </Reveal>

              <Reveal delay={90}>
                <ul className="space-y-3">
                  {LIMITS.map((line) => (
                    <li
                      key={line}
                      className="flex gap-3.5 rounded-xl border border-ink-200/70 bg-paper-soft px-5 py-4"
                    >
                      <span
                        aria-hidden
                        className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500"
                      />
                      <span className="body">{line}</span>
                    </li>
                  ))}
                </ul>
              </Reveal>
            </div>
          </div>
        </section>

        {/* ======================= Principles ======================= */}
        <section id="principles" className="section section-soft">
          <div className="shell">
            <Reveal>
              <div className="mb-12 max-w-2xl">
                <p className="eyebrow">Why it is built this way</p>
                <h2 className="h2">An assistant that shows its work</h2>
              </div>
            </Reveal>

            <div className="grid gap-x-10 gap-y-9 md:grid-cols-2 lg:grid-cols-3">
              {PRINCIPLES.map((p, i) => (
                <Reveal key={p.title} delay={i * 50}>
                  <div className="border-t-2 border-brand-200 pt-4">
                    <h3 className="h3 text-[14.5px]">{p.title}</h3>
                    <p className="body mt-2">{p.body}</p>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* ======================= FAQ ======================= */}
        <section id="faq" className="section bg-white">
          <div className="shell">
            <div className="grid gap-10 lg:grid-cols-[0.72fr_1.28fr] lg:gap-14">
              <Reveal>
                <p className="eyebrow">FAQ</p>
                <h2 className="h2">Frequently asked</h2>
                <p className="lede mt-5">
                  What it is, what it needs, and what it cannot do. If something is missing here it
                  is probably in the limits above.
                </p>
              </Reveal>

              <Reveal delay={90}>
                <Faq />
              </Reveal>
            </div>
          </div>
        </section>

        {/* ======================= Run it ======================= */}
        <section id="run" className="section section-soft">
          <div className="shell">
            <div className="grid gap-12 lg:grid-cols-[1.05fr_0.95fr]">
              <Reveal>
                <p className="eyebrow">Get started</p>
                <h2 className="h2">Three commands and a browser</h2>
                <p className="lede mt-5">
                  A FastAPI service over the artefact directory, and a Next.js workspace on top of
                  it. Nothing leaves the machine, and the API binds to loopback because it has no
                  authentication.
                </p>

                <div className="mt-8 overflow-hidden rounded-card border border-ink-200/70 bg-ink-900 shadow-card">
                  <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2.5">
                    <span aria-hidden className="h-2 w-2 rounded-full bg-accent-400/70" />
                    <span className="font-mono text-[10.5px] text-white/40">terminal</span>
                  </div>
                  <pre className="overflow-x-auto px-5 py-5 font-mono text-[11.5px] leading-[1.95] text-white/80">
                    <code>{`make install          # core + imaging + api extras
make api              # FastAPI on 127.0.0.1:8000
make web              # workspace on localhost:3000

make infer CASE=oaizib-405   # segment, measure and plan one case`}</code>
                  </pre>
                </div>

                <p className="mt-4 text-[11.5px] leading-relaxed text-ink-500">
                  On Windows, where <span className="code">make</span> is not available, the same
                  targets run through <span className="code">.\morp.ps1 &lt;target&gt;</span> in the
                  repository root.
                </p>

                <p className="mt-3 text-[11.5px] leading-relaxed text-ink-400">
                  This site is separate from all of that: a static export in{" "}
                  <span className="code">site/</span> with no backend, so it can be published
                  anywhere without exposing the API or any imaging data.
                </p>
              </Reveal>

              <Reveal delay={90}>
                <p className="eyebrow">Built with</p>
                <div className="space-y-4">
                  {STACK.map((group) => (
                    <div key={group.group} className="card p-5">
                      <h3 className="h3 text-[13.5px]">{group.group}</h3>
                      <ul className="mt-3 flex flex-wrap gap-1.5">
                        {group.items.map((item) => (
                          <li
                            key={item}
                            className="rounded-full border border-ink-200 bg-paper-soft px-2.5 py-1 text-[11.5px] text-ink-600"
                          >
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </Reveal>
            </div>
          </div>
        </section>

        {/* ======================= Footer ======================= */}
        <footer className="bg-ink-900 pt-16 text-white/70">
          <div className="shell">
            <div className="grid gap-10 pb-12 md:grid-cols-2 lg:grid-cols-[1.4fr_repeat(3,0.86fr)]">
              <div>
                <div className="flex items-center gap-2.5">
                  <span
                    aria-hidden
                    className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-500"
                  >
                    <span className="h-2.5 w-2.5 rounded-full bg-white" />
                  </span>
                  <span className="text-[17px] font-bold tracking-[0.14em] text-white">MORP</span>
                </div>
                <p className="mt-4 max-w-xs text-[12.5px] leading-relaxed text-white/45">
                  {PROJECT.expansion}. Explainable patient-specific knee planning from a single MR
                  scan.
                </p>
                <p className="mt-5 font-mono text-[11px] leading-relaxed text-white/35">
                  {PROJECT.philosophy}
                </p>
              </div>

              {FOOTER_LINKS.map((col) => (
                <div key={col.group}>
                  <h2 className="mb-4 font-mono text-[10.5px] font-semibold uppercase tracking-[0.16em] text-white/40">
                    {col.group}
                  </h2>
                  <ul className="space-y-2.5">
                    {col.links.map((l) => (
                      <li key={l.label}>
                        <a
                          href={l.href}
                          className="text-[12.5px] text-white/65 transition-colors hover:text-white"
                        >
                          {l.label}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            {/* Where a commercial site lists its certifications. Ours says the opposite,
                in the same position, on purpose. */}
            <div className="grid gap-4 border-t border-white/10 py-8 md:grid-cols-3">
              {[
                {
                  k: "Regulatory status",
                  v: "None. Not a medical device, not registered, not clinically validated.",
                },
                {
                  k: "Intended use",
                  v: "Research, method development and internal evaluation only.",
                },
                {
                  k: "Data licence",
                  v: "OAIZIB-CM under CC-BY-NC-4.0. Non-commercial use only.",
                },
              ].map((item) => (
                <div key={item.k} className="rounded-xl border border-white/10 px-4 py-3.5">
                  <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-brand-300">
                    {item.k}
                  </p>
                  <p className="mt-1.5 text-[12px] leading-relaxed text-white/55">{item.v}</p>
                </div>
              ))}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 border-t border-white/10 py-6">
              <p className="font-mono text-[10.5px] tracking-[0.18em] text-white/30">
                MORP · MORPHOMETRY · RECONSTRUCTION · PLANNING
              </p>
              <a href="#top" className="text-[12px] font-semibold text-white/60 hover:text-white">
                Back to top ↑
              </a>
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
