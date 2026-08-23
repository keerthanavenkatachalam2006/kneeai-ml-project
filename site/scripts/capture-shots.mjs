/**
 * Re-capture the screenshots this site shows, from the running product.
 *
 * The site claims its images are the real application, so they have to be re-taken when the
 * interface changes. This drives a headless Chromium through the workspace one workflow step
 * at a time and writes `site/public/shots/*.png`.
 *
 * Prerequisites, in the repository root:
 *   1. the API:       make api        (127.0.0.1:8000)
 *   2. the workspace: make web        (localhost:3000)
 *   3. the browser:   npx playwright install chromium
 *
 * Then, from site/:  npm run shots
 *
 * The case is chosen automatically: the first AI-segmented OAIZIB case that has meshes, a
 * plan, and a positive OA indicator. The implant and simulation steps are deliberately
 * withheld on a knee that does not need a replacement, and a capture of a blocked step is
 * not a capture of the feature. Override with MORP_CASE=oaizib_409.
 */

import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, "..", "public", "shots");

const WEB = process.env.MORP_WEB ?? "http://localhost:3000";
const API = process.env.MORP_API ?? "http://127.0.0.1:8000";

const VIEWPORT = { width: 1600, height: 900 };
const SCALE = 1.5;

/** Steps to capture, by the label rendered in the workspace stepper. */
const STEPS = [
  { file: "anatomy.png", step: "3D anatomy", settle: 4500 },
  { file: "slices.png", step: "2D slices", settle: 4000 },
  { file: "measurements.png", step: "Measurements", settle: 1200 },
  { file: "oa.png", step: "OA analysis", settle: 1200 },
  { file: "plan.png", step: "Implant & plan", settle: 5000 },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function stepButton(page, label) {
  return page.locator('nav[aria-label="Workflow"] button', { hasText: label }).first();
}

async function candidates() {
  if (process.env.MORP_CASE) return [process.env.MORP_CASE];

  const res = await fetch(`${API}/api/cases`);
  if (!res.ok) throw new Error(`api /api/cases returned ${res.status}`);
  const { cases } = await res.json();

  const usable = cases.filter((c) => c.has_plan && c.has_meshes);
  // Model output on a named dataset case first: the captions say so. Ad-hoc uploads last,
  // since their case ids are timestamps and mean nothing to a reader.
  const rank = (c) => (c.is_ai ? 0 : 2) + (/^oaizib_/.test(c.case_id) ? 0 : 1);
  return [...usable]
    .sort((a, b) => rank(a) - rank(b) || a.case_id.localeCompare(b.case_id))
    .map((c) => c.case_id);
}

async function main() {
  await mkdir(OUT, { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: SCALE,
    colorScheme: "dark",
  });
  const page = await context.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") console.warn("  browser:", m.text().slice(0, 160));
  });

  const ids = await candidates();
  if (ids.length === 0) throw new Error("no case has both meshes and a plan");

  await page.goto(WEB, { waitUntil: "networkidle" });
  await page.waitForSelector('select[aria-label="Select case"]');

  let chosen = null;
  for (const id of ids) {
    await page.selectOption('select[aria-label="Select case"]', id);
    await page.waitForTimeout(2500);
    if (await stepButton(page, "Implant & plan").isEnabled().catch(() => false)) {
      chosen = id;
      break;
    }
    console.log(`  ${id}: plan step withheld (no OA) — trying the next case`);
  }
  if (!chosen) {
    chosen = ids[0];
    console.warn(`! no OA-positive case found; falling back to ${chosen}`);
    await page.selectOption('select[aria-label="Select case"]', chosen);
    await page.waitForTimeout(2500);
  }
  console.log(`case: ${chosen}`);

  for (const shot of STEPS) {
    const button = stepButton(page, shot.step);
    if (!(await button.isEnabled().catch(() => false))) {
      console.warn(`! skipping ${shot.file}: "${shot.step}" is not available`);
      continue;
    }
    await button.click();
    await sleep(shot.settle);
    await page.screenshot({ path: path.join(OUT, shot.file) });
    console.log(`✓ ${shot.file}`);
  }

  // --- The robot cell, which is its own window in the product ---
  const robot = await context.newPage();
  await robot.goto(`${WEB}/robot?case=${encodeURIComponent(chosen)}`, { waitUntil: "networkidle" });
  await robot.waitForSelector("canvas", { timeout: 30000 });
  await sleep(3500);
  const play = robot.getByRole("button", { name: /play sweep/i });
  if (await play.isEnabled().catch(() => false)) {
    await play.click();
    // Mid-sweep: the tool is in the bone and part of the path is already travelled.
    await sleep(4200);
  }
  await robot.screenshot({ path: path.join(OUT, "robot.png") });
  console.log("✓ robot.png");

  await browser.close();
  console.log(`\nwrote captures to ${OUT}`);
  console.log("if the case changed, update the caption line in components/Screens.tsx");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
