# MORP — showcase site

A single static page that explains the project. **This is not the product.** It makes no API
calls, ships no model and contains no imaging data; the only assets are screenshots of the
running workspace.

The product itself is `apps/web` (Next.js workspace) plus `src/kneeai` (FastAPI service).

## Run

```bash
npm install
npm run dev        # localhost:3001 — port 3000 is the product
```

On Windows, from the repository root, `.\morp.ps1 site-install` then `.\morp.ps1 site` do
the same thing. `make` is not available there and the Makefile is POSIX-only.

## Build

```bash
npm run build      # static export to out/
```

`out/` is plain HTML and assets. Drop it on any static host — there is no server to run and
no API to expose. For a project subpath (GitHub Pages, for instance):

```bash
NEXT_PUBLIC_BASE_PATH=/morp npm run build
```

## Screenshots

`public/shots/*.png` are real captures of the workspace, not mockups. To retake them after an
interface change, start the API and the workspace, then:

```bash
npx playwright install chromium   # once
npm run shots
```

The script picks the first AI-segmented OAIZIB case that has meshes, a plan and a positive OA
indicator, because the implant and simulation steps are deliberately withheld on a knee that
does not need a replacement. Override with `MORP_CASE=oaizib_409 npm run shots`.

If a capture is missing, the gallery frame degrades to a labelled note rather than a broken
image, so the page still builds and reads correctly in a fresh clone.

## Content

Every figure on the page lives in `content/project.ts`, and each block names the artefact it
was copied from (`results/validation_report.json`, `docs/DATASET_ACQUISITION.md`,
`models/segresnet_knee/training_log.json`). Change a number there and nowhere else. The weak
results are on the page on purpose: the project's argument is that a measurement without
provenance is not worth much, and a page that quoted only the good numbers would undercut it.
