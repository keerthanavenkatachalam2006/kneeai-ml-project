# ML → Application Contract (v1)

**Status:** authored by the ML engineer, for review by the application engineer.
**Neither side may change this unilaterally.** Changes go through the process in
[§9](#9-changing-this-contract).

This document defines the only interface between the machine-learning pipeline and
the rest of the platform. If it is not described here, the application must not
depend on it.

---

## 1. What this contract covers

The ML pipeline produces **per-case inference artifacts**. Everything downstream —
3D reconstruction, landmarking, measurement, implant matching, planning,
simulation, reporting, the web UI — consumes those artifacts and nothing else.

In particular, the application does **not** import `torch`, `monai`, or any model
code. It reads files. This keeps the deployed application installable without a
CUDA toolchain, which matters because training runs on Google Colab (PRD §65) while
the application runs on a hospital workstation.

```
                    ML side                    │              Application side
  ────────────────────────────────────────────┼───────────────────────────────────
  scripts/infer_case.py                       │  reads artifacts/cases/<case_id>/
    ├── segmentation.nii.gz  ────────────────►│    ├── segmentation.nii.gz
    └── result.json          ────────────────►│    └── result.json
                                              │
  artifacts/models/seg_oaizib/                │  reads model_card.json for
    ├── checkpoint.pt                         │  provenance display only
    └── model_card.json      ────────────────►│
                                              │  writes meshes/*.glb
```

---

## 2. Artifact layout

Per case, under the artifact root (`KNEEAI_ARTIFACT_ROOT`, default `./artifacts`):

```
artifacts/cases/<case_id>/
├── segmentation.nii.gz    written by ML       int16 label volume
├── result.json            written by ML       manifest (§3)
└── meshes/*.glb           written by App      Draco-compressed, millimetre units
```

`<case_id>` is filesystem-safe. `kneeai.config.Settings.case_dir()` already rejects
identifiers containing path separators, `.` or `..`; the ML side uses that helper
rather than joining paths itself.

### 2.1 `segmentation.nii.gz`

| Property   | Requirement                                                          |
|------------|----------------------------------------------------------------------|
| Format     | NIfTI-1, gzip-compressed                                             |
| Data type  | `int16`                                                              |
| Values     | Integers appearing in `segmentation.label_map`, plus `0` = background |
| Affine     | Correct, and byte-for-byte consistent with `geometry.affine`          |
| Spacing    | Millimetres, consistent with `geometry.spacing_mm`                   |
| Grid       | The **source image grid**, not the model's internal working grid      |

The last row matters. The model runs on a preprocessed volume (cropped, resampled,
intensity-normalised). The mask it emits is mapped back onto the original
acquisition grid with `kneeai.imaging.preprocess.resample_like` before being
written, so every downstream measurement is made in the geometry the image was
acquired in. Shipping the mask on the working grid would silently change every
measurement by the resampling factor.

A JSON sidecar (`segmentation.json`) is also written by
`kneeai.imaging.io.write_volume`. It is **not** part of this contract — it is an
implementation detail of the imaging layer. Consume `result.json`.

---

## 3. `result.json` schema

```jsonc
{
  "contract_version": "1",                     // string, required
  "case_id": "str",                            // required
  "source": "ai_prediction" | "ground_truth" | "phantom" | "mock",   // required

  "model": {
    "model_id":     "str|null",
    "version":      "str|null",
    "architecture": "str|null",
    "model_card":   "path|null"                // relative to artifact root
  },

  "segmentation": {
    "path": "segmentation.nii.gz",             // relative to this file
    "label_map": {                             // required, non-empty
      "femur": 1,
      "femoral_cartilage": 2,
      "tibia": 3,
      "medial_tibial_cartilage": 4,
      "lateral_tibial_cartilage": 5
    },
    "structures_present": ["femur", "..."],                       // required
    "structures_expected_but_missing": ["medial_meniscus", "lateral_meniscus"]
  },

  "geometry": {
    "spacing_mm":  [x, y, z],                  // required, all > 0
    "orientation": "LPS",                      // required, 3 letters from RLAPIS
    "affine":      [[4x4]],                    // required, voxel index -> mm
    "image_shape": [i, j, k],                  // required
    "side": "left" | "right" | "unknown"       // required
  },

  "confidence": {
    "per_structure": { "femur": 0.97 },         // 0..1, or null where undefined
    "method": "str describing how it was computed",
    "volume_mm3": { "femur": 156354.0 }
  },

  "provenance": { /* kneeai.core.provenance.Provenance.to_dict() */ },

  "warnings": ["str"]
}
```

### 3.1 Field reference

**`source`** — mandatory. One of four values, and the application must surface it in
every view and every export. See [rule 2](#rule-2-source-is-mandatory-and-must-be-visible).

**`model`** — all fields may be `null` when `source` is not `ai_prediction` (a
phantom has no model). When `source == "ai_prediction"`, `model_id` and `version`
must both be non-null; the validator enforces this.

**`segmentation.label_map`** — structure name → integer label. Names are
`kneeai.core.anatomy.Structure` values. The map must be a subset of
`kneeai.core.anatomy.DEFAULT_LABEL_MAP` **with identical integer assignments**.
Model 1's five labels are exactly labels 1–5 of that map:

| Label | Structure                  |
|------:|----------------------------|
|     0 | background                 |
|     1 | femur                      |
|     2 | femoral_cartilage          |
|     3 | tibia                      |
|     4 | medial_tibial_cartilage    |
|     5 | lateral_tibial_cartilage   |

**`structures_present`** — structures with at least one voxel in the written mask.
Derived from the mask, not from what the model was asked to predict. A structure in
`label_map` but absent here was genuinely not found.

**`structures_expected_but_missing`** — structures the clinical pipeline wants but
this model cannot produce. For Model 1 this always contains both menisci, because
OAIZIB-CM has no meniscus labels (see `docs/DATASET_ACQUISITION.md`). This field
exists so the application can distinguish *"the model looked and found nothing"*
from *"no model has ever looked"* — which are very different messages to a surgeon.

**`geometry.affine`** — the 4×4 mapping voxel index → millimetres in the IMAGE
frame, i.e. `Volume.voxel_to_image.matrix.tolist()`. Row-major, 4 rows of 4.

**`confidence.per_structure`** — mean predicted softmax probability over the voxels
assigned to each structure. `null` for a structure with no voxels: there is no
meaningful confidence for something that was not found, and `0.0` would be read as
"found, with low confidence". `confidence.method` states in words how the number
was computed, because "confidence" means nothing on its own.

**`confidence.volume_mm3`** — voxel count × voxel volume, per structure. Provided
here so the application does not have to load the mask to show a volume.

**`provenance`** — verbatim `Provenance.to_dict()`. Keys: `provenance_id`,
`component`, `component_version`, `origin`, `model_id`, `model_version`, `inputs`,
`parameters`, `created_at`, `platform_version`, `schema_version`, `environment`,
`notes`, `signature`.

**`warnings`** — human-readable strings. Populated from
`kneeai.imaging.validation` findings and from post-processing actions that removed
a material fraction of a structure. Empty list, never `null`.

---

## 4. Hard rules

### Rule 1: no accuracy metrics in `result.json`

Dice, IoU, HD95 and ASSD all require ground truth. **Ground truth does not exist at
inference time on a patient scan.** A `"dice"` field in `result.json` could only
ever be one of three things: a leak of ground truth into the prediction path, a
number copied from a different cohort and misrepresented as this patient's, or a
fabrication.

Reference metrics live **only** in the model card, via
`kneeai.ml.registry.ModelMetrics`, which separates `internal_validation`,
`internal_test` and `external_test` precisely so a blended "accuracy" cannot hide
the generalisation gap.

The validator rejects any of `dice`, `iou`, `hd95`, `assd`, `hausdorff`,
`accuracy`, `sensitivity`, `specificity`, `precision`, `recall`, `f1` appearing
anywhere in `result.json`.

### Rule 2: `source` is mandatory and must be visible

Ground-truth or phantom output must **never** be presented as an AI result. This is
not a cosmetic concern: a demo that shows hand-drawn labels as model output
misrepresents the model's capability to whoever is evaluating it.

- `ai_prediction` — a trained model ran on this image.
- `ground_truth` — expert labels from a dataset. Model performance ceiling, not model output.
- `phantom` — synthetic geometry from `kneeai.phantom`. Pipeline fixture, no patient.
- `mock` — placeholder for UI development. Must be visually unmistakable in the UI.

### Rule 3: `side` is mandatory

`core/anatomy.py` documents getting this wrong as a wrong-site-surgery class error,
and every medial/lateral resolution in the platform depends on it —
`PatientFrame.lateral` raises rather than guessing when the side is unknown.

`"unknown"` is a permitted value, but it is a **blocking** condition downstream, not
a default to be quietly tolerated. When the side cannot be established from the
data it must appear in `warnings` as well.

### Rule 4: `label_map` is mandatory

`ml/registry.py` already refuses a checkpoint whose label map differs from its model
card. The application performs the same check against
`core.anatomy.DEFAULT_LABEL_MAP`. A mask whose label ordering differs from what the
consumer assumes produces confident, anatomically scrambled output and no error.

### Rule 5: `affine` is mandatory — voxel indices are not millimetres

`core/frames.py` is built on frame-tagged transforms specifically so this cannot be
fudged. A bare orientation string loses the spacing, origin and direction cosines
that `Volume.voxel_to_image` needs, and an anisotropic volume interpreted as
isotropic is wrong by the anisotropy ratio in every measurement.

### Rule 6: reuse existing types

Serialise `kneeai.imaging.Volume` and `kneeai.core.provenance.Provenance`. Do not
invent parallel schemas. Two representations of the same fact drift apart.

---

## 5. Validation and the canonical implementation

### 5.1 `kneeai.ml.contract` is canonical

**`src/kneeai/ml/contract.py` is the single source of truth** for this schema: the
`ArtifactSource` enum, validation, and payload construction. `kneeai.api.contract`
**re-exports from it** and must not redefine any of them.

Three reasons the canonical copy lives on the ML side:

1. **The producer defines the format.** The ML pipeline writes these files; a schema
   owned by the consumer would drift from what is actually emitted.
2. **It is torch-free.** `kneeai.ml.contract` imports only `kneeai.core.anatomy`,
   `kneeai.core.errors` and the standard library. It imports cleanly on a machine
   with no `torch`, no `monai` and no CUDA toolchain — verified by importing it in an
   environment where torch is not installed at all. The application can therefore
   depend on it without pulling in the training stack.
3. **This document lives beside it.** Schema and prose change together.

Two independent definitions of one schema is the failure this resolves. If both
sides define `SegmentationSource`/`ArtifactSource` separately, they agree until the
day one is edited, and then they disagree silently — the consumer validates against
its own copy and accepts a manifest the producer never intended to emit.

Name mapping for the re-export:

| Canonical (`kneeai.ml.contract`) | Application alias           |
|----------------------------------|-----------------------------|
| `ArtifactSource`                 | `SegmentationSource`        |
| `validate_result_payload`        | —                           |
| `validate_result_json`           | —                           |
| `build_result_payload`           | —                           |
| `CONTRACT_VERSION`               | `CONTRACT_VERSION`          |

Typed accessor dataclasses on the application side (`CaseResult`, `ModelRef`,
`SegmentationRef`, `GeometryRef`, `ConfidenceBlock`) are fine and useful — they are a
*typed view* over a validated payload. They must parse a payload that
`validate_result_payload` has already accepted, and must not implement a second,
divergent validation.

### 5.2 UI badge strings

`ArtifactSource.badge` returns the exact uppercase chip text the viewer renders:

| `source`          | `badge`          |
|-------------------|------------------|
| `ai_prediction`   | `AI PREDICTION`  |
| `ground_truth`    | `GROUND TRUTH`   |
| `phantom`         | `PHANTOM`        |
| `mock`            | `MOCK`           |

These strings are part of the contract and are rendered verbatim. `display_label`
is a separate, longer form for report headers and tooltips, where there is room to
carry the caveat that ground truth and phantom output are not model output.

### 5.3 Calling the validator

```python
from kneeai.ml.contract import validate_result_json, CONTRACT_VERSION

report = validate_result_json("artifacts/cases/<case_id>/result.json")
report.raise_if_failed()          # or inspect report.errors / report.warnings
```

`scripts/infer_case.py` validates its own output before exiting, so a malformed
manifest never reaches the application.

Both sides should call it: the producer to avoid shipping a bad manifest, the
consumer to avoid trusting one.

---

## 6. Laterality: left-knee inputs must be mirrored

**Model 1's training data contains 507 right knees and zero left knees.** This is
verified, not assumed — see `docs/DATASET_ACQUISITION.md` §1.6. Without handling,
the model silently fails on roughly half of all patients.

### 6.1 The required inference path

A left-knee input is **canonicalised to right** before inference and the mask is
mirrored back afterwards:

```
left-knee volume
   → mirror across the sagittal plane        (image only)
   → predict                                  (model sees a right knee)
   → mirror the predicted mask back           (mask returns to left-knee geometry)
   → write segmentation.nii.gz on the ORIGINAL grid and affine
```

`scripts/infer_case.py` does this automatically from `geometry.side`. Two
consequences the application must know:

1. The mask in `segmentation.nii.gz` is always in the **original** patient geometry.
   The mirroring is internal and fully reversed. Downstream code needs no
   laterality-specific handling.
2. When mirroring was applied, `warnings` contains an explicit entry, and
   `provenance.parameters` records `mirrored_for_inference: true`. A left-knee result
   is therefore always distinguishable from a right-knee result in the audit trail.

Mirroring is label-safe: medial and lateral are anatomical designations that travel
with the tissue, so reflecting image and mask together moves
`medial_tibial_cartilage` to the side where a left knee's medial compartment belongs
while preserving its label value. This is verified numerically, not asserted.

### 6.2 Why this is still a limitation, not a fix

Training also applies random sagittal mirroring (`mirror_lr_prob=0.5`), so the model
does see mirrored-left samples. **A mirrored right knee is not a native left knee.**
Real left knees carry their own anatomical variation, and the model has never seen
any. So:

- Model 1 performance on left knees is **unmeasured**, not merely lower.
- The model card records this as both a limitation and a contraindication.
- Any left-knee result carries the warning above, and a reviewer should treat its
  confidence figures as less trustworthy than a right-knee result's.

The honest resolution is left-knee data or a left-knee evaluation set. Until then,
mirroring is a mitigation, and the contract makes it visible rather than invisible.

---

## 7. Model registry path contract

```
artifacts/models/seg_oaizib/
├── checkpoint.pt
└── model_card.json
```

**This exact path is the contract.** `kneeai.config.Settings.model_registry`
already points at `artifacts/models`, and
`kneeai.ml.registry.ModelRegistry` resolves `seg_oaizib` from there.

`model_card.json` is a serialised `kneeai.ml.registry.ModelCard`. It cannot be
constructed without a `DatasetContract` (PRD §113 Rule 1), so it always records the
dataset ids, ground-truth description, annotator, licence, patient count and split
strategy behind the model.

The card carries `approved_for_clinical_use`, which defaults to `False`. A newly
trained checkpoint is a research artefact until somebody decides otherwise, and
`ModelRegistry.approve_for_clinical_use()` requires a named approver and a written
justification. **The application must surface this flag.**

---

## 8. What the ML side does *not* provide

Stated explicitly so the application does not wait for it:

| Not provided | Why |
|---|---|
| Meniscus labels | OAIZIB-CM has none. SKM-TEA is blocked (900 GB, gated, indivisible download). |
| Patella / patellar cartilage | Not labelled in OAIZIB-CM. Labels 6–7 are reserved but unused. |
| Landmarks | Derived deterministically by `kneeai.landmarks`, not predicted. |
| Measurements | Derived by `kneeai.measure` from the mask and landmarks. |
| Meshes | The application generates `meshes/*.glb` from `segmentation.nii.gz`. |
| OA / KL grading | No classification model in scope. |

---

## 9. Changing this contract

1. `contract_version` is bumped.
2. Both engineers agree in writing before either side ships the change.
3. The validator is updated in the same change as the schema.
4. Additive optional fields are a minor change; renaming, removing or retyping a
   field is breaking and requires a version bump on both sides.

---

## Appendix A — worked example

Model 1 inference on a real OAIZIB-CM case (values illustrative of shape, not
measured results):

```json
{
  "contract_version": "1",
  "case_id": "oaizib-9001104",
  "source": "ai_prediction",
  "model": {
    "model_id": "seg_oaizib",
    "version": "0.1.0",
    "architecture": "swin_unetr",
    "model_card": "models/seg_oaizib/model_card.json"
  },
  "segmentation": {
    "path": "segmentation.nii.gz",
    "label_map": {
      "femur": 1, "femoral_cartilage": 2, "tibia": 3,
      "medial_tibial_cartilage": 4, "lateral_tibial_cartilage": 5
    },
    "structures_present": [
      "femur", "femoral_cartilage", "tibia",
      "medial_tibial_cartilage", "lateral_tibial_cartilage"
    ],
    "structures_expected_but_missing": ["medial_meniscus", "lateral_meniscus"]
  },
  "geometry": {
    "spacing_mm": [0.365, 0.365, 0.7],
    "orientation": "LPS",
    "affine": [[0.365,0,0,-70.1],[0,0.365,0,-52.4],[0,0,0.7,-56.0],[0,0,0,1]],
    "image_shape": [384, 384, 160],
    "side": "right"
  },
  "confidence": {
    "per_structure": {
      "femur": 0.98, "femoral_cartilage": 0.81, "tibia": 0.98,
      "medial_tibial_cartilage": 0.79, "lateral_tibial_cartilage": 0.80
    },
    "method": "mean predicted softmax probability over the voxels assigned to each structure, computed before post-processing",
    "volume_mm3": {
      "femur": 168420.0, "femoral_cartilage": 12980.0, "tibia": 142310.0,
      "medial_tibial_cartilage": 2410.0, "lateral_tibial_cartilage": 2690.0
    }
  },
  "provenance": { "component": "recon.surface", "origin": "ai", "...": "..." },
  "warnings": [
    "Model 1 does not segment the menisci; meniscus measurements are unavailable for this case."
  ]
}
```
