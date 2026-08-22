# Dataset Acquisition

Companion to `data/registry/dataset_registry.csv`. Machine-readable facts live in the
CSV; this document explains how each dataset is obtained, what gates it, and whether
we use it.

**Verification convention.** Every claim below is tagged:

- **VERIFIED** — checked against the source on 2026-08-22 by HTTP request, and the
  evidence is reproducible with `scripts/verify_oaizib_metadata.py`.
- **UNVERIFIED** — from the PRD or upstream prose, not independently confirmed.
- **NOT ATTEMPTED** — deliberately not downloaded.

Nothing is marked VERIFIED on the strength of the PRD alone. PRD Part V is a plan,
not evidence.

---

## Summary

| Dataset | Access | Role | Used for Model 1 | Blocker |
|---|---|---|---|---|
| OAIZIB-CM | **OPEN** | fine-tuning | **YES** | — |
| OAI (parent) | GATED | — | no | NDA Data Use Certification |
| SKM-TEA | GATED | — | no | ~900 GB, indivisible, gated |
| ShapeMed-Knee | UNVERIFIED | morphology | no | accessibility unverified |
| MedShapeNet 2.0 | UNVERIFIED | morphology | no | accessibility unverified |
| OAI Tissue Segmentations | UNVERIFIED | — | no | provenance unverified |
| KneeXNet | UNVERIFIED | external validation | no | accessibility unverified |
| MRNet | GATED | — | no | no voxel labels |
| Open Knee(s) | UNVERIFIED | — | no | ~10 cadaveric specimens |
| Natural Knee Data | UNVERIFIED | — | no | ~12 specimens |
| KISTI CT Models | UNVERIFIED | — | no | no confirmed distribution |
| Hospital partnership | not acquired | — | no | not yet negotiated |

**One dataset is in use: OAIZIB-CM.** Everything else is deferred with a named
blocker, per instruction.

---

## 1. OAIZIB-CM — the only dataset we use

### 1.1 Identity and licence

| Property | Value | Status |
|---|---|---|
| Host | HuggingFace `YongchengYAO/OAIZIB-CM` | VERIFIED |
| Revision | `d2af7abff1e9bb85212bdd70e11d60bbb21939fa` | VERIFIED |
| Gated | **No** — API reports `gated: false`, `private: false` | VERIFIED |
| Token needed | **No** — anonymous `HEAD` returns 200 on every archive | VERIFIED |
| Licence | **CC-BY-NC-4.0** | VERIFIED (API tag + `info/license`) |
| Commercial use | **NO** | VERIFIED — follows from CC-BY-NC |
| Citation | **Compulsory** | VERIFIED (upstream README) |

The upstream README's download snippet shows `huggingface-cli login --token`, which
implies a token is required. It is not: anonymous range requests succeed. We
verified this rather than inheriting the assumption, because it determines whether
the pipeline needs credential handling at all.

**Compulsory citation** (upstream states *"It is compulsory to cite these papers if
you use the dataset"*):

> Yao Y, Zhong J, Zhang L, Khan S, Chen W. *CartiMorph: A framework for automated
> knee articular cartilage morphometrics.* Medical Image Analysis 91:103035, 2024.
> doi:10.1016/j.media.2023.103035

The underlying OAIZIB segmentations come from:

> Ambellan F, Tack A, Ehlke M, Zachow S. *Automated Segmentation of Knee Bone and
> Cartilage combining Statistical Shape Knowledge and Convolutional Neural Networks:
> Data from the Osteoarthritis Initiative.* Medical Image Analysis, 2019.
> doi:10.1016/j.media.2018.11.009

**Commercial consequence.** CC-BY-NC-4.0 forbids commercial use. A model trained on
OAIZIB-CM therefore **cannot ship in a commercial product**. It is valid for
research, method development and internal evaluation only. This is recorded in the
model card's `DatasetContract.commercial_use_permitted = False`, and PRD §112's
commercial path requires either a licence negotiation or retraining on
hospital-partnership data.

### 1.2 Actual file sizes — VERIFIED by HTTP HEAD

| Archive | Bytes | Size | Range requests | Fetched? |
|---|---:|---:|:---:|:---:|
| `imagesTr.zip` | 10,032,464,145 | **10.03 GB** | yes | **no** |
| `imagesTs.zip` | 2,544,663,978 | **2.54 GB** | yes | partial |
| `labelsTr.zip` | 57,538,129 | 57.5 MB | yes | no |
| `labelsTs.zip` | 14,122,969 | 14.1 MB | yes | whole |
| `info.zip` | 51,680 | 51.7 KB | yes | whole |
| `template_atlas.zip` | 3,625,715 | 3.6 MB | yes | no |

The brief quoted 2.4 GB and 9.3 GB; the measured values are 2.54 GB and 10.03 GB.
Small discrepancies, but we use the measured figures.

**Every archive advertises `Accept-Ranges: bytes`.** This is what makes partial
acquisition possible: `remotezip` reads the ZIP central directory via range requests
and then fetches only the member bytes for the cases we want, at roughly 24 MB per
case, instead of the 2.54 GB whole-archive download.

### 1.3 Contents and split

| Fact | Value | Status |
|---|---|---|
| Total cases | 507 | VERIFIED |
| Train / test | 404 / 103 | VERIFIED |
| Unique `SubjectID` in train | 404 (zero duplicates) | VERIFIED |
| Unique `SubjectID` in test | 103 (zero duplicates) | VERIFIED |
| Train ∩ test `SubjectID` | **0 — no upstream leakage** | VERIFIED |

One knee per subject, so case-level and patient-level splitting coincide *for this
dataset*. We still group by `SubjectID` rather than filename: it is the correct key,
and relying on the coincidence would break silently if the dataset ever gained a
second visit per subject (PRD §63, §115).

### 1.4 Label semantics — identical to our canonical map

Upstream `labels_map`, quoted verbatim from the README:

```python
labels_map = {"1": "Femur", "2": "Femoral Cartilage", "3": "Tibia",
              "4": "Medial Tibial Cartilage", "5": "Lateral Tibial Cartilage"}
```

This matches `kneeai.core.anatomy.DEFAULT_LABEL_MAP` labels 1–5 exactly:

| Label | OAIZIB-CM | `DEFAULT_LABEL_MAP` | Match |
|---:|---|---|:---:|
| 1 | Femur | `femur` | ✓ |
| 2 | Femoral Cartilage | `femoral_cartilage` | ✓ |
| 3 | Tibia | `tibia` | ✓ |
| 4 | Medial Tibial Cartilage | `medial_tibial_cartilage` | ✓ |
| 5 | Lateral Tibial Cartilage | `lateral_tibial_cartilage` | ✓ |

`src/kneeai/datasets/oaizib.py` **asserts** this equality and never remaps. A remap
would be a silent relabelling; an assertion fails loudly if upstream ever changes.

**Not labelled:** patella, patellar cartilage, both menisci, ligaments. Labels 6–7
(patella, patellar cartilage) and 8–9 (menisci) are reserved in
`DEFAULT_LABEL_MAP` and left unused. Model 1 reports both menisci in
`structures_expected_but_missing`.

### 1.5 Metadata — `info.zip` contents VERIFIED

The brief listed OA grading as unconfirmed. **It is present.** `info.zip` (51.7 KB,
downloaded whole) contains:

| Member | Size | Contents |
|---|---:|---|
| `info/kneeSideInfo.csv` | 12,676 B | 507 rows, `oaizib_NNN.nii.gz,side` |
| `info/subInfo_train.xlsx` | 38,049 B | 404 rows + header |
| `info/subInfo_test.xlsx` | 16,126 B | 103 rows + header |
| `info/README.md` | 3,474 B | dataset description, label map, changelog |
| `info/license` | 97 B | CC-BY-NC-4.0 statement |

Both spreadsheets share the columns:

```
SubjectID | CMT-ID | Path | MRBarCode | KneeSide | KLGrade | Gender | Age | BMI
```

- `SubjectID` — the real OAI subject id (e.g. `9005075`). **This is the patient
  grouping key.**
- `CMT-ID` — sequential case number matching the filename (`001` → `oaizib_001.nii.gz`).
- `KLGrade` — **present**, Kellgren–Lawrence grade.

Measured distributions across all 507 cases:

| Field | Distribution |
|---|---|
| `KLGrade` | 0:103, 1:58, 2:108, 3:139, 4:73, **blank:26** |
| `Gender` | 1:262, 2:245 |
| `Age` | n=507, 45–79, mean 61.9 |
| `BMI` | n=506 (**1 blank**), 19.8–44.6, mean 29.3 |

So the cohort spans the full KL 0–4 osteoarthritis range, weighted toward
moderate-to-severe (KL 3 is the largest group). That is the population TKA planning
targets, which makes it a reasonable fine-tuning set — but it is not a
population-representative sample, and 26 cases have no KL grade at all.

> **Parsing note.** An `.xlsx` worksheet omits empty cells entirely. Reading cells in
> document order without honouring each cell's `r` reference shifts every value after
> a blank one column to the left. Doing that initially made `Gender` appear to
> contain ages (`48`, `54`, `57`…) and looked like upstream corruption. It was our
> parser. `scripts/verify_oaizib_metadata.py` resolves cell references properly. The
> figures above are from the corrected parser.

### 1.6 All 507 knees are RIGHT — a material limitation

**VERIFIED.** `kneeSideInfo.csv` reports `right` for all 507 rows, and `KneeSide=1`
for all 507 spreadsheet rows. The two sources agree.

**This dataset contains no left knees.**

Consequences, which must not be discovered later:

1. A model trained only on OAIZIB-CM **has never seen a left knee.** Its performance
   on one is unmeasured, not merely lower.
2. Our augmentation pipeline deliberately does **not** flip left/right
   (`ml/datasets.py`), for the documented reason that a mirrored right knee *is* a
   left knee — so flip augmentation would teach the model to ignore the one cue that
   distinguishes them. That decision is correct, and it means laterality invariance is
   not learned either.
3. The only sound way to run Model 1 on a left knee is to **mirror the volume,
   predict, then mirror the mask back**, recording in `warnings` that this was done.
   Until that path is implemented and evaluated, left knees are out of scope.

This goes into the model card as an explicit limitation **and** a contraindication,
and into `result.json` `warnings` on any left-knee inference.

### 1.7 Case 454 — corrupt, then repaired upstream

The upstream changelog, verbatim:

```
- [15 Feb, 2025] Update file imagesTs/oaizib_454_0000.nii.gz
- [14 Feb, 2025] Identify corrupted files: case 454
```

Read in order: the corruption was identified on 14 February and **the file was
replaced the following day**. So case 454 is probably fine in the current revision.
It is in `imagesTs` — the split we fetch — and its metadata is intact
(`SubjectID 9880455`, KL grade 2, BMI 44.2).

**Decision: exclude by default, overridable via `--include-454`.** Conservative,
matching the brief, but recorded as *likely repaired* rather than as an established
defect, and the download script verifies the file loads if it is included. Excluding
one of 103 test cases costs ~1% of the held-out split; silently training on a corrupt
volume costs more.

### 1.8 What we download, and why

We fetch **the test split only**, partially:

| Archive | Action | Reason |
|---|---|---|
| `imagesTs.zip` | partial, `--limit N` (default 40) | 2.54 GB whole; ~24 MB/case via range requests |
| `labelsTs.zip` | whole (14.1 MB) | small enough that partial fetch adds no value |
| `info.zip` | whole (51.7 KB) | metadata, side, `SubjectID`, KL grade |
| `imagesTr.zip` | **not fetched** | 10.03 GB |
| `labelsTr.zip` | **not fetched** | pairs with `imagesTr` |
| `template_atlas.zip` | not fetched | CLAIR-Knee-103R template, not needed for Model 1 |

**Split discipline.** These 103 cases are upstream's *test* split, but for us they
are the entire available pool, so we split them into **train/val only** and hold
upstream's `imagesTr` out as a future clean external set. We never use our own
validation cases for anything other than model selection, and the reported
validation Dice comes from data the model never trained on.

This does mean Model 1 is fine-tuned on a **small** set (≤40 cases by default). That
is a deliberate budget choice, not an accident, and it is the reason a pretrained
encoder matters (Experiment A). It will show up most in cartilage Dice, which will be
materially lower than bone Dice — cartilage is a 2–3 mm sheet where a one-voxel
boundary error is a large fractional error.

### 1.9 Reproducing the verification

```bash
python scripts/verify_oaizib_metadata.py          # re-checks every VERIFIED claim above
python scripts/download_oaizib.py --limit 40      # partial image fetch + labels + info
```

---

## 2. Deferred datasets, with named blockers

### 2.1 SKM-TEA — blocks Model 2 (meniscus)

- **Blocker:** ~900 GB compressed / ~1.6 TB expanded; requires a Stanford AIMI
  account; **upstream documents that the download cannot be split by track**, so no
  subset is possible.
- **Consequence:** meniscus segmentation is not achievable now. OAIZIB-CM has **zero**
  meniscus labels, so there is no substitute within reach.
- **Status:** NOT ATTEMPTED. Model 2 deferred. Both menisci are reported in
  `structures_expected_but_missing` on every Model 1 case so the application can say
  *"no model has looked"* rather than *"nothing was found"*.

### 2.2 ShapeMed-Knee and MedShapeNet 2.0 — block Model 3 (morphology)

- **Blocker:** accessibility and licence **UNVERIFIED**. Instruction is to verify and
  report, not download.
- **MedShapeNet caveat:** not knee-specific. PRD §46 warns against treating it as a
  primary knee dataset; it is a shape-pretraining supplement at best.
- **Status:** NOT ATTEMPTED. Model 3 deferred.

### 2.3 Experiment B — second label-compatible dataset

- **Blocker:** requires a second dataset with a label map compatible with
  `DEFAULT_LABEL_MAP` 1–5. The only realistic candidates (OAI Tissue Segmentations,
  KneeXNet) are both UNVERIFIED, and OAI Tissue is OAI-derived — so it would need
  `SubjectID` de-duplication against OAIZIB-CM before combining, or it silently leaks
  the same patients into train and test.
- **Status:** deferred.

### 2.4 Experiment C — needs Models 1 and 2

- **Blocker:** requires both Model 1 and Model 2 to exist. Model 2 is blocked by
  SKM-TEA.
- **Status:** deferred.

### 2.5 Others

| Dataset | Blocker |
|---|---|
| **OAI (parent)** | NDA account + Data Use Certification. Also the overlap risk in §2.3. |
| **OAI Tissue Segmentations** | Kaggle re-upload; provenance and licence UNVERIFIED (PRD §49). |
| **KneeXNet** | UNVERIFIED. Reserved for **external validation only** (PRD §50, §64) — must never enter training. |
| **MRNet** | Exam-level pathology labels only, no voxel masks. PRD §51: not equivalent to meniscus segmentation. |
| **Open Knee(s)** | ~10 cadaveric specimens. PRD §19, §45: unsuitable as general ML training data. Biomechanics validation only. |
| **Natural Knee Data** | ~12 specimens. Same reasoning. |
| **KISTI CT Models** | No confirmed distribution URL; access and licence require independent verification (PRD §53). |
| **Hospital partnership** | Not negotiated. PRD §24, §54, §58: the only route to real implant labels (pre-op imaging + plan + actual implant + size + position + post-op). No public anatomy dataset supplies these, so the implant-selection ML model of PRD §23 is not trainable today. |

---

## 3. Standing rules

1. **Never** download anything marked GATED or UNVERIFIED.
2. **Never** use upstream's `imagesTr`/`labelsTr` as validation for a model selected
   on our own split, and never use our test material for model selection.
3. **Always** group splits by `SubjectID`, not filename. `ml.datasets.verify_no_leakage()`
   is run on every manifest build and fails loudly.
4. Any future OAI-derived dataset must be `SubjectID`-de-duplicated against
   OAIZIB-CM before being combined with it (PRD §115).
5. Training reads local files only. The download step is separate and cached; no
   training run touches the network.
6. CC-BY-NC-4.0 means **no commercial deployment** of any model trained on
   OAIZIB-CM. Recorded in the model card, not just here.
