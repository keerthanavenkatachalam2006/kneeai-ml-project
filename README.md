# Knee Intelligence Platform

Converts a knee MRI into an explainable, surgeon-editable 3D surgical plan, and
simulates a constrained robotic execution of that plan.

> **RESEARCH / SIMULATION PROTOTYPE.** Not a medical device. Not clinically
> validated. Not for use in the care of any patient. **Simulation only — no
> hardware control.**

---

## What actually works today

```
uploaded scan (NIfTI / DICOM zip)  or  real knee MRI (OAIZIB-CM)
        ↓
ingest + validation ..... modality, spacing, FOV, laterality; 2D X-ray refused
        ↓                 with a reason rather than silently mishandled
AI segmentation ......... trained 3D SegResNet / UNet, 5 structures
        ↓
3D reconstruction ....... watertight meshes, mm units
        ↓
landmark detection ...... 18 landmarks + patient frame
        ↓
measurements ............ 26 computed, 13 reported unavailable with reasons
        ↓
OA assessment ........... trained classifier, KL >= 2, probability + severity
        ↓                 + per-feature contributions + held-out metrics
implant matching ........ ranked candidates, runner-up and why it lost
        ↓
surgical planning ....... 6 resection cuts, approval + locking + versioning
        ↓
geometric simulation .... virtual cuts, collision, kinematics
        ↓
robotic readiness ....... percentage over 6 weighted criteria, blockers cap it
        ↓
surgeon approval ........ mandatory, named, enforced by raising
        ↓
robot simulation ........ 27 abstract tool poses, keep-out boundary
        ↓
3D viewer ............... Next.js + react-three-fiber
```

Upload a scan at <http://localhost:3000>, and the answer comes back as: is this
osteoarthritis, here are the three orthogonal views, and here is how ready the case
is for a robotic execution of its plan — each with its reasoning attached.

### Models

Two learned components. Everything else is deterministic geometry.

**1. Segmentation.** Two checkpoints are registered; the inference adapter reads
whichever `artifacts/models/seg_oaizib/model_card.json` points at.

| | `0.1.0-local` | `0.2.0-segresnet` |
|---|---|---|
| Architecture | MONAI 3D UNet, 4.81 M params | MONAI SegResNet, 4.70 M params |
| Provenance | trained here, full record | **supplied as a bare state dict, no record** |
| Preprocessing | 1.0 mm iso, RAS, z-score | 0.5 mm iso, RAS, z-score — *recovered by measurement* |
| Mean val Dice | **0.695** | ~0.47 |

`0.2.0-segresnet` is `best_model.pth`. It arrived with no label map, no
`model_config` and no training log, so its architecture and preprocessing were
recovered by grid search against ground truth — see `scripts/identify_checkpoint.py`
and the evidence block in `scripts/register_best_model.py`. GroupNorm(8) vs affine
InstanceNorm was decided by measurement (0.455 vs 0.127 Dice); both fit the tensor
shapes identically. It measures **worse** than the model trained here, and its error
pattern — high recall, inflated volumes, soft boundaries — is that of an undertrained
network rather than a preprocessing mismatch. All 16 axis flips were checked; none
beat the identity.

To switch back: `cp artifacts/models/seg_oaizib/model_card.previous.json
artifacts/models/seg_oaizib/model_card.json`.

**2. OA classification.** `kneeai.oa` — L2-regularised logistic regression on seven
cartilage-geometry features, fitted to the cohort's own Kellgren-Lawrence grades
dichotomised at KL ≥ 2. 48 training knees, 12 validation knees, patient-level split
from the dataset manifest. Its measured held-out performance is attached to every
prediction it makes and rendered next to the verdict in the interface, because a
probability from a model fitted to 48 samples should never appear without it.

| | |
|---|---|
| Segmentation trained on | OAIZIB-CM, real OAI knee MRI |
| Split | patient-level, KL-stratified, zero shared patients |
| Structures | femur, tibia, femoral cartilage, medial + lateral tibial cartilage |

Validation Dice (30 train / 10 val patients, laptop run):

| Structure | Dice |
|---|---|
| Femur | 0.913 |
| Tibia | 0.890 |
| Femoral cartilage | 0.794 |
| Medial tibial cartilage | 0.455 |
| Lateral tibial cartilage | 0.423 |
| **Mean** | **0.695** |

Bone is usable. **Cartilage is not** — the tibial cartilages are 0.1% of all
voxels and the laptop run resampled to 1.0 mm, which discards most of the 0.365 mm
in-plane detail that defines them. `notebooks/ml/01_train_oaizib_colab.ipynb`
fixes this with class-balanced sampling, measured CE weights, 404 cases and an
SSL-pretrained SwinUNETR at 0.7 mm.

---

## Running it

### 1. Install

```bash
make install          # core + imaging + api + dev
make web-install      # frontend
```

### 2. Generate a case

```bash
make demo             # synthetic phantom, no patient data, always works
```

Or with the trained model on a real scan:

```bash
make infer CASE=oaizib-417
```

### 3. Start the stack

Two terminals:

```bash
make api              # FastAPI on 127.0.0.1:8000
make web              # dashboard on http://localhost:3000
```

Open <http://localhost:3000>.

### 3b. Or upload your own scan

The dashboard's first step is **Upload scan**. Drop in a NIfTI (`.nii`, `.nii.gz`),
a DICOM series as a `.zip`, or a single `.dcm`, pick the operative side, and the
whole pipeline runs on a background worker with a pollable progress bar. Roughly
4 minutes on CPU, one upload at a time.

What each input does:

| Input | What happens |
|---|---|
| Knee MR volume | full pipeline: segmentation → 3D → measurements → OA verdict → plan → readiness |
| CT volume | stored and validated, segmentation **refused** — the model card declares MR and cross-modality use is unvalidated |
| 2D radiograph | stored, then stopped with a reason: every model here is 3D, and no KL-graded radiograph dataset is present to train a 2D one |
| Unknown side | segmentation, measurements and the OA verdict still run; implant matching, planning and simulation are withheld |

Cases produced before the OA and readiness stages existed can be brought up to date
without re-running the network, since both are pure functions of artefacts already on
disk:

```bash
make backfill
```

### 4. Walk the demo

1. Pick a case from the selector. The badge shows whether the geometry is
   `AI PREDICTION`, `GROUND TRUTH`, `PHANTOM` or `MOCK`.
2. Inspect the 3D knee — orbit, isolate structures, adjust opacity, clip.
3. Read the measurements. Note the **unavailable** section and its reasons.
4. Read the implant card — recommended size, fit metrics, runner-up, why it lost.
5. Enter a surgeon name and approve the plan. Watch the safety checks run.
6. Run the robot simulation. The tool follows the trajectory, the resection
   appears, the command log highlights each step.
7. Press **Emergency stop** mid-run to see the state machine terminate.

Try running the simulation *before* approving. The button is disabled, and the
API returns 409 even if you bypass the UI.

---

## Training a better model

The laptop cannot do this — 8 GB unified memory and a 9.3 GB training archive.
Use Colab:

1. Upload `notebooks/ml/01_train_oaizib_colab.ipynb` to
   [Colab](https://colab.research.google.com).
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Run all.**

`PROFILE = "full"` is 404 cases with SwinUNETR, roughly 3 hours.
`PROFILE = "quick"` is 150 cases with UNet, roughly 45 minutes.

The notebook downloads the data itself and writes the checkpoint in the format
`kneeai.ml.infer` expects, so it drops straight in:

```bash
unzip seg_oaizib.zip -d artifacts/models/seg_oaizib/
make model-card
make infer CASE=oaizib-417
```

---

## Architecture

```
apps/web/                Next.js dashboard + react-three-fiber viewer
        │ HTTP (proxied same-origin)
src/kneeai/api/          FastAPI: cases, meshes, approval, simulation
        │
        ├── ml/          segmentation model, registry, metrics, contract
        ├── robotics/    safety state machine, abstract commands, plan document
        │
        ├── core/        frames, geometry, units, provenance, case state
        ├── imaging/     DICOM/NIfTI ingest, validation, anonymisation
        ├── oa/          OA features, classifier, assessment
        ├── recon/       voxel masks -> watertight meshes
        ├── landmarks/   geometric detection + patient frame
        ├── measure/     bone, alignment, soft tissue
        ├── implants/    catalog, footprints, matching
        ├── planning/    strategy, resections, optimiser, approval
        └── simulation/  virtual cuts, collision, kinematics
```

**Two** components are learned models: segmentation (`ml/`) and OA classification
(`oa/`). Everything else — reconstruction, landmarks, measurement, implant matching,
planning, simulation, and the robotic readiness score — is deterministic geometry,
which is why it needed no training and does not hallucinate.

The readiness percentage is deliberately in that second group. It is a weighted mean
over six criteria scored from the artefacts, with every weight and sub-score reported
so the total can be recomputed by hand. Anything that genuinely prevents execution
imposes a *ceiling* rather than contributing a term, because a weighted average will
otherwise report a comfortable 78 % for a case whose geometric simulation failed.

### The ML → application contract

`docs/ML_CONTRACT.md`, enforced by `kneeai.ml.contract`. Four rules:

1. **No accuracy metrics in `result.json`.** Dice requires ground truth, which
   does not exist at inference time. Reference metrics live on the model card.
2. **`source` is mandatory** and surfaced in every view and export.
3. **`side` is mandatory** — a wrong-site-surgery class error otherwise.
4. **`affine` is mandatory** — voxel indices are not millimetres.

---

## Known limitations

**Alignment cannot be computed from MRI.** HKA, MLDFA and MPTA need the femoral
head and ankle centres. A knee MRI covers ~112 mm and contains neither. The
patient frame falls back to `shaft_axis+TEA`. A long-leg radiograph or full-leg CT
is required, and that is a missing product input, not a bug.

**Cartilage segmentation is weak** — Dice 0.42–0.46 on the tibial cartilages, and
they reconstruct into 150–200 disconnected components. `ml/postprocess.py` has the
cleanup and it is not yet wired into the inference path.

**No meniscus or patella model.** OAIZIB-CM has no such labels. SKM-TEA does, but
it is 900 GB behind a Stanford account and cannot be subsetted.

**All training knees are right knees.** Left-knee performance is unmeasured and
rests entirely on sagittal mirror augmentation.

**Training data is CC-BY-NC-4.0.** This model may not be used commercially. A
commercially-licensed or hospital-partnered dataset is required for a product.

**Implant geometry is a reference catalogue**, not real manufacturer data.

**No authentication.** The API binds to loopback and must not be exposed without
adding auth first. `pyjwt` and `passlib` are already declared for that.

**Single split, no cross-validation.** The reported Dice has a wide and
unquantified confidence interval on 10 validation patients.

---

## Citation

OAIZIB-CM requires this citation:

> Yao Y, Zhong J, Zhang L, Khan S, Chen W. CartiMorph: A framework for automated
> knee articular cartilage morphometrics. *Medical Image Analysis* 91:103035, 2024.
> doi:10.1016/j.media.2023.103035

Dataset: [YongchengYAO/OAIZIB-CM](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM)
(CC-BY-NC-4.0). Pretrained encoder from
[MONAI's self-supervised SwinUNETR](https://github.com/Project-MONAI/tutorials/tree/main/self_supervised_pretraining).
