# Knee Intelligence Platform

Converts a knee MRI into an explainable, surgeon-editable 3D surgical plan, and
simulates a constrained robotic execution of that plan.

> **RESEARCH / SIMULATION PROTOTYPE.** Not a medical device. Not clinically
> validated. Not for use in the care of any patient. **Simulation only — no
> hardware control.**

---

## What actually works today

```
real knee MRI (OAIZIB-CM)
        ↓
AI segmentation ......... trained 3D UNet, 5 structures
        ↓
3D reconstruction ....... watertight meshes, mm units
        ↓
landmark detection ...... 18 landmarks + patient frame
        ↓
measurements ............ 26 computed, 13 reported unavailable with reasons
        ↓
implant matching ........ ranked candidates, runner-up and why it lost
        ↓
surgical planning ....... 6 resection cuts, approval + locking + versioning
        ↓
geometric simulation .... virtual cuts, collision, kinematics
        ↓
surgeon approval ........ mandatory, named, enforced by raising
        ↓
robot simulation ........ 27 abstract tool poses, keep-out boundary
        ↓
3D viewer ............... Next.js + react-three-fiber
```

### Model

| | |
|---|---|
| Architecture | MONAI 3D UNet, 4.81 M parameters |
| Trained on | OAIZIB-CM, real OAI knee MRI |
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
        ├── recon/       voxel masks -> watertight meshes
        ├── landmarks/   geometric detection + patient frame
        ├── measure/     bone, alignment, soft tissue
        ├── implants/    catalog, footprints, matching
        ├── planning/    strategy, resections, optimiser, approval
        └── simulation/  virtual cuts, collision, kinematics
```

Only **one** component is a learned model. Everything after segmentation is
deterministic geometry — which is why it needed no training and does not
hallucinate.

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
