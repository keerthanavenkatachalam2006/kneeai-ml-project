# PRD — Explainable 3D Patient-Specific Knee Intelligence & Planning Platform

**Status:** Development Blueprint  
**Purpose:** Full implementation plan for the hackathon and research prototype  
**Primary basis:** Supplied MedTech problem statement + dataset corpus and project strategy developed in this conversation

---

# 1. Executive Summary

The problem statement asks for an AI-assisted platform that:

1. Identifies the femur, tibia, and medial meniscus from knee medical images.
2. Measures medial meniscus thickness.
3. Studies the relationship between meniscus measurements, osteoarthritis (OA), age, and sex.
4. Measures femoral and tibial anatomy.
5. Compares those measurements with knee implant dimensions.
6. Provides suitable/ranked implant-size candidates.
7. Displays the AI-generated measurements for clinical/research review.

Our project will **fully cover all of these requirements** and then extend them.

The expanded platform will add:

- 3D knee reconstruction where the imaging data supports it.
- 3D anatomical measurements.
- 3D femur morphology analysis.
- Patient-specific anatomical profiles.
- Explainable implant matching.
- Ranked implant candidates with measurable reasons.
- Virtual implant fitting as an advanced planning feature.
- Implant fit analysis such as dimensional mismatch and, where geometry supports it, coverage/overhang.
- AI quality/confidence indicators.
- Human-review warnings.
- Cross-dataset validation.
- Cross-MRI robustness testing.
- Age/sex-aware reference analysis.
- Research dashboards.
- Future longitudinal tracking and CT support.

The core philosophy is:

> **Understand → Measure → Analyze → Compare → Explain → Assist**

This is a clinical decision-support and research prototype, not an autonomous diagnostic or surgical system.

---

# 2. Problem Statement Alignment

## 2.1 Mandatory PS Requirements

| PS requirement | Our implementation |
|---|---|
| Identify femur | AI segmentation |
| Identify tibia | AI segmentation |
| Identify medial meniscus | AI segmentation |
| Measure meniscus thickness | Measurement engine |
| Compare OA/non-OA cases | OA research engine |
| Analyze age and sex | Demographic analysis |
| Measure femoral anatomy | 2D/3D measurement engine |
| Measure tibial anatomy | 2D/3D measurement engine |
| Compare with implant database | Implant matching engine |
| Recommend/rank implant sizes | Ranked candidate engine |
| Visualize results | Interactive 2D/3D interface |

## 2.2 Our Expansion Beyond the PS

The project will not stop at satisfying the listed requirements.

We will extend the system with:

1. 3D patient-specific knee reconstruction.
2. 3D anatomical morphology analysis.
3. Shape comparison using ShapeMed-Knee.
4. Explainable implant-fit scoring.
5. Ranked implant alternatives.
6. "Why this implant?" explanation.
7. Patient anatomical profile.
8. Age/sex-aware reference comparison.
9. Confidence and quality warnings.
10. External validation on independent datasets.
11. Cross-MRI-domain robustness testing.
12. Indian-population external OA validation.
13. Interactive 3D implant visualization.
14. Virtual implant fitting where geometry/data quality supports it.
15. Coverage/overhang analysis where the available geometry supports valid calculations.
16. Research dashboard.
17. Longitudinal analysis as a later extension.
18. CT integration as a later extension.

---

# 3. Product Vision

## Vision

> **Turn a knee medical scan into a measurable, explainable, patient-specific anatomical model that supports OA research and personalized knee implant planning.**

The system should not behave like a black box.

Instead of:

> "Implant Size M"

it should show:

> "The patient's anatomy was measured as X. Candidate M has the smallest validated dimensional mismatch among the available candidates."

---

# 4. Core Product Flow

```text
                    PATIENT KNEE SCAN
                           |
                           v
                  IMAGE QUALITY CHECK
                           |
                           v
                   3D/2D IMAGE PROCESSING
                           |
                           v
                    AI SEGMENTATION
                           |
            +--------------+--------------+
            |              |              |
            v              v              v
          FEMUR          TIBIA        MENISCUS
            |              |              |
            +--------------+--------------+
                           |
                           v
                 PATIENT-SPECIFIC ANATOMY
                           |
                           v
                    MEASUREMENT ENGINE
                           |
             +-------------+-------------+
             |                           |
             v                           v
        OA ANALYSIS                IMPLANT PLANNING
             |                           |
      Age / Sex / OA             Implant specifications
             |                           |
             +-------------+-------------+
                           |
                           v
                  3D/ANATOMICAL PROFILE
                           |
                           v
                EXPLAINABLE RESULTS
                           |
                           v
                 CLINICIAN/RESEARCHER
                       REVIEW
```

---

# 5. What "3D" Means in This Project

The project should not use the word "3D" merely because an MRI contains multiple slices.

The intended 3D pipeline is:

```text
3D MRI volume
      |
      v
3D segmentation
      |
      v
3D femur/tibia/meniscus representation
      |
      v
3D surface/mesh
      |
      v
3D measurements
      |
      v
3D anatomical model
      |
      v
3D implant comparison
```

The extent of 3D reconstruction will depend on the available image quality, voxel spacing, segmentation quality, and implant geometry.

---

# 6. Target Users

## Primary

- Orthopedic researchers.
- Medical imaging researchers.
- Radiology researchers.
- Pre-operative planning researchers.

## Secondary

- ML researchers.
- Medical AI developers.
- Hackathon judges/demo users.

---

# 7. System Modules

# Module 1 — Image Input and Quality Control

## Input

Primary development input:

- Knee MRI.

Potential future input:

- Knee X-ray.
- CT.

## Responsibilities

1. Load image.
2. Validate file.
3. Detect compatible imaging format.
4. Preserve image dimensions.
5. Preserve pixel/voxel spacing.
6. Identify image orientation where possible.
7. Flag low-quality or incompatible input.

## Critical Rule

Physical measurements require physical image scale.

The system must retain:

- Pixel spacing.
- Slice spacing.
- Voxel dimensions.

Never convert pixels directly into millimetres without correct spacing information.

---

# Module 2 — AI Segmentation

## Goal

Find:

- Femur.
- Tibia.
- Medial meniscus.

## Simple explanation

Segmentation means:

> The AI marks exactly which pixels/voxels belong to each knee structure.

Example:

```text
MRI
 |
 +-- Femur mask
 +-- Tibia mask
 +-- Medial meniscus mask
```

## Model strategy

Do not force the first version to use one giant model.

### Bone model

Primary development data:

- OAIZIB-CM.

Outputs:

- Femur mask.
- Tibia mask.

### Meniscus model

Primary development data:

- SKM-TEA.

Output:

- Medial meniscus mask.

Then combine them into a unified inference pipeline.

---

# 8. Dataset Strategy

## 8.1 OAIZIB-CM — Core Dataset

Source:

https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM

Role:

- Femur segmentation.
- Tibia segmentation.
- OA analysis.
- Age analysis.
- Sex analysis.
- BMI analysis if useful.
- Knee-side information.
- Measurement development.

The supplied corpus describes 507 3D DESS knees with femur/tibia/cartilage masks and clinical fields including KL grade, gender, age, BMI, and knee side.

This is the primary development dataset for the bone and OA layers.

---

## 8.2 SKM-TEA — Core Meniscus Dataset

Sources:

https://aimi.stanford.edu/datasets/skm-tea-knee-mri

https://github.com/StanfordMIMI/skm-tea

Role:

- Meniscus segmentation.
- Cartilage/tissue segmentation.
- Pathology-related auxiliary research.
- Domain-shift evaluation.

The supplied corpus describes 155 qDESS knees with manual meniscus segmentation and pathology annotations.

This is the main dataset for the meniscus bottleneck.

---

## 8.3 SKI10 — Independent Bone Validation

Source:

https://ski10.grand-challenge.org/Download/

Role:

- External femur/tibia segmentation validation.
- Robustness testing.
- Pathological/post-surgical case evaluation.

Use primarily for independent evaluation rather than automatically adding it to training.

---

## 8.4 Open Knee(s)

Sources:

http://archive.simtk.org/oks/MRI/

https://simtk.org/projects/openknee

Role:

- MRI sequence diversity.
- Multi-plane imaging.
- Robustness testing.
- 3D geometry/FE research.
- Potential future 3D work.

This is a small dataset, so its value is diversity rather than sample count.

---

## 8.5 MRNet

Source:

https://aimi.stanford.edu/datasets/mrnet-knee-mris

Role:

- Auxiliary pathology classification.
- Meniscus-tear-related research.
- Optional encoder/auxiliary learning.

Important limitation:

MRNet has exam-level labels and is not a replacement for a detailed meniscus segmentation dataset.

---

## 8.6 fastMRI Knee

Sources:

https://cai2r.net/resources/fastmri-dataset/

https://fastmri.med.nyu.edu/

https://github.com/facebookresearch/fastMRI

Role:

- Large-scale MRI variety.
- Self-supervised pretraining.
- Domain robustness.

Use only after the supervised baseline is functioning.

---

## 8.7 fastMRI Abnormality Annotations

Source:

https://github.com/fcaliva/fastMRI_BB_abnormalities_annotation

Role:

- Weak localization.
- Abnormality research.
- Auxiliary training.

A bounding box is a rough region around a finding, not an exact anatomical outline.

---

## 8.8 ShapeMed-Knee — 3D Morphology

Sources:

https://huggingface.co/datasets/aagatti/ShapeMedKnee

https://github.com/gattia/shapemedknee

Role:

- 3D femur shape analysis.
- Statistical shape modeling.
- Patient-specific morphology.
- Shape comparison.
- Advanced implant planning.

This is one of the main datasets enabling the project's 3D expansion.

---

## 8.9 TotalSegmentator

Source:

https://zenodo.org/records/10047292

Role:

- CT-based bone research.
- Future CT integration.
- Additional 3D anatomy research.

Do not make it a core dependency until exact label availability and licensing are verified for the intended use.

---

## 8.10 MedShapeNet

Sources:

https://medshapenet.ikim.nrw/

https://github.com/GLARKI/MedShapeNet2.0

Role:

- Additional 3D shape references.
- Shape research.
- Visualization.
- Future 3D learning.

Optional for the first implementation.

---

## 8.11 Knee OA Severity X-ray Dataset

Sources:

https://data.mendeley.com/datasets/56rmx5bjcr/1

https://github.com/PingjunChen/GradingKneeOA

Role:

- OA severity analysis.
- X-ray research.
- Additional OA examples.

Important:

The supplied research notes identify this as OAI-derived. It must not be treated as independent from OAI-based datasets.

---

## 8.12 Digital Knee X-ray Images

Source:

https://data.mendeley.com/datasets/t9ndx37v5h/1

Role:

- External OA validation.
- Indian-population evaluation.
- Expert-reading comparison where labels support it.

---

## 8.13 KNEEL

Source:

https://github.com/MIPT-Oulu/KNEEL

Role:

- Knee landmarks.
- Joint-space measurement.
- Radiograph research.

Optional.

---

# 9. Data Integration Strategy

We do NOT merge all datasets into one giant training dataset.

Instead:

```text
OAIZIB-CM
    |
    +--> Bone model
    +--> OA research
    +--> Demographic analysis

SKM-TEA
    |
    +--> Meniscus model

SKI10
    |
    +--> External bone validation

ShapeMed-Knee
    |
    +--> 3D morphology

Indian X-rays
    |
    +--> External OA validation

fastMRI
    |
    +--> Optional self-supervised pretraining
```

We combine **models, measurements, research outputs, and validation results**, not necessarily patient records.

---

# 10. Critical Patient-Overlap Rule

This is a major research requirement.

OAI-derived datasets can contain the same underlying patients.

Potential overlap must be checked between:

- OAIZIB-CM.
- ShapeMed-Knee.
- OAI-derived X-ray datasets.

## Required process

```text
Extract subject IDs
        |
        v
Create master ID table
        |
        v
Find overlaps
        |
        v
Create patient-level splits
        |
        v
Prevent leakage
```

Never randomly split MRI slices.

All data belonging to the same patient must remain in the same split.

---

# 11. Unified Metadata Schema

Create a master table:

```text
patient_id
dataset
knee_side
image_path
image_type
sequence
scanner
vendor
age
sex
BMI
OA_grade
femur_mask
tibia_mask
meniscus_mask
pixel_spacing
slice_spacing
measurement_source
implant_label
```

Not every dataset will contain every field.

Missing data must remain missing.

Never invent values.

---

# 12. Module 3 — 3D Knee Reconstruction

## Goal

Convert segmented 3D image regions into patient-specific 3D anatomical representations.

Pipeline:

```text
3D MRI
  |
  v
3D segmentation
  |
  v
Femur / Tibia / Meniscus masks
  |
  v
Post-processing
  |
  v
3D surface / mesh
  |
  v
Interactive patient-specific model
```

## Outputs

- Femur 3D surface.
- Tibia 3D surface.
- Meniscus 3D representation where reliable.
- Measurement landmarks.
- Anatomical axes where defined.

## Quality requirement

3D reconstruction must be validated for:

- Segmentation quality.
- Surface quality.
- Anatomical correctness.
- Measurement consistency.

---

# 13. Module 4 — Anatomical Measurement Engine

The measurement engine receives segmentation masks/meshes.

## Femur

Potential measurements:

- Width.
- Front-to-back dimension.
- Other implant-relevant dimensions.
- 3D shape descriptors.

## Tibia

Potential measurements:

- Width.
- Front-to-back dimension.
- Other implant-relevant dimensions.
- 3D shape descriptors.

## Meniscus

Potential measurements:

- Thickness.
- Thickness at predefined locations.
- Optional 3D thickness distribution where validated.

### Principle

```text
AI finds anatomy
       |
       v
Geometry calculates measurements
       |
       v
Measurements reported in mm
```

---

# 14. Meniscus Thickness Module

## Pipeline

```text
MRI
 |
 v
Meniscus segmentation
 |
 v
Predefined measurement locations
 |
 v
Thickness calculation
 |
 v
Measurements in mm
```

Example format:

```text
Anterior:  X.X mm
Middle:    X.X mm
Posterior: X.X mm
```

The exact anatomical locations must be fixed before final research analysis.

Do not choose locations based on the results.

---

# 15. Module 5 — OA Research Engine

Inputs:

- Meniscus measurements.
- OA grade.
- Age.
- Sex.
- Optional BMI.
- Knee side.

## Research questions

### RQ1

Is medial meniscus thickness associated with OA severity?

### RQ2

Does age relate to meniscus thickness?

### RQ3

Are there measurable differences between male and female groups?

### RQ4

Do meniscus measurements provide useful information alongside age and sex?

### RQ5

Does the relationship remain after accounting for relevant demographic variables?

## Scientific wording

Use:

> "Associated with"

rather than:

> "Causes"

unless the study design supports a causal claim.

---

# 16. Age/Sex-Aware Reference Analysis

For a patient:

```text
Age: 62
Sex: Female
Meniscus: 3.7 mm
```

Compare against a defined reference group from the dataset.

Example:

```text
Reference:
Female patients
Age 60–65

Patient:
3.7 mm

Comparison:
Below the reference distribution
```

Do not automatically call the patient "abnormal."

Use:

> "Compared with this study population."

---

# 17. Module 6 — Patient Anatomical Profile

Generate a single structured profile:

```text
PATIENT KNEE PROFILE

Demographics
Age:
Sex:
Knee side:

FEMUR
Width:
AP dimension:
3D shape information:

TIBIA
Width:
AP dimension:
3D shape information:

MEDIAL MENISCUS
Thickness:
Location-wise measurements:

OA ANALYSIS
OA grade:
Reference comparison:

MODEL QUALITY
Segmentation quality:
Measurement quality:

IMPLANT CANDIDATES
Candidate 1:
Candidate 2:
Candidate 3:
```

This is the central result object for the application.

---

# 18. Module 7 — 3D Morphology Analysis

## Goal

Understand the patient's overall bone shape, not only a few measurements.

Use ShapeMed-Knee as the main reference source.

Pipeline:

```text
Patient femur
      |
      v
3D representation
      |
      v
Compare against reference shapes
      |
      v
Shape similarity / morphology analysis
      |
      v
Support implant planning
```

Potential outputs:

- Shape similarity.
- Shape clusters.
- Morphological variation.
- Reference shape comparison.

This should be presented as research/morphology information, not a clinical diagnosis.

---

# 19. Module 8 — Implant Database

Create a structured database:

```text
manufacturer
implant_model
component
size
femur_AP
femur_ML
tibia_AP
tibia_ML
other_dimensions
source_document
source_url
license_notes
```

Potential research sources include public manufacturer surgical/product documents and published implant-planning research.

Every dimension must have a traceable source.

Do not manually invent dimensions.

---

# 20. Module 9 — Explainable Implant Matching

## Basic approach

```text
Patient anatomy
      |
      v
Femur/tibia measurements
      |
      +
      |
Implant specification
      |
      v
Dimensional comparison
      |
      v
Candidate ranking
```

For every candidate, calculate a defined mismatch.

Example:

```text
Patient femur:
71.2 × 52.8 mm

Candidate M:
70.8 × 52.7 mm

Differences:
0.4 mm
0.1 mm
```

The exact fit score must be defined mathematically and validated.

---

# 21. Ranked Implant Candidates

Example:

```text
1. Size M
   Best dimensional match

2. Size L
   Alternative

3. Size S
   Larger mismatch
```

The system should say:

> "Candidate implant sizes for clinical review."

It must not say:

> "This is the implant the patient needs."

---

# 22. Module 10 — 3D Virtual Implant Fitting

This is the major expansion beyond the PS.

## Goal

Place candidate implant geometry against the patient's reconstructed anatomy.

Pipeline:

```text
Patient 3D femur/tibia
        |
        v
Candidate implant geometry
        |
        v
Alignment / placement
        |
        v
Fit analysis
        |
        v
Visual comparison
```

Candidates:

```text
Size S
Size M
Size L
```

The user can switch between candidates.

---

# 23. Implant Fit Analysis

Where geometry and validation support it, calculate:

- AP mismatch.
- Width mismatch.
- Surface/edge mismatch.
- Coverage.
- Potential overhang.
- Potential under-coverage.
- Alignment-related geometric measures where appropriately defined.

Example:

```text
SIZE M

AP mismatch:      X.X mm
Width mismatch:   X.X mm
Coverage:         XX%
Overhang:         X.X mm

Overall geometric score: XX
```

All values must be generated from actual geometry.

No fabricated demo numbers in the final product.

---

# 24. "Why This Implant?" Feature

This should be a prominent UI action.

When the user clicks:

> **Why Size M?**

show:

```text
PATIENT ANATOMY
        |
        v
Measured dimensions
        |
        v
IMPLANT M
        |
        +--> AP difference
        +--> Width difference
        +--> Coverage
        +--> Overhang
        |
        v
Why it ranked highest
```

This is the central explainability feature.

---

# 25. 3D Interactive Viewer

The interface should allow:

- Rotate.
- Zoom.
- Pan.
- Toggle femur.
- Toggle tibia.
- Toggle meniscus.
- Toggle implant.
- Change implant candidate.
- Display measurements.
- Display measurement points.
- Display fit warnings.

Suggested visualization:

```text
+--------------------------------+
|                                |
|       INTERACTIVE 3D KNEE      |
|                                |
|      Femur + Tibia + Meniscus  |
|             + Implant          |
|                                |
+--------------------------------+
| Measurements                   |
| Femur: XX mm                   |
| Tibia: XX mm                   |
| Meniscus: XX mm                |
+--------------------------------+
| Implant: M                      |
| Fit: Best candidate             |
+--------------------------------+
```

---

# 26. Module 11 — Confidence and Quality

Every important AI output should have a quality indicator.

Example:

```text
Femur segmentation
HIGH QUALITY

Tibia segmentation
HIGH QUALITY

Meniscus segmentation
REVIEW RECOMMENDED
```

If percentages are displayed, confidence must be calibrated and validated.

Do not equate raw neural-network probability with medical certainty.

---

# 27. Human Review Workflow

The user should be able to:

1. View original scan.
2. View segmentation.
3. Toggle structures.
4. Inspect measurement locations.
5. Inspect measurements.
6. Inspect implant candidates.
7. Inspect fit explanation.
8. Flag questionable results.
9. Manually review low-quality cases.

The system assists the user; it does not replace them.

---

# 28. External Validation

## Bone

Train:

```text
OAIZIB-CM
```

Test:

```text
SKI10
```

## Meniscus

Train:

```text
SKM-TEA
```

Then evaluate transfer to compatible external data only after validation.

## MRI robustness

Test across:

- OAIZIB-CM.
- SKM-TEA.
- Open Knee where technically compatible.

## OA

Use the Indian X-ray dataset for external evaluation of compatible OA tasks.

---

# 29. Evaluation Metrics

## Segmentation

- Dice score.
- IoU.
- Hausdorff distance where appropriate.
- Boundary error.

Simple interpretation:

> How closely does the AI's outline match the reference outline?

## Measurements

- Mean absolute error.
- Median absolute error.
- Measurement distribution.
- Bland–Altman analysis where appropriate.

## OA analysis

- Correlation.
- Group comparison.
- Regression.
- Effect size.
- Confidence intervals.
- Appropriate classification metrics if a classification task is defined.

## Implant matching

If using geometric matching:

- Dimensional mismatch.
- Candidate ranking.
- Coverage/overhang metrics where supported.

If actual implant-label data becomes available:

- Exact-size accuracy.
- Within ±1 size accuracy.
- Confusion matrix.

---

# 30. Research Experiment Plan

## Experiment 1 — Bone Segmentation

Question:

> Can the model identify femur and tibia accurately?

Train:

- OAIZIB-CM.

Test:

- Held-out OAIZIB-CM patients.
- SKI10.

---

## Experiment 2 — Meniscus Segmentation

Question:

> Can the model identify the medial meniscus?

Train:

- SKM-TEA.

Test:

- Held-out SKM-TEA.
- Carefully selected external cases if available.

---

## Experiment 3 — Meniscus Measurement

Question:

> Are AI-derived meniscus measurements reliable?

Compare:

- AI measurement.
- Reference/expert measurement.

---

## Experiment 4 — OA Relationship

Question:

> Is meniscus thickness associated with OA severity?

Analyze:

- Thickness.
- OA grade.
- Age.
- Sex.

---

## Experiment 5 — Age/Sex Reference Analysis

Question:

> How does a patient's measurement compare with a relevant age/sex group?

Use:

- OAIZIB-CM.

---

## Experiment 6 — 3D Morphology

Question:

> Can we characterize patient-specific femur shape beyond simple width/AP measurements?

Use:

- ShapeMed-Knee.
- Patient-specific 3D reconstruction where supported.

---

## Experiment 7 — Implant Matching

Question:

> Can anatomical measurements identify geometrically suitable implant candidates?

Compare:

- Patient anatomy.
- Implant dimensions.

---

## Experiment 8 — 3D Virtual Fitting

Question:

> Does virtual placement provide additional useful information beyond simple dimension matching?

Evaluate:

- Coverage.
- Overhang.
- Surface mismatch.
- Candidate ranking.

Only report metrics supported by the available implant geometry.

---

## Experiment 9 — External Generalization

Question:

> Does the model work on data it did not train on?

Use:

- SKI10.
- Open Knee where compatible.
- Indian X-ray dataset for appropriate OA tasks.

---

## Experiment 10 — Self-Supervised Pretraining

Optional.

Compare:

```text
Direct supervised training
          VS
fastMRI pretraining
          +
supervised fine-tuning
```

Only retain it if validated improvement is meaningful.

---

# 31. Development Plan

# Phase 0 — Dataset Audit

Before model development:

- Access datasets.
- Inspect files.
- Inspect labels.
- Check image formats.
- Check image spacing.
- Check MRI sequences.
- Extract subject IDs.
- Check overlap.
- Check licenses.
- Build metadata table.

### Deliverable

Clean dataset inventory and master metadata table.

---

# Phase 1 — Bone Segmentation

Build:

```text
MRI → Femur + Tibia
```

Train with OAIZIB-CM.

Validate internally.

Test externally on SKI10.

### Deliverable

Validated bone segmentation baseline.

---

# Phase 2 — Meniscus Segmentation

Build:

```text
MRI → Medial Meniscus
```

Train using SKM-TEA.

Validate internally.

### Deliverable

Validated meniscus segmentation baseline.

---

# Phase 3 — Unified Segmentation

Combine:

```text
Bone model
+
Meniscus model
```

Output:

```text
Femur
Tibia
Medial meniscus
```

### Deliverable

One unified knee segmentation pipeline.

---

# Phase 4 — 3D Reconstruction

Convert segmentation masks into:

- 3D surfaces.
- 3D meshes.
- Patient-specific anatomical visualization.

### Deliverable

Interactive 3D knee model.

---

# Phase 5 — Measurement Engine

Implement:

- Femur measurements.
- Tibia measurements.
- Meniscus thickness.
- 3D measurements where validated.
- Measurement visualization.

### Deliverable

Validated anatomical measurement engine.

---

# Phase 6 — OA Research

Build:

- OA analysis.
- Age analysis.
- Sex analysis.
- Reference comparison.
- Statistical dashboard.

### Deliverable

OA research module.

---

# Phase 7 — 3D Morphology

Integrate ShapeMed-Knee.

Build:

- Shape representation.
- Shape comparison.
- Patient morphology profile.

### Deliverable

3D morphology module.

---

# Phase 8 — Implant Database

Build structured implant database.

Every dimension must have a source.

### Deliverable

Versioned implant specification database.

---

# Phase 9 — Implant Matching

Build:

- Measurement matching.
- Candidate ranking.
- Explainable fit score.
- "Why this implant?" output.

### Deliverable

Explainable implant candidate engine.

---

# Phase 10 — 3D Virtual Implant Fitting

Build:

- Implant placement.
- Candidate switching.
- Geometric fit analysis.
- Coverage/overhang where supported.
- 3D visualization.

### Deliverable

3D patient-specific implant planning prototype.

---

# Phase 11 — External Validation

Run final experiments on independent datasets.

### Deliverable

Validation report.

---

# Phase 12 — Product Integration

Build final:

- Backend.
- Frontend.
- 2D viewer.
- 3D viewer.
- Research dashboard.
- Patient profile.
- Implant planner.
- Explainability panel.

### Deliverable

End-to-end working application.

---

# 32. Hackathon MVP

The MVP should not wait for every advanced feature.

## Mandatory demo

```text
Knee MRI
   ↓
Femur/Tibia/Meniscus segmentation
   ↓
3D knee visualization
   ↓
Measurements
   ↓
OA-related analysis
   ↓
Implant candidates
   ↓
Explain why candidate ranked highest
```

## Strong differentiators

- 3D patient-specific anatomy.
- 3D morphology.
- Interactive implant candidate visualization.
- Explainable fit analysis.
- Confidence/quality warning.
- External validation results.

---

# 33. Stretch Features

Only implement after the core pipeline works.

## Stretch 1

Virtual implant alignment optimization.

## Stretch 2

Detailed 3D coverage/overhang analysis.

## Stretch 3

fastMRI self-supervised pretraining.

## Stretch 4

Longitudinal knee tracking.

```text
2024 scan
    ↓
2025 scan
    ↓
2026 scan
    ↓
Change over time
```

## Stretch 5

CT support.

---

# 34. Long-Term Research Vision

```text
                   PATIENT SCAN
                        |
                        v
                 3D AI ANALYSIS
                        |
                        v
              PATIENT 3D ANATOMY
                        |
          +-------------+-------------+
          |                           |
          v                           v
      OA MORPHOLOGY              BONE MORPHOLOGY
          |                           |
          +-------------+-------------+
                        |
                        v
               IMPLANT PLANNING
                        |
                        v
              VIRTUAL FITTING
                        |
                        v
              EXPLAINABLE RESULTS
                        |
                        v
               CLINICIAN REVIEW
```

Future research may add:

- Longitudinal change tracking.
- CT integration.
- Post-operative comparison.
- More advanced 3D implant alignment.
- Additional clinical variables.

These are not required for the first release.

---

# 35. Software Architecture

```text
                    FRONTEND
                       |
                       v
                BACKEND API
                       |
             +---------+---------+
             |                   |
             v                   v
       IMAGE SERVICE        MODEL SERVICE
                                 |
                +----------------+----------------+
                |                |                |
                v                v                v
             FEMUR            TIBIA           MENISCUS
              MODEL            MODEL            MODEL
                |                |                |
                +----------------+----------------+
                                 |
                                 v
                         3D RECONSTRUCTION
                                 |
                                 v
                       MEASUREMENT ENGINE
                                 |
                +----------------+----------------+
                |                                 |
                v                                 v
           OA ANALYSIS                     IMPLANT ENGINE
                |                                 |
                |                         Implant database
                |                                 |
                +----------------+----------------+
                                 |
                                 v
                          RESULT SERVICE
                                 |
                     +-----------+-----------+
                     |                       |
                     v                       v
                PATIENT UI              RESEARCH UI
```

---

# 36. Suggested Technology Stack

## Medical imaging / ML

- Python.
- PyTorch.
- MONAI.
- NumPy.
- SciPy.
- scikit-learn.

## Image processing

- SimpleITK.
- NiBabel where applicable.
- OpenCV for compatible 2D operations.

## 3D

Potential options:

- VTK.
- PyVista.
- Open3D.
- Three.js on the frontend.
- React Three Fiber if appropriate.

Choose only what is necessary; do not add a large framework stack without a reason.

## Backend

- FastAPI.

## Frontend

- React/Next.js.

## Experiment tracking

- MLflow or equivalent.

---

# 37. Repository Structure

```text
knee-ai/
|
├── data/
│   ├── raw/
│   ├── processed/
│   ├── metadata/
│   └── splits/
|
├── models/
│   ├── bone_segmentation/
│   ├── meniscus_segmentation/
│   ├── oa_analysis/
│   ├── morphology/
│   └── implant_matching/
|
├── preprocessing/
│   ├── dicom/
│   ├── nifti/
│   ├── normalization/
│   └── resampling/
|
├── segmentation/
|
├── reconstruction/
│   ├── surfaces/
│   ├── meshes/
│   └── quality/
|
├── measurement/
│   ├── femur/
│   ├── tibia/
│   └── meniscus/
|
├── morphology/
|
├── research/
│   ├── oa/
│   ├── statistics/
│   └── validation/
|
├── implant/
│   ├── database/
│   ├── matching/
│   ├── fitting/
│   └── scoring/
|
├── api/
|
├── frontend/
|
├── experiments/
|
├── configs/
|
└── docs/
```

---

# 38. Data and Experiment Rules

1. Never mix patient IDs across train/test.
2. Never tune using the external test set.
3. Preserve image spacing.
4. Keep raw data unchanged.
5. Never fabricate missing clinical information.
6. Track dataset source for every sample.
7. Track licensing restrictions.
8. Record preprocessing.
9. Version models.
10. Version implant specifications.
11. Keep an audit trail for generated measurements.
12. Record which model produced every patient result.
13. Keep experimental and production/demo data separate.

---

# 39. Clinical Safety

The system must not claim:

- "You have OA."
- "You need surgery."
- "Use this implant."
- "This implant will definitely fit."
- "This result replaces a surgeon/radiologist."

Preferred wording:

- "OA-related analysis."
- "Measured anatomical feature."
- "Candidate implant."
- "Geometric match."
- "For clinical/research review."
- "Manual review recommended."

---

# 40. Major Risks

## Risk 1 — Meniscus model does not transfer well

Solution:

- Validate on external cases.
- Review a stratified sample.
- Fine-tune if appropriate.
- Do not use unvalidated predictions for final OA research.

## Risk 2 — 3D reconstruction is unreliable

Solution:

- Keep the 2D/measurement pipeline working.
- Validate surface quality.
- Make 3D visualization an additional layer rather than a single point of failure.

## Risk 3 — Implant specifications are incomplete

Solution:

- Start with a limited, clearly sourced implant database.
- Clearly label unsupported implant families.
- Never invent specifications.

## Risk 4 — Dataset leakage

Solution:

- Patient-level IDs.
- Master overlap table.
- Independent external testing.

## Risk 5 — Overclaiming medical performance

Solution:

- Use clinical decision-support language.
- Report limitations.
- Keep human review in the workflow.

## Risk 6 — Too much scope

Solution:

Priority order:

```text
PS requirements
      ↓
Reliable measurements
      ↓
OA research
      ↓
3D visualization
      ↓
Implant matching
      ↓
3D implant fitting
      ↓
Advanced research
```

---

# 41. Success Criteria

## Core PS success

The system can:

- Find femur.
- Find tibia.
- Find medial meniscus.
- Measure meniscus thickness.
- Analyze OA-related measurements.
- Consider age/sex.
- Measure femoral/tibial anatomy.
- Rank implant candidates.
- Display results.

## Research success

The project can:

- Report segmentation metrics.
- Report measurement error.
- Perform patient-level validation.
- Perform external validation.
- Analyze meniscus/OA relationships.
- Clearly report limitations.

## Differentiation success

The project can demonstrate:

- Patient-specific 3D anatomy.
- 3D morphology.
- Explainable implant matching.
- Interactive implant candidates.
- Fit analysis where supported.
- Confidence/quality warnings.

---

# 42. Final Product Positioning

## Do not pitch it as:

> "We invented AI knee segmentation."

or:

> "We invented AI implant sizing."

Those capabilities already exist in research and commercial systems.

## Pitch it as:

> **An explainable, patient-specific knee intelligence and planning platform that fulfills automated anatomical measurement and implant-sizing requirements while extending them into 3D morphology, virtual implant fitting, research analytics, and cross-dataset validation.**

---

# 43. Final One-Line Flow

```text
SCAN → SEGMENT → RECONSTRUCT → MEASURE → ANALYZE OA → UNDERSTAND SHAPE → VIRTUALLY TEST IMPLANTS → EXPLAIN → REVIEW
```

---

# 44. Final Project Principle

The original PS is the **foundation**, not the ceiling.

We will:

```text
                 ORIGINAL PS
                     |
        +------------+------------+
        |            |            |
     Meniscus       OA         Implant
     Analysis     Analysis      Sizing
        |            |            |
        +------------+------------+
                     |
                BASE SYSTEM
                     |
        +------------+-------------+
        |            |             |
       3D        Explainability  Validation
        |            |             |
        +------------+-------------+
                     |
               EXPANDED SYSTEM
                     |
        +------------+-------------+
        |            |             |
     3D Shape    Virtual Implant  Research
     Analysis       Fitting       Platform
                     |
                     v
        PATIENT-SPECIFIC KNEE
             PLANNING SYSTEM
```

The goal is not to add features just to make the project look bigger.

Every major addition must have:

**Data → Method → Validation → Explainable Result**

That is the standard we should use throughout development.
