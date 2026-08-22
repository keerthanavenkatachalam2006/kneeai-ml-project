# PATIENT-SPECIFIC KNEE INTELLIGENCE, SURGICAL PLANNING & ROBOTIC ASSISTANCE PLATFORM

## Master Product Requirements Document (PRD)
## Technical Requirements Document (TRD)
## AI/ML Specification
## Dataset & Data Engineering Specification
## 3D Simulation Specification
## Future Robotics Specification

**Document version:** 2.0 — Expanded Master Specification  
**Status:** Engineering/Product Blueprint  
**Primary domain:** Knee orthopaedics / Total Knee Arthroplasty (TKA)  
**Initial geography:** India, with South India as an initial target market  
**Primary users:** Orthopaedic surgeons, robotic surgery teams, radiologists, biomedical engineers, clinical researchers  
**Development compute:** Google Colab + CUDA-compatible NVIDIA GPU runtime  
**Optional data platform:** Databricks  
**Cloud dependency:** No AWS dependency for the initial product  
**AI framework:** PyTorch + MONAI  
**Primary segmentation baseline:** nnU-Net  
**Future robotics:** ROS 2 + MoveIt 2 + dedicated safety/control architecture  

---

# PART I — EXECUTIVE PRODUCT DEFINITION

# 1. What We Are Building

We are building a patient-specific knee intelligence and surgical planning platform that converts medical imaging into a clinically useful digital representation of a patient's knee.

The system is intended to progressively perform:

1. Imaging ingestion
2. DICOM validation and preprocessing
3. Automatic anatomical segmentation
4. Femur/tibia/cartilage/meniscus identification
5. Anatomical landmark detection
6. 3D anatomical reconstruction
7. OA/pathology analysis
8. Quantitative measurements
9. Meniscus morphology analysis
10. Patient-specific anatomical modelling
11. Implant candidate matching
12. Implant size recommendation
13. Alignment planning
14. Bone-resection planning
15. 3D surgical simulation
16. Surgeon review and editing
17. Surgeon approval
18. Future intraoperative navigation
19. Future constrained robotic execution

The long-term concept is:

```text
PATIENT IMAGING
      ↓
AI ANATOMICAL UNDERSTANDING
      ↓
PATIENT-SPECIFIC 3D KNEE
      ↓
CLINICAL MEASUREMENTS
      ↓
PATHOLOGY / OA ANALYSIS
      ↓
IMPLANT INTELLIGENCE
      ↓
SURGICAL PLAN
      ↓
3D SIMULATION
      ↓
SURGEON REVIEW
      ↓
SURGEON APPROVAL
      ↓
INTRAOPERATIVE NAVIGATION
      ↓
SAFETY CONTROLLER
      ↓
ROBOTIC EXECUTION
```

The first commercial product should **not** attempt autonomous human surgery.

The product should first establish value through imaging intelligence, planning, simulation and surgeon workflow.

---

# 2. Product Philosophy

The system must be designed around five principles.

## 2.1 Patient-specific

The system should not simply output a generic "normal knee."

It should reconstruct and reason about:

- this patient's femur
- this patient's tibia
- this patient's cartilage
- this patient's menisci
- this patient's alignment
- this patient's anatomical measurements
- this patient's implant fit

---

## 2.2 Explainable

Every important AI recommendation should have measurable evidence.

For example:

```text
Femoral Implant Candidate: Size M

AP measurement: XX mm
ML measurement: YY mm
Predicted coverage: ZZ%
Predicted overhang: XX mm
Fit score: XX
Alternative Size L: YY
```

The system should show the surgeon **why** a recommendation was generated.

---

## 2.3 Surgeon-controlled

The platform is an assistant, not an autonomous doctor.

The surgeon must be able to:

- inspect
- modify
- reject
- approve
- compare
- override

AI recommendations.

---

## 2.4 Safety-first

The robotics architecture must never allow an unrestricted neural network or LLM to directly command a surgical actuator.

The architecture is:

```text
AI
 ↓
Recommendation
 ↓
Deterministic Planning
 ↓
Surgeon Approval
 ↓
Safety Validation
 ↓
Motion Planning
 ↓
Robot
```

---

## 2.5 Progressive commercialization

The product should be built in stages:

### Product 1
AI planning software

### Product 2
AI + intraoperative navigation

### Product 3
AI + navigation + robotic assistance

This reduces technical, clinical and regulatory risk.

---

# 3. Product Positioning

The product should not initially position itself as:

> "An AI that performs knee replacement."

Instead:

> **A patient-specific surgical intelligence platform that converts medical imaging into an explainable 3D surgical plan and can eventually provide constrained robotic assistance.**

The robotic component becomes the final execution layer.

---

# PART II — PROBLEM & MARKET CONTEXT

# 4. Clinical Problem

Total knee arthroplasty requires decisions involving:

- patient anatomy
- deformity
- alignment
- bone geometry
- implant dimensions
- component orientation
- bone resections
- soft tissue balance
- implant coverage
- postoperative alignment

Traditional workflows rely heavily on:

- radiographs
- CT/MRI when clinically indicated
- manual measurements
- surgeon experience
- templating
- intraoperative judgement

Robotic and navigation systems can improve planning and execution, but a complete patient-specific intelligence layer can still be developed around:

- automated anatomical understanding
- quantitative measurements
- 3D modelling
- meniscus/cartilage analysis
- implant matching
- simulation
- explainable recommendations
- data-driven planning optimization

---

# 5. Existing System Classes We Studied

The product concept was influenced by studying existing systems including:

## 5.1 Stryker Mako

Conceptual capabilities relevant to us:

- patient-specific planning
- 3D anatomical modelling
- implant planning
- intraoperative navigation
- robotic assistance
- surgeon-controlled execution

Our architecture should learn from the **workflow**, not blindly copy proprietary implementation.

---

## 5.2 Think Surgical / related robotic knee systems

Relevant concepts:

- CT-based planning
- implant planning
- robotic assistance
- constrained execution
- surgeon interaction

Again, these are reference points for system architecture rather than implementation specifications.

---

## 5.3 IB Lab KOALA CE / related AI-assisted knee analysis

Relevant concept:

- AI-assisted medical image analysis
- anatomical/pathological assessment
- automated measurements

This reinforces the software-first entry point.

---

## 5.4 Peek Med / AI imaging approach

Relevant concept:

- medical imaging intelligence
- computer vision
- clinical workflow support

The key takeaway is that the first product can deliver value **before** robotics.

---

# PART III — PRODUCT REQUIREMENTS

# 6. User Personas

## 6.1 Orthopaedic Surgeon

Needs:

- quick case understanding
- measurements
- 3D anatomy
- implant options
- plan comparison
- control over final plan

---

## 6.2 Radiologist

Needs:

- imaging quality
- segmentation visualization
- pathology support
- measurements
- report generation

---

## 6.3 Robotic Surgery Team

Needs:

- patient registration
- planning transfer
- tracking
- safety boundaries
- intraoperative plan verification

---

## 6.4 Biomedical Engineer

Needs:

- 3D models
- implant geometry
- simulation
- biomechanical models
- data export

---

## 6.5 Clinical Researcher

Needs:

- dataset management
- experiment tracking
- anonymized data
- model comparison
- evaluation metrics

---

# 7. Product Modules

The platform consists of:

```text
1. Case Management
2. DICOM Imaging
3. AI Segmentation
4. Landmark Detection
5. 3D Reconstruction
6. Clinical Intelligence
7. Meniscus Intelligence
8. Implant Intelligence
9. Surgical Planning
10. 3D Simulation
11. Surgeon Review
12. Reporting
13. Audit/Model Registry
14. Future Navigation
15. Future Robotics
```

---

# 8. Module 1 — Case Management

Features:

- create patient case
- de-identified patient identifier
- upload study
- assign study type
- track processing status
- view model version
- view planning version
- maintain audit history

Example state:

```text
UPLOADED
   ↓
VALIDATING
   ↓
PREPROCESSING
   ↓
AI PROCESSING
   ↓
3D RECONSTRUCTION
   ↓
PLANNING
   ↓
SURGEON REVIEW
   ↓
APPROVED
```

---

# 9. Module 2 — Medical Imaging

Supported modalities:

### MRI
Primary use:

- soft tissue
- cartilage
- meniscus
- pathology
- anatomy

### CT
Primary use:

- bone
- 3D geometry
- alignment
- implant planning

### X-ray
Primary use:

- alignment
- 2D measurements
- implant templating
- longitudinal clinical workflow

---

# 10. DICOM Requirements

The system should support:

- DICOM import
- series detection
- modality detection
- study validation
- orientation normalization
- metadata extraction
- anonymization
- DICOM-to-NIfTI conversion where appropriate
- preservation of source data

Libraries:

- pydicom
- SimpleITK
- ITK
- Orthanc
- OHIF

---

# 11. Module 3 — AI Segmentation

The first major AI module is automatic segmentation.

Target structures:

### Bone
- femur
- tibia
- patella

### Cartilage
- femoral cartilage
- medial tibial cartilage
- lateral tibial cartilage
- patellar cartilage where supported

### Soft tissue
- medial meniscus
- lateral meniscus

### Future
- ACL
- PCL
- collateral ligaments
- additional structures

---

# 12. Segmentation Pipeline

```text
DICOM
 ↓
Preprocessing
 ↓
ROI detection
 ↓
3D volume
 ↓
Segmentation model
 ↓
Post-processing
 ↓
Mask validation
 ↓
3D surface extraction
```

Primary baseline:

**nnU-Net**

Advanced experiments:

- SegResNet
- UNETR
- Swin UNETR

Framework:

**MONAI + PyTorch**

---

# 13. Segmentation Metrics

For each structure:

### Dice Similarity Coefficient

Measures overlap.

### IoU

Measures intersection/union.

### Hausdorff Distance

Measures worst-case boundary discrepancy.

### Average Surface Distance

Measures average boundary error.

### Clinical measurement error

The final test should not stop at Dice.

We must determine:

> Does segmentation error materially affect clinical measurements?

---

# 14. Module 4 — Anatomical Landmark Detection

Landmarks may include:

- femoral epicondyles
- tibial plateau landmarks
- mechanical axis points
- anatomical axis points
- femoral head reference where available
- tibial center
- intercondylar region
- other clinically relevant points

Output:

```text
Landmark
X
Y
Z
Confidence
```

Evaluation:

**Mean error in millimetres.**

---

# 15. Module 5 — 3D Reconstruction

The segmentation mask becomes a 3D model.

Pipeline:

```text
Voxel mask
 ↓
Marching Cubes
 ↓
Surface
 ↓
Mesh cleaning
 ↓
Smoothing where appropriate
 ↓
Topology validation
 ↓
3D model
```

Tools:

- VTK
- Open3D
- PyVista

Output:

- femur mesh
- tibia mesh
- cartilage mesh
- meniscus mesh

---

# 16. Shape Modelling

ShapeMedKnee and related datasets should support research into:

- statistical shape models
- neural shape models
- patient-specific reconstruction
- anatomical variation
- population-level modelling

The system should learn:

```text
Population
   ↓
Shape distribution
   ↓
Patient-specific anatomy
```

This is useful for reconstruction and quality control.

---

# 17. Module 6 — OA / Clinical Intelligence

Inputs:

- MRI
- X-ray
- anatomical measurements
- cartilage information
- meniscus morphology
- structured clinical metadata

Potential outputs:

- OA-related measurements
- cartilage status
- meniscus morphology
- supported pathology classification
- severity indicators

Potential ML metrics:

- AUROC
- AUPRC
- sensitivity
- specificity
- calibration

The prototype must not claim diagnosis unless clinically validated.

---

# 18. Module 7 — Meniscus Intelligence

This is one of the important additions identified during dataset analysis.

Pipeline:

```text
MRI
 ↓
Meniscus segmentation
 ↓
3D reconstruction
 ↓
Morphological analysis
 ↓
Measurements
```

Potential outputs:

- medial meniscus volume
- lateral meniscus volume
- thickness
- extrusion
- coverage
- morphology

The system should distinguish:

### Segmentation

Where is the meniscus?

### Quantification

How large/thick/voluminous is it?

### Pathology classification

Is there evidence of a supported abnormality?

These are separate tasks.

---

# 19. Meniscus Data Strategy

Core datasets:

- SKM-TEA
- OAI tissue segmentations
- KneeXNet
- Open Knee(s)
- relevant OAI-derived meniscus studies

Important:

Open Knee(s) is excellent for anatomical/biomechanical context but contains a small number of cadaveric specimens.

It should not be treated as a large general-purpose ML training dataset.

---

# 20. Module 8 — Implant Intelligence

This is one of the most commercially important modules.

The system must eventually answer:

> Which authorized implant configuration best fits this patient's anatomy?

Inputs:

- femur geometry
- tibia geometry
- anatomical landmarks
- measurements
- implant geometry
- surgeon preferences
- validated planning constraints

Outputs:

- candidate implant
- femoral size
- tibial size
- fit score
- coverage
- overhang
- undercoverage
- alignment compatibility

---

# 21. Implant Database

The implant database should represent:

```text
Manufacturer
Model
Component
Component type
Size
Geometry
Dimensions
Allowed configurations
Reference coordinate system
Approved-use metadata
```

Potential component types:

- femoral component
- tibial component
- polyethylene insert
- patellar component

Only authorized implant geometry should be used for commercial deployment.

---

# 22. Implant Matching — Initial Approach

Do NOT start with a black-box ML model.

Start with deterministic geometric matching:

```text
Patient geometry
       ↓
Candidate implant geometry
       ↓
Registration
       ↓
Coverage calculation
       ↓
Overhang calculation
       ↓
Undercoverage calculation
       ↓
Alignment compatibility
       ↓
Fit score
       ↓
Rank candidates
```

This creates an explainable baseline.

---

# 23. Future Implant ML Model

Once hospital data becomes available:

Input:

```text
Pre-op imaging
+
patient geometry
+
clinical measurements
```

Output:

```text
femoral size
tibial size
implant family
fit probability
mismatch risk
```

Training labels should ideally include:

- actual implant selected
- actual component size
- actual position
- postoperative validation

---

# 24. Critical Implant Dataset Gap

Public anatomy datasets are not sufficient to fully train a commercial implant-selection model.

The high-value dataset we need is:

```text
Pre-op imaging
+
3D anatomy
+
surgeon plan
+
actual implant
+
actual implant size
+
implant position
+
post-op imaging
```

This should eventually come from hospital partnerships.

---

# 25. Module 9 — Surgical Planning

Inputs:

- patient-specific 3D model
- landmarks
- implant candidates
- surgeon constraints
- anatomical measurements

Outputs:

- component selection
- alignment
- orientation
- resection parameters
- implant position
- surgical plan

The plan must remain editable.

---

# 26. Surgical Planning Workflow

```text
Patient imaging
      ↓
3D anatomy
      ↓
Measurements
      ↓
Candidate implant
      ↓
Initial AI recommendation
      ↓
Geometric optimization
      ↓
Surgical plan
      ↓
3D simulation
      ↓
Surgeon edits
      ↓
Surgeon approval
      ↓
Locked plan
```

---

# 27. Plan Comparison

The surgeon should be able to compare:

```text
Plan A
Plan B
Plan C
```

Comparison:

- implant size
- alignment
- orientation
- resection
- coverage
- overhang
- predicted metrics

The system should not hide alternatives.

---

# 28. Module 10 — 3D Surgical Simulation

Version 1:

- 3D knee
- implant placement
- bone-cut planes
- alignment axes
- component orientation
- collision/intersection detection
- before/after visualization

Technology:

- Three.js
- React Three Fiber
- VTK
- Open3D

---

# 29. Advanced Biomechanical Simulation

Future:

- finite element analysis
- joint mechanics
- soft-tissue behaviour
- contact pressure
- range of motion
- kinematics
- kinetics

Potential technology:

- SOFA Framework
- Open Knee(s)
- Natural Knee Data

This should be treated as a research module until validated.

---

# 30. Module 11 — Surgeon Dashboard

## Main layout

```text
+------------------------------------------------+
| Patient / Case                    AI Status     |
+------------------------------------------------+
|                                                |
|       2D DICOM Viewer        3D Knee           |
|                                                |
|                                                |
+----------------------+-------------------------+
| AI Findings          | Measurements            |
|                      |                         |
+----------------------+-------------------------+
| Implant Candidates   | Surgical Plan           |
|                      |                         |
+----------------------+-------------------------+
| Simulation                                      |
+------------------------------------------------+
| [EDIT PLAN]             [APPROVE PLAN]          |
+------------------------------------------------+
```

---

# 31. AI Findings UI

Show:

- structure
- result
- confidence
- supporting visualization
- model version

Example:

```text
Medial Meniscus
Segmentation: Complete
Confidence: XX%
Volume: XX
Extrusion: XX mm
```

---

# 32. Module 12 — Clinical Report

Report should include:

- study metadata
- anatomical measurements
- AI findings
- segmentation summary
- implant candidates
- selected plan
- surgeon modifications
- final approval
- model versions
- timestamp

The report must distinguish:

**AI-generated**

from

**surgeon-confirmed.**

---

# PART IV — AI / ML ARCHITECTURE

# 33. AI System Architecture

```text
                    IMAGE
                      |
                      v
             PREPROCESSING
                      |
                      v
            SEGMENTATION MODEL
                      |
                      v
              3D ANATOMY
                      |
          +-----------+-----------+
          |                       |
          v                       v
   LANDMARK MODEL          PATHOLOGY MODEL
          |                       |
          +-----------+-----------+
                      |
                      v
              MEASUREMENT ENGINE
                      |
                      v
             IMPLANT ENGINE
                      |
                      v
             PLANNING ENGINE
                      |
                      v
               SIMULATION
```

---

# 34. Why We Should Not Use One Giant Model

The product should use a modular architecture.

Reasons:

- easier validation
- easier debugging
- easier clinical interpretation
- easier model replacement
- different datasets support different tasks
- safety isolation
- better error localization

Example:

```text
Segmentation model fails
```

should not mean:

```text
Entire system is impossible to debug.
```

---

# 35. Model A — Segmentation

Input:

- 3D MRI/CT

Output:

- voxel masks

Baseline:

- nnU-Net

Training:

- supervised segmentation
- augmentation
- patch-based 3D training

---

# 36. Model B — Landmark Detection

Input:

- segmentation / image

Output:

- 3D landmark coordinates

Potential architectures:

- heatmap regression
- keypoint networks
- geometry-assisted models

---

# 37. Model C — Pathology Classification

Input:

- MRI/X-ray
- optionally derived measurements

Output:

- class probabilities

Do not use the classifier to directly make a surgical decision.

---

# 38. Model D — Meniscus

Separate:

```text
Segmentation
+
Morphology
+
Pathology
```

This makes the pipeline auditable.

---

# 39. Model E — Implant Recommendation

Phase 1:

**Geometry-based optimizer**

Phase 2:

**ML ranking model**

Phase 3:

**Hybrid system**

```text
ML
 ↓
Candidate probabilities
 ↓
Geometric constraints
 ↓
Clinical constraints
 ↓
Final ranked options
```

---

# 40. Model F — Surgical Plan Optimization

Potential optimization variables:

- component orientation
- alignment
- resection
- implant size
- coverage

Constraints:

- anatomical
- implant-specific
- surgeon-defined
- safety-related

The optimization engine should be deterministic and inspectable.

---

# 41. LLM Architecture

The LLM is a **workflow and explanation layer**.

Use cases:

- explain AI results
- summarize a case
- draft reports
- answer questions from approved documentation
- retrieve device information
- explain why candidates differ

Architecture:

```text
Structured AI outputs
        +
Approved knowledge base
        ↓
       LLM
        ↓
Explanation / Report
```

The LLM must never:

- directly control the robot
- modify a locked plan
- invent measurements
- fabricate patient information
- override safety constraints

---

# PART V — DATASET MASTER PLAN

# 42. Dataset Strategy

The datasets fall into five categories:

```text
1. Anatomy
2. Meniscus / pathology
3. Implant planning
4. Biomechanics
5. Robotics / surgical execution
```

---

# 43. Dataset A — OAIZIB-CM / OAI-ZIB

Source:

https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM

Use:

- femur
- femoral cartilage
- tibia
- tibial cartilage
- OA
- demographics

Reported characteristics from the team's dataset notes:

- 507 3D DESS knee MRI examinations
- Siemens 3T
- high-resolution voxel spacing
- segmentation masks
- KL grade
- gender
- age
- BMI
- knee side

Use:

- segmentation
- statistical analysis
- OA modelling

License must be verified before commercial use.

---

# 44. Dataset B — ShapeMedKnee

Hugging Face:

https://huggingface.co/datasets/aagatti/ShapeMedKnee

GitHub:

https://github.com/gattia/shapemedknee

Use:

- 3D shape modelling
- femur
- cartilage
- patient-specific reconstruction
- anatomical variation

Role:

**Core 3D modelling dataset.**

---

# 45. Dataset C — Open Knee(s)

Use:

- femur
- tibia
- cartilage
- medial meniscus
- lateral meniscus
- ACL
- PCL
- ligaments
- 3D surfaces
- FE meshes
- kinematics
- kinetics
- mechanical testing

Role:

**Biomechanics + simulation + validation.**

Not primary large-scale ML training.

---

# 46. Dataset D — MedShapeNet 2.0

Website:

https://medshapenet.ikim.nrw/

GitHub:

https://github.com/GLARKI/MedShapeNet2.0

Role:

- supplementary 3D shapes
- reconstruction
- shape pretraining
- augmentation
- visualization research

Do not treat it as the primary knee-specific dataset.

---

# 47. Dataset E — OAI

Official:

https://nda.nih.gov/oai

Use:

- MRI
- X-ray
- clinical data
- demographics
- longitudinal information

This is especially important because multiple derived datasets may originate from OAI.

---

# 48. Dataset F — SKM-TEA

Stanford AIMI:

https://aimi.stanford.edu/datasets/skm-tea-knee-mri

Use:

- meniscus segmentation
- cartilage segmentation
- pathology
- multi-task MRI learning

Role:

**Major missing meniscus dataset.**

---

# 49. Dataset G — OAI Tissue Segmentations

Example access:

https://www.kaggle.com/datasets/kgaooo/oai-tissue-segmentations

Use:

- large-scale segmentation
- femur
- tibia
- cartilage
- meniscus

Important:

Verify provenance and licensing.

Do not treat longitudinal OAI visits as independent patients.

---

# 50. Dataset H — KneeXNet

GitHub:

https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D

Use:

- meniscus
- cartilage
- segmentation

Recommended initial role:

**External validation.**

---

# 51. Dataset I — MRNet

Stanford:

https://stanfordmlgroup.github.io/competitions/mrnet/

Use:

- meniscus pathology classification
- ACL pathology classification
- abnormality classification

Important:

This is not equivalent to a voxel-level meniscus segmentation dataset.

---

# 52. Dataset J — Natural Knee Data

https://digitalcommons.du.edu/natural_knee_data/

Use:

- biomechanics
- mechanical testing
- kinematics
- simulation
- validation

---

# 53. Dataset K — KISTI CT Models

Use based on published implant-sizing research:

- pre-segmented femur/tibia CT models
- patient-specific geometry
- implant matching research

Access/license must be independently verified.

---

# 54. Implant Dataset Gap

The following are needed for a real implant model:

```text
Pre-op X-ray/CT/MRI
+
3D anatomy
+
implant selected
+
implant size
+
implant position
+
post-op validation
```

Public anatomy datasets generally do not provide all of these fields together.

Therefore hospital data acquisition is a core product-development requirement.

---

# 55. 308-Patient Implant Sizing Research Dataset

A published study used:

- preoperative X-rays
- height
- weight
- femoral prosthesis size
- tibial prosthesis size

with 308 valid samples.

The dataset is a useful reference for the expected training schema.

Do not assume the raw patient data are publicly available.

---

# 56. 103-Patient Planned-vs-Actual Dataset

A published TKA study included:

- preoperative X-ray
- planning
- navigation recommendation
- actual implant size

This is useful because it shows the importance of:

```text
Prediction
vs
Actual surgical decision
```

The raw patient dataset must be separately verified.

---

# 57. Implant Mismatch Dataset Category

Research datasets should be investigated containing:

- preoperative templating
- actual implant
- manufacturer
- component family
- component size
- mismatch

This enables a better objective:

> predict the probability of sizing mismatch and recommend candidates that minimize mismatch.

---

# 58. Hospital Dataset — Ultimate Data Moat

The target dataset schema:

```text
case_id
imaging_id
modality
preoperative_image
patient_anatomy
landmarks
surgeon_plan
AI_plan
implant_family
implant_model
femoral_size
tibial_size
implant_position
bone_resection
intraoperative_tracking
navigation_data
postoperative_image
clinical_outcome
```

This should become the proprietary dataset.

---

# 59. Dataset Matrix

| Dataset | MRI | CT | X-ray | Segmentation | Meniscus | 3D | Biomechanics | Implant Labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OAIZIB-CM | Yes | No | No | Yes | Limited/not core | Yes | No | No |
| ShapeMedKnee | Derived MRI | No | No |/3D | Limited | Yes | No | No |
| OAI | Yes | Available components vary | Yes | Derived | Derived | Yes | No | No |
| SKM-TEA | Yes | No | No | Yes | Yes | Yes | No | No |
| OAI Tissue | Yes | No | No | Yes | Yes | Yes | No | No |
| KneeXNet | Yes | No | No | Yes | Yes | Yes | No | No |
| MRNet | Yes | No | No | No voxel segmentation focus | Pathology | Limited | No | No |
| Open Knee(s) | Yes | Supporting imaging | No | Yes | Yes | Yes | Yes | No |
| Natural Knee | Yes | Yes | Supporting | Anatomical | Yes | Yes | Yes | No |
| KISTI | No/CT | Yes | No | Bone | No | Yes | No | No |
| Hospital dataset | Depends | Depends | Depends | Yes | Yes | Yes | Yes | **Yes** |

---

# PART VI — DATA ENGINEERING

# 60. Data Directory

```text
data/
├── raw/
│   ├── oaizib/
│   ├── shapemedknee/
│   ├── oai/
│   ├── skm_tea/
│   ├── oai_tissue/
│   ├── kneexnet/
│   ├── mrnet/
│   ├── open_knee/
│   └── natural_knee/
│
├── interim/
│
├── processed/
│
├── manifests/
│
├── splits/
│
└── external_test/
```

---

# 61. Dataset Registry

Create:

```text
dataset_registry.csv
```

Fields:

```text
dataset_id
dataset_name
source
url
version
license
patient_count
study_count
modality
structures
labels
resolution
preprocessing
commercial_use
download_date
checksum
notes
```

---

# 62. Data Quality Checks

Every dataset should be checked for:

- corrupt files
- missing slices
- incorrect orientation
- invalid spacing
- missing labels
- empty masks
- duplicate studies
- duplicate patients
- inconsistent metadata

---

# 63. Patient-Level Leakage

This is critical.

Because OAI-derived datasets may overlap:

Never do:

```text
slice-level random split
```

Do:

```text
patient-level split
```

Example:

```text
Patient 001
  MRI 1
  MRI 2
  X-ray 1
```

All must remain in the same split.

---

# 64. Recommended Split

Where dataset size allows:

```text
70% train
15% validation
15% test
```

But for external validation:

```text
TRAIN
↓
Internal validation
↓
External dataset
```

Example:

```text
Train:
OAIZIB + selected OAI data + SKM-TEA

External test:
KneeXNet
```

Do not mix external test data into training.

---

# PART VII — COMPUTE & MLOPS

# 65. Google Colab Workflow

All initial ML work should run through Google Colab.

Typical notebook:

```text
01_dataset_loading.ipynb
02_preprocessing.ipynb
03_segmentation_training.ipynb
04_evaluation.ipynb
05_3d_reconstruction.ipynb
06_meniscus.ipynb
07_implant_experiments.ipynb
```

---

# 66. CUDA Setup

At notebook start:

```bash
nvidia-smi
```

Then:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
```

Use:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Training should prefer GPU.

Use:

- mixed precision
- gradient accumulation
- checkpointing
- patch-based 3D training

when memory is constrained.

---

# 67. Databricks

Optional.

Use Databricks when:

- dataset preprocessing becomes too large
- multiple people need shared pipelines
- large-scale inference is required
- metadata processing becomes expensive
- data quality pipelines need distributed processing

Do not introduce Databricks just because it is available.

---

# 68. MLOps

Use:

- MLflow
- DVC
- Git
- GitHub
- Docker

Track:

```text
model
dataset
code commit
hyperparameters
metrics
preprocessing version
```

---

# 69. Model Registry

Example:

```text
knee-segmentation-v0.1
knee-segmentation-v0.2
meniscus-segmentation-v0.1
landmark-model-v0.1
oa-classifier-v0.1
implant-ranker-v0.1
```

---

# PART VIII — 3D SYSTEM

# 70. 3D Representation

Each patient should have:

```text
Femur
Tibia
Patella
Femoral cartilage
Tibial cartilage
Medial meniscus
Lateral meniscus
```

Each mesh should preserve:

- coordinate system
- units
- patient orientation
- source modality
- model version

---

# 71. Coordinate System

All 3D calculations must use an explicitly defined coordinate system.

Never mix:

- image coordinates
- world coordinates
- mesh coordinates
- robot coordinates

without an explicit transformation.

---

# 72. Transform Chain

Future navigation requires:

```text
Image Coordinate
      ↓
Patient Coordinate
      ↓
Planning Coordinate
      ↓
Tracking Coordinate
      ↓
Robot Coordinate
```

Each transform must be:

- explicitly represented
- timestamped where required
- validated
- auditable

---

# PART IX — SIMULATION

# 73. Simulation Levels

## Level 1 — Geometric

- implant placement
- bone cuts
- alignment
- collision detection

## Level 2 — Kinematic

- joint movement
- range of motion
- component interference

## Level 3 — Biomechanical

- soft tissue
- forces
- contact
- FE analysis

## Level 4 — Robotic

- robot workspace
- trajectories
- tracking
- safety limits

---

# 74. Simulation Workflow

```text
Patient anatomy
      ↓
Surgical plan
      ↓
Virtual bone cuts
      ↓
Virtual implant
      ↓
Motion simulation
      ↓
Collision checks
      ↓
Safety checks
      ↓
Plan report
```

---

# PART X — ROBOTICS

# 75. Robotics Philosophy

Robotics is a future stage.

The initial product does not require a physical robot.

Build the software architecture first.

---

# 76. Robotics Data

Eventually need:

- surgical plans
- surgeon trajectories
- tracking data
- robot joint positions
- force/torque
- bone cuts
- implant positions
- intraoperative events

---

# 77. Robot Control Architecture

```text
Patient-specific AI
        ↓
Surgical planning
        ↓
Surgeon approval
        ↓
Safety validator
        ↓
Motion planner
        ↓
Robot controller
        ↓
Robot
        ↑
Position feedback
        ↑
Force feedback
```

---

# 78. ROS 2

Future robotics stack:

- ROS 2
- MoveIt 2
- robot-specific SDK
- real-time control architecture
- tracking system
- safety controller

The exact robot/hardware interface cannot be assumed before selecting or partnering with a hardware platform.

---

# 79. Safety Controller

Must independently enforce:

- workspace limits
- velocity limits
- acceleration limits
- force limits
- forbidden regions
- plan boundaries
- emergency stop
- watchdog
- surgeon override

---

# 80. Robot Must Not Receive Direct LLM Output

Forbidden architecture:

```text
LLM → Robot
```

Forbidden:

```text
Neural Network → actuator command
```

without an independently validated safety/control layer.

Correct:

```text
AI
 ↓
Structured plan
 ↓
Safety validation
 ↓
Deterministic controller
 ↓
Robot
```

---

# PART XI — SECURITY & PRIVACY

# 81. Patient Data

Requirements:

- de-identification
- access control
- encryption
- audit logs
- least privilege
- secure storage
- controlled export

---

# 82. Medical AI Security

Protect against:

- malicious DICOM
- malformed input
- model poisoning
- unauthorized model changes
- data leakage
- prompt injection into LLM workflows
- unauthorized access

---

# 83. LLM Security

The LLM must only access:

- approved documents
- approved patient structured outputs
- authorized case context

It should not freely browse sensitive patient storage.

---

# PART XII — TESTING

# 84. Unit Testing

Test:

- DICOM parsing
- transforms
- measurements
- mesh generation
- implant geometry
- fit calculations

---

# 85. ML Testing

Test:

- segmentation
- external validation
- distribution shifts
- missing data
- poor image quality
- unusual anatomy

---

# 86. Clinical Measurement Testing

For every measurement:

```text
AI measurement
vs
ground truth measurement
```

Evaluate:

- MAE
- RMSE
- confidence intervals
- inter-rater comparison

---

# 87. Implant Testing

Compare:

```text
AI recommendation
vs
surgeon recommendation
vs
actual implant
```

Metrics:

- exact size accuracy
- top-2 accuracy
- mismatch rate
- overhang
- undercoverage
- alignment deviation

---

# 88. Simulation Testing

Test:

- invalid anatomy
- impossible implant placement
- collision
- transform errors
- missing landmarks
- incorrect orientation

The system must fail safely.

---

# PART XIII — UI/UX

# 89. Design Principle

Clinical UI should be:

- minimal
- high-information
- low cognitive load
- clear
- professional
- fast

Avoid unnecessary animations.

---

# 90. Main Navigation

```text
Cases
Imaging
AI Analysis
3D Model
Planning
Simulation
Reports
Settings
```

Future:

```text
Navigation
Robotics
```

---

# 91. Case Workflow UI

```text
1. Upload
2. Validate
3. Analyze
4. Review AI
5. View 3D
6. Plan
7. Simulate
8. Edit
9. Approve
10. Export
```

---

# PART XIV — DATABASE

# 92. Core Tables

Potential schema:

```text
patients
cases
studies
series
images
models
segmentations
landmarks
measurements
pathology_results
implants
implant_components
implant_candidates
surgical_plans
simulation_runs
approvals
reports
audit_logs
model_versions
dataset_versions
```

---

# 93. Example Surgical Plan Record

```json
{
  "case_id": "...",
  "model_version": "planner-v0.1",
  "implant_family": "...",
  "femoral_size": "...",
  "tibial_size": "...",
  "alignment": {},
  "orientation": {},
  "bone_resection": {},
  "fit_metrics": {},
  "surgeon_approved": false
}
```

---

# PART XV — DEVELOPMENT PLAN

# 94. Phase 0 — Data Foundation

Deliver:

- dataset registry
- download scripts
- preprocessing
- patient-level splitting
- DICOM pipeline
- Colab setup
- CUDA verification
- experiment tracking

---

# 95. Phase 1 — Segmentation

Deliver:

- femur
- tibia
- cartilage
- meniscus

Baseline:

nnU-Net

---

# 96. Phase 2 — 3D

Deliver:

- 3D meshes
- landmarks
- measurements
- browser visualization

---

# 97. Phase 3 — Clinical Intelligence

Deliver:

- OA analysis
- meniscus morphology
- pathology classification
- structured report

---

# 98. Phase 4 — Implant Intelligence

Deliver:

- implant database
- implant geometry
- geometric matching
- size recommendation
- ranking

---

# 99. Phase 5 — Surgical Planning

Deliver:

- alignment
- component orientation
- bone cuts
- plan comparison
- surgeon editing

---

# 100. Phase 6 — Simulation

Deliver:

- virtual implant
- bone cuts
- collision
- basic kinematics
- simulation report

---

# 101. Phase 7 — Clinical Workflow

Deliver:

- case management
- DICOM
- surgeon dashboard
- audit
- reports
- model versioning

---

# 102. Phase 8 — Navigation Research

Deliver:

- patient registration
- tracking
- landmarks
- navigation
- safety boundaries

---

# 103. Phase 9 — Robotics Research

Deliver:

- ROS 2
- MoveIt 2
- simulation
- trajectory generation
- safety controller
- phantom testing

---

# 104. Phase 10 — Clinical Validation

Only after appropriate:

- regulatory planning
- risk management
- QMS
- verification
- validation
- ethical approvals
- clinical study design
- hardware validation

---

# PART XVI — REPOSITORY

# 105. Project Structure

```text
knee-intelligence/
│
├── apps/
│   ├── web/
│   └── api/
│
├── ml/
│   ├── segmentation/
│   ├── meniscus/
│   ├── landmarks/
│   ├── pathology/
│   ├── reconstruction/
│   └── implant/
│
├── imaging/
│   ├── dicom/
│   ├── preprocessing/
│   └── validation/
│
├── planning/
│   ├── geometry/
│   ├── implants/
│   ├── alignment/
│   └── resection/
│
├── simulation/
│   ├── anatomy/
│   ├── implants/
│   ├── kinematics/
│   └── biomechanics/
│
├── robotics/
│   ├── ros2/
│   ├── tracking/
│   ├── planning/
│   └── safety/
│
├── datasets/
│   ├── registry/
│   └── manifests/
│
├── notebooks/
│
├── configs/
├── tests/
├── docs/
└── README.md
```

---

# PART XVII — VERSION 1 DEFINITION OF DONE

# 106. V1 Must Demonstrate

Given a valid knee imaging study:

```text
INPUT
MRI / CT / X-ray
```

The system must:

1. Validate the study
2. Preprocess it
3. Run segmentation
4. Generate 3D anatomy
5. Detect landmarks
6. Calculate measurements
7. Analyze supported pathology/OA features
8. Analyze meniscus morphology where supported
9. Load authorized implant geometries
10. Generate candidate implant matches
11. Produce an explainable plan
12. Visualize the plan
13. Simulate the plan
14. Allow surgeon editing
15. Allow surgeon approval
16. Generate a report
17. Store model/data versions
18. Store an audit trail

---

# 107. V1 Non-Goals

V1 does NOT:

- autonomously operate on humans
- directly control a surgical robot
- replace a surgeon
- claim regulatory approval
- claim clinical efficacy without validation
- use an LLM for surgical decision-making

---

# PART XVIII — LONG-TERM PRODUCT

# 108. Final Architecture

```text
                  PATIENT
                     |
                     v
             MEDICAL IMAGING
                     |
                     v
              AI INTELLIGENCE
                     |
                     v
            DIGITAL KNEE MODEL
                     |
                     v
           CLINICAL MEASUREMENTS
                     |
                     v
            IMPLANT INTELLIGENCE
                     |
                     v
           SURGICAL PLAN ENGINE
                     |
                     v
              3D SIMULATION
                     |
                     v
             SURGEON APPROVAL
                     |
                     v
           INTRAOPERATIVE NAV
                     |
                     v
            SAFETY CONTROLLER
                     |
                     v
              ROBOTIC SYSTEM
```

---

# 109. Product Moat

The long-term moat is not simply:

> "We have an AI model."

The moat becomes:

```text
Imaging
+
Anatomical models
+
Clinical measurements
+
Implant geometry
+
Surgical plans
+
Actual implant outcomes
+
Intraoperative data
+
Postoperative outcomes
+
Simulation
+
Safety validation
```

This creates a continuously improving patient-specific surgical intelligence platform.

---

# 110. Proprietary Data Flywheel

```text
Hospital cases
      ↓
De-identified data
      ↓
Model improvement
      ↓
Better planning
      ↓
More hospital value
      ↓
More cases
      ↓
More data
      ↓
Better models
```

This is the long-term defensibility strategy.

---

# PART XIX — COMMERCIALIZATION

# 111. Product Packaging

## Tier 1 — AI Planning

Includes:

- imaging
- segmentation
- 3D
- measurements
- implant planning
- reports

Possible pricing model:

- hospital annual license
- per-case fee
- hybrid

---

## Tier 2 — Navigation

Includes:

- everything in Tier 1
- registration
- tracking
- intraoperative navigation

---

## Tier 3 — Robotic Platform

Includes:

- everything in Tier 2
- robotic hardware
- control system
- safety controller
- service/support

---

# 112. Initial Commercial Strategy

The first product should be sold as:

> **AI-assisted patient-specific knee surgical planning and simulation.**

This avoids requiring a full robotic hardware platform on day one.

The hardware can follow after:

- clinical validation
- hospital partnerships
- data acquisition
- regulatory pathway
- robotics engineering

---

# PART XX — CRITICAL ENGINEERING RULES

# 113. Rule 1 — Never Train Without a Dataset Contract

Before training any model, document:

```text
What is input?
What is output?
What is ground truth?
Who annotated it?
How many patients?
What is the license?
What is the split?
```

---

# 114. Rule 2 — Never Trust Dice Alone

A segmentation model can have excellent Dice but still produce clinically meaningful measurement errors.

Always evaluate:

```text
Segmentation
+
Surface
+
Measurement
+
Clinical relevance
```

---

# 115. Rule 3 — Never Mix OAI Patients Across Splits

OAI-derived datasets can overlap.

Maintain a global patient identity mapping where legally and technically possible.

---

# 116. Rule 4 — Never Let LLM Output Enter the Safety Loop

LLM:

```text
Explanation
```

not:

```text
Actuator command
```

---

# 117. Rule 5 — Never Let AI Directly Command the Robot

The robot must operate through:

```text
Approved Plan
+
Safety Constraints
+
Deterministic Controller
```

---

# 118. Rule 6 — Every Recommendation Must Be Traceable

For every output:

```text
Input
→ Model
→ Version
→ Measurement
→ Logic
→ Recommendation
```

must be reconstructable.

---

# 119. Rule 7 — Simulation Before Hardware

Before physical robotic execution:

```text
Digital simulation
 ↓
Software safety testing
 ↓
Physical phantom
 ↓
Cadaveric research
 ↓
Formal verification/validation
 ↓
Clinical pathway
```

---

# PART XXI — FINAL TECHNICAL STACK

# 120. Final Stack

## Frontend

- Next.js
- React
- TypeScript
- Tailwind
- shadcn/ui
- Three.js
- React Three Fiber
- OHIF

## Backend

- Python
- FastAPI
- PostgreSQL
- Redis
- background workers

## Imaging

- DICOM
- pydicom
- SimpleITK
- ITK
- VTK
- Orthanc

## AI

- PyTorch
- MONAI
- nnU-Net
- SegResNet
- UNETR
- Swin UNETR
- scikit-learn

## 3D

- VTK
- Open3D
- PyVista
- Three.js

## Simulation

- SOFA
- Open Knee(s)
- custom geometry engine

## Robotics — future

- ROS 2
- MoveIt 2
- robot SDK
- tracking
- force/torque sensing

## MLOps

- GitHub
- Git
- MLflow
- DVC
- Docker

## Compute

- Google Colab
- NVIDIA CUDA
- PyTorch CUDA
- optional Databricks

## Cloud

- No AWS dependency for V1

---

# PART XXII — FINAL BUILD ORDER

The engineering team should follow this exact order.

```text
STEP 1
Dataset registry
        ↓
STEP 2
Download + validate datasets
        ↓
STEP 3
DICOM preprocessing
        ↓
STEP 4
Patient-level splits
        ↓
STEP 5
nnU-Net femur/tibia baseline
        ↓
STEP 6
Cartilage segmentation
        ↓
STEP 7
Meniscus segmentation
        ↓
STEP 8
External validation
        ↓
STEP 9
3D reconstruction
        ↓
STEP 10
Landmark detection
        ↓
STEP 11
Measurement engine
        ↓
STEP 12
OA/pathology module
        ↓
STEP 13
Implant geometry database
        ↓
STEP 14
Geometric implant matching
        ↓
STEP 15
Surgical planning
        ↓
STEP 16
3D simulation
        ↓
STEP 17
Surgeon dashboard
        ↓
STEP 18
Clinical data partnership
        ↓
STEP 19
Real implant-label model
        ↓
STEP 20
Navigation research
        ↓
STEP 21
ROS 2 simulation
        ↓
STEP 22
Physical robotics research
```

---

# PART XXIII — WHAT WE STILL NEED TO ACQUIRE

The current public datasets give us a strong foundation, but there are three major missing categories.

## Missing Category 1 — Large, high-quality meniscus segmentation

Partially addressed by:

- SKM-TEA
- OAI Tissue
- KneeXNet
- Open Knee(s)

---

## Missing Category 2 — Real implant labels

Need:

```text
Imaging
+
planned implant
+
actual implant
+
size
+
position
```

This should be acquired through:

- hospital partnership
- clinical research collaboration
- authorized retrospective dataset

---

## Missing Category 3 — Intraoperative robotics data

Need eventually:

```text
Patient anatomy
+
approved plan
+
tracking
+
robot trajectory
+
force
+
bone cut
+
implant position
+
post-op validation
```

This cannot realistically be replaced by downloading generic ML datasets.

It requires:

- robotics research
- hardware
- simulation
- phantom testing
- cadaveric work
- institutional collaboration

---

# PART XXIV — FINAL PRODUCT DEFINITION

The product we are actually building is NOT merely:

> "An ML model trained on knee MRI."

It is a complete system:

```text
                    ┌─────────────────────┐
                    │     MEDICAL DATA    │
                    │ MRI / CT / X-RAY    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  IMAGING PIPELINE   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │      AI MODELS      │
                    │ Segmentation        │
                    │ Landmarks           │
                    │ Pathology           │
                    │ Meniscus             │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │    DIGITAL KNEE     │
                    │       3D MODEL      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ CLINICAL MEASURES   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ IMPLANT INTELLIGENCE│
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ SURGICAL PLANNING   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ 3D SIMULATION       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ SURGEON APPROVAL    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ FUTURE NAVIGATION   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ SAFETY CONTROLLER   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ FUTURE ROBOTICS     │
                    └─────────────────────┘
```

---

# 125. Success Definition

The project is successful when the team has a reproducible pipeline where:

```text
A knee imaging study
        ↓
becomes
        ↓
a validated patient-specific 3D model
        ↓
with quantitative measurements
        ↓
with explainable AI findings
        ↓
with implant candidates
        ↓
with a simulated surgical plan
        ↓
that a surgeon can inspect, modify and approve.
```

The future robotics program then builds on this validated digital foundation.

---

# 126. Important Disclaimer

This document is an engineering/product development blueprint, not a clinical protocol or regulatory approval.

Any transition from research software to clinical use requires appropriate:

- medical-device regulatory assessment
- quality management
- risk management
- clinical validation
- cybersecurity
- human factors
- software verification/validation
- hardware verification/validation
- institutional approvals
- ethical approvals
- regulatory approval

The exact requirements must be established with qualified clinical, regulatory, biomedical and legal professionals.

---

# END OF MASTER PRD / TRD

Version: 2.0
