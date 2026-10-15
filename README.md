# KneeAI - Knee OA Analysis & Implant Planning Platform

AI-powered knee osteoarthritis analysis and implant planning system that takes MRI segmentation data through a complete pipeline: **3D Reconstruction → Anatomical Measurements → OA Assessment → Implant Matching → Clinical Explainability**.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React + Three.js Frontend              │
│  ┌───────────┬──────────────┬────────────┬───────────┐  │
│  │ 3D Viewer │ Measurements │  Implant   │    OA     │  │
│  │           │    Panel     │  Matching  │ Research  │  │
│  └───────────┴──────────────┴────────────┴───────────┘  │
└────────────────────────────┬────────────────────────────┘
                             │ REST API
┌────────────────────────────┴────────────────────────────┐
│                    FastAPI Backend                        │
│  ┌────────────┬──────────────┬───────────┬───────────┐  │
│  │   Data     │ Measurement  │  Implant  │    OA     │  │
│  │  Loader    │   Engine     │  Matcher  │  Engine   │  │
│  └────────────┴──────────────┴───────────┴───────────┘  │
│  ┌──────────────────────────────────────────────────┐   │
│  │           3D Mesh Reconstructor (GLB)             │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Features

- **3D Reconstruction**: Converts segmentation masks to interactive 3D meshes (GLB format)
- **Anatomical Measurements**: Physical mm measurements of bones and cartilage
- **Implant Matching**: Ranks candidates from 3 manufacturers (Stryker, Zimmer, S&N) with explainability
- **OA Assessment**: KL grade estimation with population reference comparisons
- **Interactive 3D Viewer**: Rotate, zoom, and inspect knee structures in browser

## Dataset

Using **OAIZIB-CM** (507 knee MRIs, 5-structure segmentation masks):
- Femur, Tibia, Femoral Cartilage, Medial/Lateral Tibial Cartilage
- Clinical metadata: KL Grade (0-4), Age (45-79), Gender, BMI
- Pre-defined train/test split (404/103), case 454 excluded (corrupted)

## Quick Start

### Windows: use `morp.ps1`, not `make`

`make` is not present on Windows, and the `Makefile` also assumes a Unix virtualenv layout
(`.venv/bin`), so its targets cannot work there even with make installed. The PowerShell
equivalent lives in the repository root:

```powershell
.\morp.ps1 help            # list every target
.\morp.ps1 install-all     # python package + both front ends
.\morp.ps1 api             # FastAPI on 127.0.0.1:8000
.\morp.ps1 web             # planning workspace on localhost:3000
.\morp.ps1 site            # showcase site on localhost:3001
```

The leading `.\` is required: PowerShell does not search the current directory for
commands. Each target prints the command it runs, so any of it can be done by hand. If
PowerShell blocks the script, allow it for the session with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### Backend
```bash
cd backend
pip install -r requirements.txt

# Start API server
cd ..
python scripts/start_backend.py
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### Test Pipeline
```bash
python scripts/test_pipeline.py
```

### Validate Dataset
```bash
python scripts/validate_dataset.py
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /dataset/summary` | Dataset overview and statistics |
| `GET /dataset/cases` | Browse cases with filtering |
| `GET /case/{id}/analyze` | Full analysis pipeline |
| `GET /case/{id}/mesh` | 3D GLB mesh download |
| `GET /case/{id}/measurements` | Anatomical measurements |
| `GET /case/{id}/implants` | Implant matching results |
| `GET /implants/database` | Full implant catalog |
| `GET /research/correlations` | OA correlation analysis |

## Project Structure

```
KneeAI/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration & paths
│   │   ├── data_loader.py       # OAIZIB-CM data pipeline
│   │   ├── reconstruction.py    # 3D mesh generation
│   │   ├── measurements.py      # Anatomical measurement engine
│   │   ├── implant_database.py  # Implant specs & matching
│   │   └── oa_research.py       # OA statistical analysis
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main application
│   │   ├── api.ts               # Backend API client
│   │   └── components/
│   │       ├── KneeViewer3D.tsx # Three.js 3D viewer
│   │       ├── PatientProfile.tsx
│   │       ├── MeasurementsPanel.tsx
│   │       ├── ImplantMatching.tsx
│   │       ├── OAAssessment.tsx
│   │       └── CaseSelector.tsx
│   └── package.json
├── data/
│   ├── raw/OAIZIB-CM/           # Source dataset (ZIPs)
│   ├── processed/               # Extracted NIfTI + meshes
│   └── metadata/                # Dataset registry
├── scripts/
│   ├── test_pipeline.py         # End-to-end pipeline test
│   ├── validate_dataset.py      # Dataset integrity validation
│   └── start_backend.py         # Backend launcher
└── README.md
```

## Tech Stack

- **Backend**: Python 3.11, FastAPI, nibabel, scikit-image, trimesh, scipy
- **Frontend**: React 18, TypeScript, Three.js (React Three Fiber), TailwindCSS
- **3D**: Marching cubes → Trimesh → GLB export → Three.js rendering
- **Data**: NIfTI format, ZIP-based lazy extraction

## Docker & AWS EC2 Deployment

### Quick Start (Docker)

```bash
# Build and start all services locally
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# Backend → http://localhost:8000
# Swagger → http://localhost:8000/docs
# Frontend → http://localhost:3000
```

### AWS EC2 Deployment

See full guide: [`deploy/AWS_EC2_DEPLOYMENT.md`](deploy/AWS_EC2_DEPLOYMENT.md)

```bash
# 1. Launch t3.xlarge EC2 (Ubuntu 22.04), open ports 8000 + 3000
# 2. SSH in and run the bootstrap:
curl -sSL https://raw.githubusercontent.com/shamsundarsk/MR/main/deploy/ec2_setup.sh | bash
# 3. Visit http://<EC2-PUBLIC-IP>:8000/docs
```

## Visualizations

7 analytical plots generated in `results/visualizations/`:

| Plot | Description |
|------|-------------|
| `training_curves.png` | Loss, LR, per-class Dice over 20 epochs |
| `validation_dice_boxplot.png` | Dice distribution across 102 test cases |
| `confusion_matrix.png` | OA classification confusion matrix |
| `roc_pr_curves.png` | ROC & Precision-Recall curves (AUC) |
| `measurement_error.png` | AI vs GT measurement MAE bar chart |
| `dice_heatmap.png` | Per-case × per-structure Dice heatmap |
| `feature_importance.png` | OA morphometric feature importance |

Regenerate: `python scripts/generate_visualizations.py`

## License

Dataset: CC BY-NC 4.0 (OAIZIB-CM)
Code: Research/Educational Use
