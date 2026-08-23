"""
KneeAI FastAPI Backend.
Exposes the full pipeline: data loading → segmentation → 3D reconstruction →
measurements → implant matching → OA research.
"""
import base64
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from .config import LABELS_MAP, OAIZIB_CM_INFO, MESH_OUTPUT_DIR
from .data_loader import OAIZIBDataLoader
from .reconstruction import MeshReconstructor
from .measurements import KneeMeasurementEngine
from .implant_database import ImplantMatcher
from .oa_research import OAResearchEngine
from .segmentation.inference import KneeSegmentationModel, evaluate_model_vs_ground_truth

# Initialize modules
app = FastAPI(
    title="KneeAI",
    description="AI-powered knee osteoarthritis analysis and implant planning platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module instances
data_loader = OAIZIBDataLoader()
reconstructor = MeshReconstructor(smoothing_sigma=1.0, step_size=2)
measurement_engine = KneeMeasurementEngine()
implant_matcher = ImplantMatcher()
oa_engine = OAResearchEngine()

# AI segmentation model (lazy-loaded)
ai_model = KneeSegmentationModel()

# Cache for loaded cases
_case_cache: dict = {}


# ─── API Models ───────────────────────────────────────────────────────────────

class CaseAnalysisResponse(BaseModel):
    case_id: int
    metadata: dict
    measurements: dict
    implant_matches: dict
    oa_assessment: dict
    reference_comparison: dict
    mesh_available: bool


class DatasetSummaryResponse(BaseModel):
    dataset_info: dict
    summary_stats: dict


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "KneeAI API",
        "version": "1.0.0",
        "endpoints": [
            "/dataset/summary",
            "/dataset/cases",
            "/case/{case_id}/analyze",
            "/case/{case_id}/mesh",
            "/case/{case_id}/measurements",
            "/case/{case_id}/implants",
            "/implants/database",
            "/research/correlations",
            "/api/predict-oa",
            "/api/predict-oa-from-measurements",
        ],
    }


@app.get("/dataset/summary", response_model=DatasetSummaryResponse)
def get_dataset_summary():
    """Get dataset overview and summary statistics."""
    return DatasetSummaryResponse(
        dataset_info=OAIZIB_CM_INFO.model_dump(),
        summary_stats=data_loader.get_dataset_summary(),
    )


@app.get("/dataset/cases")
def list_cases(
    split: Optional[str] = Query(None, description="Filter by 'train' or 'test'"),
    kl_grade: Optional[int] = Query(None, description="Filter by KL grade (0-4)"),
    limit: int = Query(50, ge=1, le=507),
    offset: int = Query(0, ge=0),
):
    """List available cases with optional filtering."""
    df = data_loader.load_metadata()

    if split:
        df = df[df["split"] == split]
    if kl_grade is not None:
        df = df[df["KLGrade"] == kl_grade]

    total = len(df)
    df = df.iloc[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "cases": df[
            ["CMT-ID", "SubjectID", "KLGrade", "Gender_Label", "Age", "BMI", "split", "side"]
        ].to_dict(orient="records"),
    }


@app.get("/case/{case_id}/analyze", response_model=CaseAnalysisResponse)
def analyze_case(case_id: int):
    """
    Full analysis pipeline for a single case.
    Extracts data, computes measurements, matches implants, assesses OA.
    """
    try:
        # Load or get from cache
        if case_id not in _case_cache:
            case_data = data_loader.load_case(case_id)
            _case_cache[case_id] = case_data
        else:
            case_data = _case_cache[case_id]

        label_volume = case_data["label"]
        voxel_size = case_data["voxel_size"]
        metadata = case_data["metadata"]

        # Measurements
        measurements = measurement_engine.measure_case(label_volume, voxel_size)

        # Implant matching
        implant_matches = implant_matcher.match_all(measurements)
        # Convert to serializable format
        implant_matches_serialized = {}
        for comp, matches in implant_matches.items():
            implant_matches_serialized[comp] = [m.model_dump() for m in matches]

        # OA assessment (morphometric heuristic, NOT a clinical diagnosis)
        oa_assessment = oa_engine.assess_oa_morphometrics(
            measurements, clinical_kl=metadata.get("KLGrade")
        )

        # Reference comparison
        reference = oa_engine.get_reference_comparison(
            measurements,
            patient_age=metadata.get("Age", 60),
            patient_gender=metadata.get("Gender_Label", "Unknown"),
            patient_kl=metadata.get("KLGrade"),
        )

        # Check if mesh already exists
        mesh_path = MESH_OUTPUT_DIR / f"case_{case_id:03d}.glb"
        mesh_available = mesh_path.exists()

        return CaseAnalysisResponse(
            case_id=case_id,
            metadata={
                k: v for k, v in metadata.items()
                if k in ["CMT-ID", "SubjectID", "KLGrade", "Gender_Label", "Age", "BMI", "side", "split"]
            },
            measurements=measurements,
            implant_matches=implant_matches_serialized,
            oa_assessment=oa_assessment,
            reference_comparison=reference,
            mesh_available=mesh_available,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/case/{case_id}/mesh")
def get_case_mesh(
    case_id: int,
    labels: Optional[str] = Query(None, description="Comma-separated label IDs (1-5)"),
):
    """
    Get 3D mesh (GLB format) for a case.
    Generates from segmentation if not cached.
    """
    try:
        # Check cache first
        mesh_path = MESH_OUTPUT_DIR / f"case_{case_id:03d}.glb"

        if not mesh_path.exists():
            # Generate mesh
            if case_id not in _case_cache:
                case_data = data_loader.load_case(case_id)
                _case_cache[case_id] = case_data
            else:
                case_data = _case_cache[case_id]

            label_volume = case_data["label"]
            voxel_size = case_data["voxel_size"]

            label_ids = None
            if labels:
                label_ids = [int(x.strip()) for x in labels.split(",")]

            reconstructor.export_case(case_id, label_volume, voxel_size)

        # Read and return GLB
        glb_data = mesh_path.read_bytes()
        return Response(
            content=glb_data,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f"inline; filename=case_{case_id:03d}.glb",
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")


@app.get("/case/{case_id}/measurements")
def get_case_measurements(case_id: int):
    """Get measurements only (without full analysis)."""
    try:
        if case_id not in _case_cache:
            case_data = data_loader.load_case(case_id)
            _case_cache[case_id] = case_data
        else:
            case_data = _case_cache[case_id]

        measurements = measurement_engine.measure_case(
            case_data["label"], case_data["voxel_size"]
        )
        return {"case_id": case_id, "measurements": measurements}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/case/{case_id}/implants")
def get_implant_matches(case_id: int, top_n: int = Query(5, ge=1, le=20)):
    """Get implant matching results for a case."""
    try:
        if case_id not in _case_cache:
            case_data = data_loader.load_case(case_id)
            _case_cache[case_id] = case_data
        else:
            case_data = _case_cache[case_id]

        measurements = measurement_engine.measure_case(
            case_data["label"], case_data["voxel_size"]
        )
        matches = implant_matcher.match_all(measurements, top_n=top_n)

        result = {}
        for comp, match_list in matches.items():
            result[comp] = [m.model_dump() for m in match_list]

        return {"case_id": case_id, "matches": result}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/case/{case_id}/validate")
def validate_case(case_id: int):
    """Validate data integrity for a case."""
    return data_loader.validate_case(case_id)


@app.get("/implants/database")
def get_implant_database():
    """Get the full implant database."""
    return implant_matcher.get_all_implants()


@app.get("/research/correlations")
def get_oa_correlations():
    """Get OA correlation analysis across the population."""
    return oa_engine.get_oa_correlation_summary()


@app.get("/labels")
def get_labels():
    """Get the segmentation label map."""
    return {"labels": LABELS_MAP}


@app.get("/ai/status")
def get_ai_status():
    """Check if the AI segmentation model is available."""
    model_exists = ai_model.model_path.exists()
    return {
        "model_available": model_exists,
        "model_loaded": ai_model.is_loaded,
        "model_path": str(ai_model.model_path),
    }


@app.get("/case/{case_id}/predict")
def predict_case(case_id: int):
    """
    Run AI segmentation on a case and compare against ground truth.
    Returns both the AI prediction metrics and measurements from both pipelines.
    """
    try:
        # Load model if not loaded
        if not ai_model.is_loaded:
            if not ai_model.load():
                raise HTTPException(
                    status_code=503,
                    detail="AI model not available. Train it first with scripts/train_segmentation.py"
                )

        # Load case data
        if case_id not in _case_cache:
            case_data = data_loader.load_case(case_id)
            _case_cache[case_id] = case_data
        else:
            case_data = _case_cache[case_id]

        # Run AI prediction and evaluate against ground truth
        eval_results = evaluate_model_vs_ground_truth(
            ai_model,
            case_data["image"],
            case_data["label"],
            case_data["voxel_size"],
        )

        # Also get measurements from AI prediction for comparison
        ai_label = ai_model.predict(case_data["image"], case_data["voxel_size"])
        ai_measurements = measurement_engine.measure_case(ai_label, case_data["voxel_size"])
        gt_measurements = measurement_engine.measure_case(case_data["label"], case_data["voxel_size"])

        return {
            "case_id": case_id,
            "evaluation": eval_results,
            "ai_measurements": ai_measurements,
            "ground_truth_measurements": gt_measurements,
            "pipeline": "ai_segresnet",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ─── OA Prediction Endpoint ───────────────────────────────────────────────────


class OaPredictionResponse(BaseModel):
    """Response model for the OA prediction endpoint."""

    oa_detected: bool
    probability: float
    severity: str
    approximate_kl_range: str
    confidence_score: float
    indicators: list
    compartment_assessment: dict
    features_used: dict
    features_missing: list
    warnings: list
    disclaimer: str
    method: str


@app.post("/api/predict-oa", response_model=OaPredictionResponse)
async def predict_oa(file: UploadFile = File(..., description="NIfTI file (.nii.gz)")):
    """Predict OA from an uploaded NIfTI knee MR scan.

    Runs the full pipeline: segmentation → measurements → OA classification.
    Returns a morphometric OA assessment with probability, severity, and indicators.

    Accepts: .nii.gz or .nii files.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not (file.filename.endswith(".nii.gz") or file.filename.endswith(".nii")):
        raise HTTPException(
            status_code=400,
            detail="File must be a NIfTI file (.nii or .nii.gz)",
        )

    try:
        # Save uploaded file to temp location
        suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            import numpy as np

            # Load the NIfTI file
            try:
                import nibabel as nib
            except ImportError:
                raise HTTPException(
                    status_code=503,
                    detail="nibabel is not installed. Install with: pip install nibabel",
                )

            nii = nib.load(str(tmp_path))
            image_data = np.asarray(nii.dataobj, dtype=np.float32)
            voxel_size = tuple(float(v) for v in nii.header.get_zooms()[:3])

            # Run AI segmentation
            if not ai_model.is_loaded:
                if not ai_model.load():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "AI segmentation model not available. "
                            "Train it first with: python scripts/train_segmentation.py"
                        ),
                    )

            seg_mask = ai_model.predict(image_data, voxel_size)

            # Compute measurements
            measurements_result = measurement_engine.measure_case(seg_mask, voxel_size)

            # Convert measurements to the format expected by the morphometric classifier.
            # The measurement_engine returns a dict; we need to transform it into the
            # list-of-measurements format that the pipeline outputs.
            measurements_list = _convert_measurements_to_list_format(measurements_result)

            # Run OA prediction
            from kneeai.oa.morphometric_classifier import MorphometricOaClassifier

            classifier = MorphometricOaClassifier(oa_threshold=0.35)
            prediction = classifier.predict({"measurements": measurements_list})
            result = prediction.to_dict()

            return OaPredictionResponse(
                oa_detected=result["oa_detected"],
                probability=result["probability"],
                severity=result["severity"],
                approximate_kl_range=result["approximate_kl_range"],
                confidence_score=result["confidence_score"],
                indicators=result["indicators"],
                compartment_assessment=result["compartment_assessment"],
                features_used=result["features_used"],
                features_missing=result["features_missing"],
                warnings=result["warnings"],
                disclaimer=result["disclaimer"],
                method=result["method"],
            )

        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OA prediction failed: {str(e)}")


@app.post("/api/predict-oa-from-measurements")
async def predict_oa_from_measurements_endpoint(measurements: dict):
    """Predict OA from pre-computed measurements JSON.

    Accepts the measurements.json payload directly (useful when measurements
    have already been computed by the pipeline).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

    try:
        from kneeai.oa.morphometric_classifier import MorphometricOaClassifier

        classifier = MorphometricOaClassifier(oa_threshold=0.35)
        prediction = classifier.predict(measurements)
        return prediction.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OA prediction failed: {str(e)}")


def _convert_measurements_to_list_format(measurements: dict) -> list[dict]:
    """Convert the measurement_engine output to the pipeline list format.

    The measurement_engine returns a nested dict; the morphometric classifier
    expects the pipeline's list-of-measurements format with name/quantity pairs.
    """
    result = []

    # Map from measurement_engine keys to pipeline names
    fc = measurements.get("femoral_cartilage", {})
    mtc = measurements.get("medial_tibial_cartilage", {})
    ltc = measurements.get("lateral_tibial_cartilage", {})
    derived = measurements.get("derived", {})
    bone = measurements.get("bone_geometry", {})

    mappings = [
        ("femoral_cartilage_mean_thickness", fc.get("thickness_mean_mm"), "mm"),
        ("medial_tibial_cartilage_mean_thickness", mtc.get("thickness_mean_mm"), "mm"),
        ("lateral_tibial_cartilage_mean_thickness", ltc.get("thickness_mean_mm"), "mm"),
        ("femoral_cartilage_volume", fc.get("volume_mm3"), "mm^3"),
        ("medial_tibial_cartilage_volume", mtc.get("volume_mm3"), "mm^3"),
        ("lateral_tibial_cartilage_volume", ltc.get("volume_mm3"), "mm^3"),
        ("cartilage_thickness_asymmetry", derived.get("medial_lateral_thickness_asymmetry_pct"), "%"),
        ("jlca", bone.get("jlca_deg"), "deg"),
        ("posterior_tibial_slope", bone.get("posterior_tibial_slope_deg"), "deg"),
        ("femoral_ml_width", bone.get("femoral_ml_width_mm"), "mm"),
        ("tibial_ml_width", bone.get("tibial_ml_width_mm"), "mm"),
    ]

    for name, value, unit in mappings:
        if value is not None:
            result.append({
                "name": name,
                "quantity": {"value": value, "unit": unit},
            })

    return result


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Pre-load metadata on startup."""
    data_loader.load_metadata()
