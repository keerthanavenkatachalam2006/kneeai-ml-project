"""
Application configuration and paths.
"""
from pathlib import Path
from pydantic import BaseModel


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"
METADATA_DIR = DATA_ROOT / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"

# OAIZIB-CM specific
OAIZIB_ROOT = RAW_DATA / "OAIZIB-CM"
OAIZIB_INFO = OAIZIB_ROOT / "info" / "info"

# Output directories
MESH_OUTPUT_DIR = PROCESSED_DATA / "meshes"
MEASUREMENT_OUTPUT_DIR = PROCESSED_DATA / "measurements"

# Ensure directories exist
for d in [PROCESSED_DATA, METADATA_DIR, MODELS_DIR, MESH_OUTPUT_DIR, MEASUREMENT_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Segmentation label mapping
LABELS_MAP = {
    0: "Background",
    1: "Femur",
    2: "Femoral Cartilage",
    3: "Tibia",
    4: "Medial Tibial Cartilage",
    5: "Lateral Tibial Cartilage",
    6: "Medial Meniscus",
}

# Colors for 3D visualization (RGBA, 0-1 range)
LABEL_COLORS = {
    1: (0.9, 0.85, 0.75, 0.3),       # Femur - bone white, semi-transparent
    2: (0.2, 0.6, 1.0, 0.8),         # Femoral Cartilage - blue
    3: (0.85, 0.80, 0.70, 0.3),      # Tibia - bone white, semi-transparent
    4: (1.0, 0.4, 0.4, 0.8),         # Medial Tibial Cartilage - red
    5: (0.4, 1.0, 0.4, 0.8),         # Lateral Tibial Cartilage - green
    6: (1.0, 0.7, 0.0, 0.9),         # Medial Meniscus - orange
}


class DatasetInfo(BaseModel):
    """Dataset metadata schema."""
    name: str
    source: str
    license: str
    size_subjects: int
    anatomy_labels: dict
    modality: str
    intended_use: str
    oai_derived: bool
    corrupted_cases: list[int]
    train_count: int
    test_count: int
    voxel_size_mm: tuple
    image_shape: tuple


OAIZIB_CM_INFO = DatasetInfo(
    name="OAIZIB-CM",
    source="https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM",
    license="CC BY-NC 4.0",
    size_subjects=507,
    anatomy_labels=LABELS_MAP,
    modality="MRI (3D sagittal DESS)",
    intended_use="Knee bone and cartilage segmentation, OA research",
    oai_derived=True,
    corrupted_cases=[454],
    train_count=404,
    test_count=103,
    voxel_size_mm=(0.7, 0.3646, 0.3646),
    image_shape=(160, 384, 384),
)
