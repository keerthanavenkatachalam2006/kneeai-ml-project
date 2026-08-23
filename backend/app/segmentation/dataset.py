"""
MONAI Dataset and Transforms for OAIZIB-CM knee MRI segmentation.

Strategy for 6GB VRAM:
- Patch-based training: random 96×96×96 patches (not full 160×384×384 volumes)
- Sliding window inference: full-volume at test time
- Aggressive augmentation to maximize data efficiency

Data pipeline:
  ZIP → NIfTI extraction → intensity normalization → random crop → augmentation → model
"""
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.config import OAIZIB_ROOT, PROCESSED_DATA


# Training patch size (fits in 6GB VRAM with batch_size=2)
# Native resolution: (0.7, 0.3646, 0.3646) mm — anisotropic
# Patch (64, 128, 128) → physical size (44.8mm, 46.7mm, 46.7mm)
PATCH_SIZE = (64, 128, 128)

# We train at native resolution (no resampling) for speed.
# The original voxel spacing is already fine for segmentation.
TARGET_SPACING = None  # No resampling


def prepare_data_list(extract_dir: Path = None) -> tuple[list[dict], list[dict]]:
    """
    Prepare train/test file lists by extracting from ZIPs.

    Returns:
        (train_files, test_files) — each is a list of {"image": path, "label": path}
    """
    if extract_dir is None:
        extract_dir = PROCESSED_DATA / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    train_files = []
    test_files = []

    # Extract training data
    img_tr_zip = OAIZIB_ROOT / "imagesTr.zip"
    lbl_tr_zip = OAIZIB_ROOT / "labelsTr.zip"
    img_ts_zip = OAIZIB_ROOT / "imagesTs.zip"
    lbl_ts_zip = OAIZIB_ROOT / "labelsTs.zip"

    # Get lists from ZIPs without extracting everything yet
    with zipfile.ZipFile(img_tr_zip, 'r') as z:
        img_tr_names = [n for n in z.namelist() if n.endswith('.nii.gz')]

    with zipfile.ZipFile(lbl_tr_zip, 'r') as z:
        lbl_tr_names = [n for n in z.namelist() if n.endswith('.nii.gz')]

    with zipfile.ZipFile(img_ts_zip, 'r') as z:
        img_ts_names = [n for n in z.namelist() if n.endswith('.nii.gz')]

    with zipfile.ZipFile(lbl_ts_zip, 'r') as z:
        lbl_ts_names = [n for n in z.namelist() if n.endswith('.nii.gz')]

    # Build case ID → file mapping
    def case_id_from_name(name: str) -> int:
        import re
        m = re.search(r'oaizib_(\d+)', name)
        return int(m.group(1)) if m else -1

    # Build train pairs (exclude case 454)
    lbl_tr_map = {case_id_from_name(n): n for n in lbl_tr_names}
    for img_name in img_tr_names:
        case_id = case_id_from_name(img_name)
        if case_id == 454:
            continue
        if case_id in lbl_tr_map:
            img_path = extract_dir / img_name
            lbl_path = extract_dir / lbl_tr_map[case_id]
            train_files.append({"image": str(img_path), "label": str(lbl_path)})

    # Build test pairs (exclude case 454)
    lbl_ts_map = {case_id_from_name(n): n for n in lbl_ts_names}
    for img_name in img_ts_names:
        case_id = case_id_from_name(img_name)
        if case_id == 454:
            continue
        if case_id in lbl_ts_map:
            img_path = extract_dir / img_name
            lbl_path = extract_dir / lbl_ts_map[case_id]
            test_files.append({"image": str(img_path), "label": str(lbl_path)})

    return train_files, test_files


def extract_all_data(extract_dir: Path = None) -> None:
    """Extract all images and labels from ZIPs. Run once before training."""
    if extract_dir is None:
        extract_dir = PROCESSED_DATA / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    zips = [
        OAIZIB_ROOT / "imagesTr.zip",
        OAIZIB_ROOT / "labelsTr.zip",
        OAIZIB_ROOT / "imagesTs.zip",
        OAIZIB_ROOT / "labelsTs.zip",
    ]

    for zip_path in zips:
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

    print(f"All data extracted to {extract_dir}")


def get_train_transforms(patch_size: tuple = PATCH_SIZE) -> Compose:
    """
    Training transforms with augmentation.
    No resampling — train at native resolution for speed.

    Pipeline:
    1. Load NIfTI
    2. Add channel dimension
    3. Normalize intensity (zero mean, unit variance)
    4. Random patch crop (64×128×128)
    5. Random augmentations (flip, rotate, noise, intensity)
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=2,  # 2 patches per volume per iteration
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    """
    Validation/inference transforms (no augmentation, no resampling).
    Full volume is loaded and processed.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_inference_transforms() -> Compose:
    """Transforms for inference on a single image (no label, no resampling)."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])
