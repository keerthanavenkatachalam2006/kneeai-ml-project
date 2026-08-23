"""
Data loading and preprocessing module for OAIZIB-CM dataset.
Handles ZIP extraction, NIfTI loading, metadata parsing, and validation.
"""
import zipfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from .config import (
    OAIZIB_ROOT,
    OAIZIB_INFO,
    PROCESSED_DATA,
    OAIZIB_CM_INFO,
    LABELS_MAP,
)


class OAIZIBDataLoader:
    """Loader for the OAIZIB-CM knee MRI segmentation dataset."""

    def __init__(self):
        self.info = OAIZIB_CM_INFO
        self.extract_dir = PROCESSED_DATA / "extracted"
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_cache: Optional[pd.DataFrame] = None

    def load_metadata(self) -> pd.DataFrame:
        """Load and merge train/test metadata into a single DataFrame."""
        if self._metadata_cache is not None:
            return self._metadata_cache

        train_path = OAIZIB_INFO / "subInfo_train.xlsx"
        test_path = OAIZIB_INFO / "subInfo_test.xlsx"

        df_train = pd.read_excel(train_path)
        df_train["split"] = "train"

        df_test = pd.read_excel(test_path)
        df_test["split"] = "test"

        df = pd.concat([df_train, df_test], ignore_index=True)

        # Parse knee side info
        knee_side_path = OAIZIB_INFO / "kneeSideInfo.csv"
        knee_sides = pd.read_csv(
            knee_side_path, header=None, names=["filename", "side"]
        )
        knee_sides["CMT-ID"] = (
            knee_sides["filename"]
            .str.extract(r"oaizib_(\d+)")
            .astype(int)
        )
        df = df.merge(knee_sides[["CMT-ID", "side"]], on="CMT-ID", how="left")

        # Exclude corrupted cases
        df = df[~df["CMT-ID"].isin(self.info.corrupted_cases)].reset_index(drop=True)

        # Map gender codes
        df["Gender_Label"] = df["Gender"].map({1: "Male", 2: "Female"})

        self._metadata_cache = df
        return df

    def extract_case(self, case_id: int, force: bool = False) -> dict:
        """
        Extract a single case (image + label) from ZIP files.

        Returns dict with paths to extracted NIfTI files.
        """
        if case_id in self.info.corrupted_cases:
            raise ValueError(f"Case {case_id} is known to be corrupted.")

        metadata = self.load_metadata()
        case_row = metadata[metadata["CMT-ID"] == case_id]
        if case_row.empty:
            raise ValueError(f"Case {case_id} not found in metadata.")

        split = case_row.iloc[0]["split"]
        img_zip_name = f"images{'Tr' if split == 'train' else 'Ts'}.zip"
        lbl_zip_name = f"labels{'Tr' if split == 'train' else 'Ts'}.zip"

        img_filename = f"oaizib_{case_id:03d}_0000.nii.gz"
        lbl_filename = f"oaizib_{case_id:03d}.nii.gz"

        img_subdir = f"images{'Tr' if split == 'train' else 'Ts'}"
        lbl_subdir = f"labels{'Tr' if split == 'train' else 'Ts'}"

        img_path = self.extract_dir / img_subdir / img_filename
        lbl_path = self.extract_dir / lbl_subdir / lbl_filename

        # Extract if not already present
        if not img_path.exists() or force:
            self._extract_from_zip(
                OAIZIB_ROOT / img_zip_name,
                f"{img_subdir}/{img_filename}",
                self.extract_dir,
            )

        if not lbl_path.exists() or force:
            self._extract_from_zip(
                OAIZIB_ROOT / lbl_zip_name,
                f"{lbl_subdir}/{lbl_filename}",
                self.extract_dir,
            )

        return {
            "case_id": case_id,
            "split": split,
            "image_path": img_path,
            "label_path": lbl_path,
            "metadata": case_row.iloc[0].to_dict(),
        }

    def load_nifti(self, path: Path) -> dict:
        """Load a NIfTI file and return data + metadata."""
        img = nib.load(str(path))
        data = img.get_fdata()
        return {
            "data": data,
            "affine": img.affine,
            "header": img.header,
            "voxel_size": img.header.get_zooms(),
            "shape": data.shape,
        }

    def load_case(self, case_id: int) -> dict:
        """
        Full pipeline: extract + load image and label for a case.
        Returns numpy arrays and metadata.
        """
        paths = self.extract_case(case_id)
        image_data = self.load_nifti(paths["image_path"])
        label_data = self.load_nifti(paths["label_path"])

        return {
            "case_id": case_id,
            "metadata": paths["metadata"],
            "image": image_data["data"],
            "label": label_data["data"],
            "affine": image_data["affine"],
            "voxel_size": image_data["voxel_size"],
            "shape": image_data["shape"],
        }

    def validate_case(self, case_id: int) -> dict:
        """Validate a single case for data integrity."""
        try:
            case = self.load_case(case_id)
            image = case["image"]
            label = case["label"]

            checks = {
                "case_id": case_id,
                "valid": True,
                "image_shape": image.shape,
                "label_shape": label.shape,
                "shapes_match": image.shape == label.shape,
                "image_range": (float(image.min()), float(image.max())),
                "label_values": sorted(np.unique(label).astype(int).tolist()),
                "expected_labels": list(LABELS_MAP.keys()),
                "has_all_labels": set(np.unique(label).astype(int)) == set(LABELS_MAP.keys()),
                "voxel_size": case["voxel_size"],
                "errors": [],
            }

            if not checks["shapes_match"]:
                checks["errors"].append("Image and label shapes do not match")
                checks["valid"] = False

            if image.max() == 0:
                checks["errors"].append("Image is all zeros")
                checks["valid"] = False

            return checks

        except Exception as e:
            return {
                "case_id": case_id,
                "valid": False,
                "errors": [str(e)],
            }

    def get_dataset_summary(self) -> dict:
        """Get summary statistics of the full dataset."""
        metadata = self.load_metadata()
        return {
            "total_cases": len(metadata),
            "train_cases": len(metadata[metadata["split"] == "train"]),
            "test_cases": len(metadata[metadata["split"] == "test"]),
            "kl_grade_distribution": metadata["KLGrade"].value_counts().to_dict(),
            "gender_distribution": metadata["Gender_Label"].value_counts().to_dict(),
            "age_stats": {
                "mean": metadata["Age"].mean(),
                "std": metadata["Age"].std(),
                "min": metadata["Age"].min(),
                "max": metadata["Age"].max(),
            },
            "bmi_stats": {
                "mean": metadata["BMI"].mean(),
                "std": metadata["BMI"].std(),
                "min": metadata["BMI"].min(),
                "max": metadata["BMI"].max(),
            },
            "corrupted_cases": self.info.corrupted_cases,
        }

    @staticmethod
    def _extract_from_zip(zip_path: Path, member: str, dest: Path):
        """Extract a single member from a ZIP file."""
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extract(member, dest)
