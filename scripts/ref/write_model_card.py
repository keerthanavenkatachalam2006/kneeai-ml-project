"""Build and register the model card for the locally trained OAIZIB-CM model.

``ml/registry.py`` refuses to construct a :class:`ModelCard` without a
:class:`DatasetContract` (PRD §113), so this script is where the training run
stops being an anonymous ``.pt`` file and becomes an auditable artefact.

It writes two things:

1. A versioned registry entry under ``artifacts/models/seg_oaizib/<version>/``
   with the checkpoint copied in and content-hashed.
2. A copy of the card and checkpoint at
   ``artifacts/models/seg_oaizib/model_card.json`` — the flat path the
   application's inference adapter looks for.

Every metric written here is read from the training history file. Nothing is
typed in by hand.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import torch

from kneeai.core.anatomy import Modality, Structure
from kneeai.imaging.preprocess import IntensityNorm, PreprocessingSpec
from kneeai.ml.registry import (
    Architecture,
    DatasetContract,
    ModelCard,
    ModelMetrics,
    ModelRegistry,
    ModelTask,
)

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "seg_oaizib"
VERSION = "0.1.0-local"

# Background is included deliberately: ml/infer.py derives the network's
# out_channels from len(label_map), so omitting it builds a 5-channel head for a
# 6-channel checkpoint. build_result_payload filters background back out when it
# serialises result.json.
LABEL_MAP = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ckpt_path = REPO / "artifacts/models/seg_oaizib/checkpoint.pt"
    hist_path = REPO / "artifacts/metrics/oaizib_history.json"
    manifest = json.loads((REPO / "data/manifests/oaizib_40.json").read_text())

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    history = json.loads(hist_path.read_text())

    validated = [h for h in history if "val_dice_mean" in h]
    if not validated:
        raise SystemExit("no validation epochs in history; nothing to record")
    best = max(validated, key=lambda h: h["val_dice_mean"])

    train_split = json.loads((REPO / "data/splits/oaizib_40_train.json").read_text())
    val_split = json.loads((REPO / "data/splits/oaizib_40_val.json").read_text())

    kl_all: dict[str, int] = {}
    for split in (train_split, val_split):
        for grade, n in split.get("kl_histogram", {}).items():
            kl_all[grade] = kl_all.get(grade, 0) + n

    contract = DatasetContract(
        input_description=(
            "3D DESS knee MR, single channel, resampled to 1.0 mm isotropic from a "
            "native 0.7 x 0.365 x 0.365 mm acquisition, percentile-clipped (0.5-99.5) "
            "then z-score normalised on non-zero voxels"
        ),
        output_description=(
            "6-class voxel label map: background, femur, femoral cartilage, tibia, "
            "medial tibial cartilage, lateral tibial cartilage"
        ),
        ground_truth_description=(
            "Manual expert segmentation from the OAI-ZIB dataset, extended by "
            "CartiMorph (OAIZIB-CM) to split tibial cartilage into medial and "
            "lateral compartments."
        ),
        annotator="ZIB expert manual segmentation; medial/lateral tibial split by CartiMorph",
        dataset_ids=("oaizib_cm",),
        n_patients=int(manifest["n_patients"]),
        n_studies=int(manifest["n_cases"]),
        split_patients={
            "train": int(train_split["n_cases"]),
            "val": int(val_split["n_cases"]),
            "test": 0,
        },
        split_strategy="patient_level_random",
        license="CC-BY-NC-4.0",
        commercial_use_permitted=False,
        external_test_dataset_ids=(),
        population_notes=(
            "EVERY knee in OAIZIB-CM is a RIGHT knee; the model has never seen a "
            "native left knee and relies on sagittal mirror augmentation for "
            "left-side generalisation. Kellgren-Lawrence grade histogram across the "
            f"40 cases used here: {json.dumps(kl_all, sort_keys=True)}. "
            "Age, sex and BMI are available per case in the manifest. Cohort is "
            "OAI-derived, so it is US-based and skewed towards osteoarthritis risk; "
            "it is not representative of an Indian surgical population."
        ),
    )

    per_structure = best["val_dice_per_structure"]
    metrics = ModelMetrics(
        internal_validation={
            "n_cases": int(val_split["n_cases"]),
            "n_patients": int(val_split["n_cases"]),
            "dice_mean_excl_background": round(float(best["val_dice_mean"]), 4),
            "dice_per_structure": {k: round(float(v), 4) for k, v in per_structure.items()},
            "best_epoch": int(best["epoch"]),
            "epochs_trained": int(max(h["epoch"] for h in history)),
            "final_train_loss": round(float(history[-1]["train_loss"]), 4),
            "metric_definition": (
                "MONAI DiceMetric, include_background=False, mean over the validation "
                "set, computed on sliding-window inference at 64^3 with 0.25 overlap"
            ),
        },
        internal_test={},
        external_test={},
        clinical_measurement_impact={},
    )

    card = ModelCard(
        model_id=MODEL_ID,
        version=VERSION,
        task=ModelTask.SEGMENTATION,
        architecture=Architecture.UNET,
        dataset_contract=contract,
        label_map=LABEL_MAP,
        # Must round-trip through PreprocessingSpec.from_dict: ml/infer.py rebuilds
        # the spec from the card so inference cannot silently diverge from training.
        preprocessing=PreprocessingSpec(
            name="oaizib_bone_cartilage_1mm",
            orientation="RAS",
            target_spacing_mm=(1.0, 1.0, 1.0),
            intensity=IntensityNorm.ZSCORE,
            clip_percentiles=(0.5, 99.5),
            zscore_foreground_only=True,
            crop_to_roi=False,
            dtype="float32",
        ).to_dict(),
        modality=Modality.MR,
        input_channels=1,
        patch_size=(64, 64, 64),
        metrics=metrics,
        intended_use=(
            "Research and demonstration only. Produces bone and cartilage voxel "
            "labels from knee MR to drive 3D reconstruction, anatomical measurement "
            "and implant-sizing experiments inside the Knee Intelligence Platform."
        ),
        limitations=(
            "TRAINED ON 30 PATIENTS FOR A SHORT SCHEDULE ON A LAPTOP GPU. Not a "
            "clinical-grade model. Specific known limitations: (1) cartilage Dice is "
            "materially lower than bone because the tibial cartilages occupy roughly "
            "20k voxels each against 1.9M for femur, and 1.0 mm resampling removes "
            "much of the thin-structure signal; (2) no meniscus or patella output — "
            "OAIZIB-CM contains no such labels; (3) all training knees are right "
            "knees, so left-knee performance is unmeasured and rests entirely on "
            "mirror augmentation; (4) no external validation on any second dataset; "
            "(5) single train/val split, no cross-validation, so the reported Dice "
            "has a wide and unquantified confidence interval; (6) trained on an "
            "OAI-derived US cohort, not an Indian population."
        ),
        contraindications=(
            "Must not be used for any clinical decision, diagnosis or surgical plan "
            "affecting a real patient.",
            "Must not be used on left knees without mirroring to right first; "
            "performance on native left knees has never been measured.",
            "Must not be relied on for meniscus or patella geometry — the model "
            "cannot produce them.",
            "Must not be used on modalities other than 3D DESS-like knee MR.",
            "Reported Dice must not be quoted as clinical accuracy; the validation "
            "set is 10 patients.",
        ),
        framework="pytorch",
        framework_version=torch.__version__,
        created_by="local training run (Apple M1, MPS)",
        training_run_id=f"local-mps-{ckpt.get('epoch', '?')}ep",
        git_commit=git_commit(),
        notes=(
            "Augmentation during training: RandCropByPosNegLabel(pos=3, neg=1, "
            "num_samples=2); RandFlip on the sagittal axis at p=0.5, which maps a "
            "right knee onto a correctly labelled left knee; RandGaussianNoise "
            "p=0.15 std=0.03; RandShiftIntensity p=0.20 offsets=0.10; "
            "RandAdjustContrast p=0.15 gamma 0.8-1.3. Training patch 64^3. "
            "1.0 mm isotropic resampling was chosen to fit 8 GB of unified memory "
            "on an Apple M1, not because it is optimal: the native 0.365 mm "
            "in-plane resolution carries cartilage detail this discards. "
            "Trained by scripts/train_oaizib_local.py on Apple M1 MPS. This is the "
            "reduced-scale variant of the Colab recipe: same data, splits, label map, "
            "loss and metrics, but a 4.81M-parameter UNet at 1.0 mm instead of an "
            "SSL-pretrained SwinUNETR at native resolution. The SwinUNETR encoder "
            "path exists in ml/pretrained.py and remains the intended production "
            "route; it does not fit in 8 GB."
        ),
        approved_for_clinical_use=False,
    )

    registry = ModelRegistry(REPO / "artifacts/models")
    registry.register(card, checkpoint_path=ckpt_path, overwrite=True)

    # Flat contract path the application adapter looks for.
    flat = REPO / "artifacts/models/seg_oaizib"
    versioned_card = flat / VERSION / "model_card.json"
    shutil.copy2(versioned_card, flat / "model_card.json")

    print("registered:", card.key)
    print("versioned card:", versioned_card.relative_to(REPO))
    print("contract card :", (flat / "model_card.json").relative_to(REPO))
    print("checkpoint    :", ckpt_path.relative_to(REPO))
    print()
    print("recorded validation metrics:")
    print(json.dumps(metrics.internal_validation, indent=2))


if __name__ == "__main__":
    main()
