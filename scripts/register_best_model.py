"""Register ``best_model.pth`` so ``kneeai.ml.infer`` can load it.

``best_model.pth`` arrived as a bare ``OrderedDict`` with no metadata whatsoever —
no label map, no ``model_config``, no preprocessing spec, no training log. Its
configuration was recovered empirically by ``scripts/identify_checkpoint.py`` and
``scripts/diagnose_checkpoint.py``, scoring candidates against OAIZIB-CM ground
truth. What those established:

======================  =========================================================
architecture            MONAI ``SegResNet`` (``convInit`` / ``down_layers`` /
                        ``up_samples`` / ``conv_final`` key naming)
init_filters            16, from ``convInit.conv.weight`` = (16, 1, 3, 3, 3)
blocks_down/up          (1, 2, 2, 4) / (1, 1, 1), from the block index ranges
out_channels            6, from ``conv_final.2.conv.weight`` = (6, 16, 1, 1, 1)
norm                    ``GroupNorm(8)`` — MONAI's SegResNet default. Decisive:
                        GroupNorm scored 0.455 mean Dice against 0.127 for affine
                        InstanceNorm. Both fit the tensor shapes, so this could
                        only be settled by measurement.
target spacing          0.5 mm isotropic. Dice by spacing: 0.216 at 1.0 mm,
                        0.455 at 0.5 mm, 0.472 at 0.365 mm — but predicted bone
                        volume runs 0.25x truth at 1.0 mm, 1.11x at 0.5 mm and
                        1.81x at 0.365 mm. 0.5 mm is where the volumes are
                        calibrated, and it is 3x faster than native.
orientation             RAS. LPS scored 0.450 against RAS 0.455 — indistinguishable,
                        so RAS is kept to match the repo's training notebook.
axes                    verified unflipped: all 16 shape-preserving flip/swap
                        combinations scored below the identity.
======================  =========================================================

**What this checkpoint is not.** At its best recovered configuration it reaches
roughly 0.47 mean Dice, against 0.695 for the UNet already registered at
``0.1.0-local``. The failure mode is over-segmentation with soft boundaries — high
recall (femur 83%, tibia 89% of truth correctly labelled) and poor precision
(predicted bone volume 1.1-1.9x truth depending on spacing). That is the signature
of an undertrained network, not of a preprocessing mismatch: a mismatch puts the
anatomy in the wrong place, and this puts it in the right place with fuzzy edges.

So the numbers written onto this card are measured, not inherited, and the previous
model is left registered alongside rather than overwritten. If the checkpoint's
author can supply the real training spacing, patch size and intensity normalisation,
pass them with ``--spacing`` / ``--patch`` / ``--foreground-zscore`` and re-run; that
is the one thing that could close the gap.

Usage::

    python scripts/register_best_model.py --evaluate      # measure, then register
    python scripts/register_best_model.py --no-evaluate   # register with no metrics
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from kneeai.core.anatomy import Structure
from kneeai.ml.models import ModelConfig
from kneeai.ml.registry import (
    Architecture,
    DatasetContract,
    ModelCard,
    ModelMetrics,
    ModelRegistry,
    ModelTask,
)
from kneeai.version import PLATFORM_VERSION

REPO = Path(__file__).resolve().parents[1]

MODEL_ID = "seg_oaizib"
VERSION = "0.2.0-segresnet"

LABEL_MAP = {
    "background": 0,
    "femur": 1,
    "femoral_cartilage": 2,
    "tibia": 3,
    "medial_tibial_cartilage": 4,
    "lateral_tibial_cartilage": 5,
}

#: Recovered by scripts/identify_checkpoint.py. See the module docstring.
RECOVERED_FEATURES = (16, 32, 64, 128, 256)
RECOVERED_NORM = "group"
RECOVERED_NORM_GROUPS = 8

IDENTIFICATION_EVIDENCE = {
    "method": (
        "Grid search over the ambiguous configuration axes, scored by Dice against "
        "OAIZIB-CM ground truth on case oaizib-417. See scripts/identify_checkpoint.py "
        "and scripts/diagnose_checkpoint.py."
    ),
    "norm": {
        "chosen": "group(8)",
        "evidence": "mean Dice 0.455 vs 0.127 for instance(affine=True) at 0.5 mm",
        "note": (
            "Both candidates fit every tensor shape in the checkpoint, because "
            "GroupNorm and affine InstanceNorm have identical per-channel affine "
            "parameters. Only measurement separated them."
        ),
    },
    "spacing_mm": {
        "chosen": 0.5,
        "evidence": {
            "1.0": {"dice": 0.2162, "predicted_bone_volume_ratio": 0.25},
            "0.5": {"dice": 0.4568, "predicted_bone_volume_ratio": 1.11},
            "0.365": {"dice": 0.4716, "predicted_bone_volume_ratio": 1.81},
        },
        "note": (
            "0.365 mm scores marginally higher Dice but inflates predicted bone "
            "volume to 1.8x truth and costs 3x the inference time. 0.5 mm is where "
            "predicted volumes match measured volumes, which is what the downstream "
            "measurement and implant-sizing stages consume."
        ),
    },
    "orientation": {
        "chosen": "RAS",
        "evidence": "RAS 0.455 vs LPS 0.450 — within noise",
    },
    "axis_flips": {
        "checked": 16,
        "conclusion": "identity best; no flip or axis swap improved Dice",
    },
}


def build_checkpoint(state: dict[str, Any], patch: int) -> dict[str, Any]:
    config = ModelConfig(
        architecture=Architecture.SEGRESNET,
        in_channels=1,
        out_channels=len(LABEL_MAP),
        patch_size=(patch, patch, patch),
        features=RECOVERED_FEATURES,
        norm=RECOVERED_NORM,
        norm_groups=RECOVERED_NORM_GROUPS,
        dropout=0.0,
    )
    return {
        "model_state_dict": state,
        "label_map": dict(LABEL_MAP),
        "model_config": config.to_dict(),
        "n_classes": len(LABEL_MAP),
        "patch": patch,
        "run": "supplied-externally",
        "provenance_note": (
            "Received as a bare state dict with no metadata. Architecture and "
            "preprocessing recovered empirically; see "
            "scripts/register_best_model.py."
        ),
    }


def preprocessing_spec(spacing: float, orientation: str, foreground_zscore: bool) -> dict[str, Any]:
    from kneeai.imaging.preprocess import IntensityNorm, PreprocessingSpec

    return PreprocessingSpec(
        name=f"recovered_segresnet_{spacing}mm",
        orientation=orientation,
        target_spacing_mm=(spacing, spacing, spacing),
        intensity=IntensityNorm.ZSCORE,
        clip_percentiles=(0.5, 99.5),
        zscore_foreground_only=foreground_zscore,
        crop_to_roi=False,
        roi_margin_mm=15.0,
        min_shape=None,
        dtype="float32",
    ).to_dict()


def build_card(
    *,
    spacing: float,
    patch: int,
    orientation: str,
    foreground_zscore: bool,
    metrics: ModelMetrics,
) -> ModelCard:
    contract = DatasetContract(
        input_description=(
            f"3D knee MR, single channel, reoriented to {orientation} and resampled to "
            f"{spacing} mm isotropic, percentile-clipped (0.5-99.5) then z-score "
            f"normalised over {'foreground' if foreground_zscore else 'all'} voxels. "
            "This preprocessing was RECOVERED BY MEASUREMENT, not supplied by the "
            "checkpoint's author."
        ),
        output_description=(
            "6-class voxel label map: background, femur, femoral cartilage, tibia, "
            "medial tibial cartilage, lateral tibial cartilage"
        ),
        ground_truth_description=(
            "Unknown. The checkpoint was supplied as a bare state dict with no "
            "training record. Its 6-channel output head and the label ordering that "
            "maximises Dice against OAIZIB-CM both indicate it was trained on "
            "OAIZIB-CM or a close derivative, but this is inference, not a record."
        ),
        annotator="unknown (checkpoint supplied without provenance)",
        dataset_ids=("oaizib_cm",),
        n_patients=1,
        n_studies=1,
        split_patients={"train": 1, "val": 0, "test": 0},
        split_strategy="patient_level_unknown",
        license="unverified — assume CC-BY-NC-4.0 if OAIZIB-derived",
        commercial_use_permitted=False,
        external_test_dataset_ids=(),
        population_notes=(
            "UNKNOWN COHORT. Nothing is recorded about who this model was trained on. "
            "If it was trained on OAIZIB-CM then every training knee was a right knee "
            "from a US OAI-derived cohort, but that is an assumption. Treat any "
            "subgroup performance claim as unsupported."
        ),
    )

    return ModelCard(
        model_id=MODEL_ID,
        version=VERSION,
        task=ModelTask.SEGMENTATION,
        architecture=Architecture.SEGRESNET,
        dataset_contract=contract,
        label_map={Structure(name): value for name, value in LABEL_MAP.items()},
        preprocessing=preprocessing_spec(spacing, orientation, foreground_zscore),
        input_channels=1,
        patch_size=(patch, patch, patch),
        metrics=metrics,
        intended_use=(
            "Research and demonstration only. Produces bone and cartilage voxel labels "
            "from knee MR to drive 3D reconstruction, measurement, OA assessment and "
            "implant-sizing experiments inside the Knee Intelligence Platform."
        ),
        limitations=(
            "SUPPLIED WITHOUT PROVENANCE AND MATERIALLY WEAKER THAN THE MODEL IT SITS "
            "BESIDE. Measured mean Dice is roughly 0.47, against 0.695 for the UNet "
            "registered at 0.1.0-local. The error pattern is over-segmentation with "
            "soft boundaries: recall is high (femur 83%, tibia 89% of ground-truth "
            "voxels correctly labelled) and precision is poor (predicted bone volume "
            "1.1x truth at the registered spacing). That is characteristic of an "
            "undertrained network. Additionally: (1) no training record exists, so the "
            "preprocessing here is a measured best guess and may not be what the model "
            "saw; (2) the tibial cartilages are frequently confused with one another, "
            "which directly weakens any medial-versus-lateral compartment reading; "
            "(3) no external validation; (4) cohort, split strategy and licence are all "
            "unknown."
        ),
        contraindications=(
            "Must not be used for any clinical decision, diagnosis or surgical plan "
            "affecting a real patient.",
            "Must not be used where medial-versus-lateral compartment discrimination "
            "matters without human review — the two tibial cartilages are frequently "
            "swapped.",
            "Must not be used on modalities other than knee MR; the training modality "
            "is unrecorded and cross-modality behaviour is unmeasured.",
            "Must not be preferred over seg_oaizib:0.1.0-local for accuracy-sensitive "
            "work; it measures worse on the same ground truth.",
            "Reported Dice must not be quoted as clinical accuracy.",
        ),
        framework="pytorch",
        framework_version=torch.__version__,
        platform_version=PLATFORM_VERSION,
        created_at=datetime.now(timezone.utc),
        created_by="externally supplied checkpoint (best_model.pth); configuration recovered locally",
        training_run_id="unknown",
        git_commit="",
        notes=(
            "Configuration recovered empirically rather than supplied. "
            f"Evidence: {json.dumps(IDENTIFICATION_EVIDENCE)}"
        ),
        approved_for_clinical_use=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_model.pth")
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--orientation", default="RAS")
    ap.add_argument(
        "--foreground-zscore",
        action="store_true",
        help="z-score over foreground voxels only (default: all voxels)",
    )
    ap.add_argument(
        "--metrics",
        default="artifacts/segresnet_validation.json",
        help="measured validation metrics to attach, if the file exists",
    )
    ap.add_argument(
        "--activate",
        action="store_true",
        help="also write the flat model_card.json the inference adapter reads",
    )
    args = ap.parse_args()

    source = REPO / args.checkpoint
    if not source.exists():
        raise SystemExit(f"checkpoint not found: {source}")

    state = torch.load(source, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    print(f"source      : {source.relative_to(REPO)}")
    print(f"tensors     : {len(state)}")
    print(f"parameters  : {sum(v.numel() for v in state.values() if hasattr(v, 'numel')):,}")

    # --- prove the architecture round-trips before writing anything ---
    from kneeai.ml.models import build_model

    payload = build_checkpoint(state, args.patch)
    config = ModelConfig(
        architecture=Architecture.SEGRESNET,
        in_channels=1,
        out_channels=len(LABEL_MAP),
        patch_size=(args.patch,) * 3,
        features=RECOVERED_FEATURES,
        norm=RECOVERED_NORM,
        norm_groups=RECOVERED_NORM_GROUPS,
    )
    probe = build_model(config)
    missing, unexpected = probe.load_state_dict(state, strict=False)
    print(f"round-trip  : missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print("  first missing   :", list(missing)[:3])
        print("  first unexpected:", list(unexpected)[:3])
        raise SystemExit("architecture does not match the checkpoint; refusing to register")
    print("              PASS")

    # --- metrics, only if they were actually measured ---
    metrics = ModelMetrics()
    metrics_path = REPO / args.metrics
    if metrics_path.exists():
        measured = json.loads(metrics_path.read_text())
        metrics = ModelMetrics(internal_validation=measured.get("internal_validation", {}))
        print(f"metrics     : attached from {metrics_path.relative_to(REPO)}")
    else:
        print(
            f"metrics     : NONE — {args.metrics} not found. The card will carry no "
            "performance claim, which is correct until one is measured."
        )

    card = build_card(
        spacing=args.spacing,
        patch=args.patch,
        orientation=args.orientation,
        foreground_zscore=args.foreground_zscore,
        metrics=metrics,
    )

    # --- write the wrapped checkpoint, then register it ---
    registry = ModelRegistry(REPO / "artifacts" / "models")
    version_dir = registry._version_dir(MODEL_ID, VERSION)
    version_dir.mkdir(parents=True, exist_ok=True)
    wrapped = version_dir / "checkpoint.pt"
    torch.save(payload, wrapped)
    print(f"checkpoint  : {wrapped.relative_to(REPO)}")

    registry.register(card, wrapped, overwrite=True)
    print(f"registered  : {card.key}")

    if args.activate:
        # The inference adapter reads this flat path, not the registry.
        flat = REPO / "artifacts" / "models" / MODEL_ID / "model_card.json"
        previous = json.loads(flat.read_text()) if flat.exists() else None
        if previous:
            backup = flat.with_name("model_card.previous.json")
            backup.write_text(json.dumps(previous, indent=2) + "\n")
            print(f"backed up   : {backup.relative_to(REPO)} "
                  f"(was {previous.get('model_id')}:{previous.get('version')})")
        flat.write_text(json.dumps(card.to_dict(), indent=2, default=str) + "\n")
        print(f"ACTIVATED   : {flat.relative_to(REPO)} now points at {card.key}")
    else:
        print(
            "not activated. The inference adapter still reads the previous card. "
            "Re-run with --activate to switch."
        )


if __name__ == "__main__":
    main()
