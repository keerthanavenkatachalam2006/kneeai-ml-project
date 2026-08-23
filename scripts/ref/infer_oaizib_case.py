"""Run the trained OAIZIB-CM model on one case and emit contract-compliant output.

Writes ``artifacts/cases/<case_id>/{segmentation.nii.gz,result.json}`` exactly as
specified in ``docs/ML_CONTRACT.md``, so the application's inference adapter can
consume it without knowing anything about the model.

Two details that are easy to get wrong and are handled explicitly here:

**The mask must come back on the acquisition grid.** Inference runs at 1.0 mm
isotropic, but the contract requires the label volume on the *source* grid with
the source affine. Predicting at one spacing and shipping the result as though it
were the original would silently corrupt every downstream measurement. MONAI's
``Invertd`` replays the preprocessing chain backwards with nearest-neighbour
interpolation to undo it.

**NIfTI is RAS, the platform is LPS.** ``kneeai.imaging.Volume`` follows the DICOM
convention where an identity direction matrix means LPS. A nibabel affine is RAS.
The two differ by a flip of the first two axes, and skipping that conversion
mirrors the knee left-to-right — which for a medial/lateral judgement is exactly
the class of error the platform's frame tagging exists to prevent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

from kneeai.core.anatomy import Modality, Side, Structure
from kneeai.core.provenance import ArtifactRef, Origin, Provenance
from kneeai.imaging.volume import Volume
from kneeai.ml.contract import (
    ArtifactSource,
    build_result_payload,
    validate_result_payload,
)

REPO = Path(__file__).resolve().parents[1]

LABEL_MAP = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
}
MISSING = (
    Structure.MEDIAL_MENISCUS,
    Structure.LATERAL_MENISCUS,
    Structure.PATELLA,
    Structure.PATELLAR_CARTILAGE,
)
RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def volume_from_nifti_affine(
    labels: np.ndarray, affine_ras: np.ndarray, *, side: Side, provenance: Provenance
) -> Volume:
    """Build a label :class:`Volume` from a RAS affine, converting to LPS."""
    affine = RAS_TO_LPS @ np.asarray(affine_ras, dtype=np.float64)
    rotation_scale = affine[:3, :3]
    spacing = np.linalg.norm(rotation_scale, axis=0)
    direction = rotation_scale / spacing
    # Guard against a left-handed basis, which Volume rejects outright.
    if np.linalg.det(direction) < 0:
        direction[:, 2] *= -1.0
    return Volume(
        array=labels.astype(np.int16),
        spacing=spacing,
        origin=affine[:3, 3],
        direction=direction,
        modality=Modality.MR,
        side=side,
        is_label=True,
        label_map=LABEL_MAP,
        provenance=provenance,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="case_id, default = first val case")
    ap.add_argument("--checkpoint", default="artifacts/models/seg_oaizib/checkpoint.pt")
    ap.add_argument("--out-root", default="artifacts/cases")
    args = ap.parse_args()

    manifest = json.loads((REPO / "data/manifests/oaizib_40.json").read_text())
    records = {r["case_id"]: r for r in manifest["records"]}
    val_split = json.loads((REPO / "data/splits/oaizib_40_val.json").read_text())
    case_id = args.case or val_split["case_ids"][0]
    if case_id not in records:
        raise SystemExit(f"unknown case {case_id}")
    if case_id not in val_split["case_ids"]:
        print(f"NOTE: {case_id} is not in the validation split.")
    rec = records[case_id]

    ckpt = torch.load(REPO / args.checkpoint, map_location="cpu", weights_only=False)
    spacing = float(ckpt["spacing_mm"][0])
    patch = int(ckpt["patch"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"case {case_id}  patient {rec['patient_id']}  side {rec['side']}  KL {rec['kl_grade']}")
    print(f"checkpoint epoch {ckpt['epoch']}  val_dice {ckpt['val_dice_mean']:.4f}")
    print(f"device {device}   inference spacing {spacing} mm   patch {patch}^3")

    pre = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=(spacing,) * 3, mode="bilinear"),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
            NormalizeIntensityd(keys="image", nonzero=True),
        ]
    )
    data = pre({"image": str(REPO / rec["image_path"])})

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=int(ckpt["n_classes"]),
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = data["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            x, roi_size=(patch,) * 3, sw_batch_size=1, predictor=model, overlap=0.5
        )
    probs = torch.softmax(logits, dim=1)[0].cpu()
    pred = probs.argmax(dim=0, keepdim=True)

    # Mean softmax probability of the winning class, per structure. This is a
    # confidence proxy, not an accuracy estimate, and the contract forbids
    # reporting Dice here because there is no ground truth at inference time.
    confidence: dict[Structure, float | None] = {}
    for structure, label in LABEL_MAP.items():
        if structure is Structure.BACKGROUND:
            continue
        sel = pred[0] == label
        confidence[structure] = (
            round(float(probs[label][sel].mean()), 4) if bool(sel.any()) else None
        )

    # Undo preprocessing so the mask lands back on the acquisition grid.
    data["pred"] = MetaTensor(pred, meta=data["image"].meta)
    inverted = Invertd(
        keys="pred", transform=pre, orig_keys="image", nearest_interp=True, to_tensor=True
    )(data)
    labels = np.asarray(inverted["pred"][0], dtype=np.int16)

    native = nib.load(REPO / rec["image_path"])
    if labels.shape != native.shape:
        raise SystemExit(
            f"inverse transform returned {labels.shape}, native grid is {native.shape}"
        )

    provenance = Provenance(
        component="kneeai.ml.infer",
        component_version="0.1.0-local",
        origin=Origin.AI,
        model_id="seg_oaizib",
        model_version="0.1.0-local",
        inputs=(ArtifactRef.from_path("image", REPO / rec["image_path"]),),
        parameters={
            "inference_spacing_mm": [spacing] * 3,
            "roi_size": [patch] * 3,
            "sw_overlap": 0.5,
            "inverse_resampled_to_acquisition_grid": True,
            "interpolation_on_inverse": "nearest",
            "checkpoint_epoch": int(ckpt["epoch"]),
        },
        notes=(
            "Segmentation predicted from 3D DESS knee MR. Inference ran at 1.0 mm "
            "isotropic, then the label map was inverse-resampled back onto the "
            "acquisition grid with nearest-neighbour interpolation."
        ),
    )

    mask = volume_from_nifti_affine(
        labels, native.affine, side=Side(rec["side"]), provenance=provenance
    )

    out_dir = REPO / args.out_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(labels, native.affine, dtype=np.int16),
        out_dir / "segmentation.nii.gz",
    )

    payload = build_result_payload(
        case_id=case_id,
        source=ArtifactSource.AI_PREDICTION,
        mask=mask,
        confidence=confidence,
        confidence_method=(
            "mean softmax probability of the predicted class over that structure's "
            "voxels. A geometric plausibility proxy, NOT an accuracy estimate; no "
            "ground truth exists at inference time."
        ),
        provenance=provenance,
        model_id="seg_oaizib",
        model_version="0.1.0-local",
        architecture="unet",
        model_card="artifacts/models/seg_oaizib/model_card.json",
        structures_expected_but_missing=MISSING,
        warnings=[
            "RESEARCH MODEL, NOT FOR CLINICAL USE. Trained on 30 patients for 40 "
            "epochs on a laptop GPU at 1.0 mm isotropic.",
            "Cartilage accuracy is materially lower than bone; the tibial cartilages "
            "are the weakest structures. See the model card for measured per-structure "
            "Dice.",
            "No meniscus or patella output: OAIZIB-CM contains no such labels.",
            "Every training knee was a right knee. Left-knee performance is unmeasured.",
        ],
    )
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))

    report = validate_result_payload(payload, subject=str(out_dir / "result.json"), base_dir=out_dir)
    print()
    print("structures present:", payload["segmentation"]["structures_present"])
    print("volumes mm3      :", json.dumps(payload["confidence"]["volume_mm3"]))
    print("confidence       :", json.dumps(payload["confidence"]["per_structure"]))
    print("side             :", payload["geometry"]["side"])
    print("spacing_mm       :", payload["geometry"]["spacing_mm"])
    print("orientation      :", payload["geometry"]["orientation"])
    print()
    print("contract validation:", "PASS" if report.valid else "FAIL")
    for e in report.errors:
        print("  ERROR:", e)
    for w in report.warnings:
        print("  warn :", w)
    print("written to", out_dir.relative_to(REPO))


if __name__ == "__main__":
    main()
