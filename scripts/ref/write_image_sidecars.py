"""Write imaging sidecars for the downloaded OAIZIB-CM volumes.

NIfTI carries no modality or laterality tag, so ``imaging.io.read_volume`` loads
these scans as ``Modality.UNKNOWN`` / ``Side.UNKNOWN``. The predictor then refuses
to run, and it is right to: applying an MR-trained model to a volume of unknown
modality is exactly the cross-modality misuse ``ml/infer.py`` guards against.

The information exists — it is in the dataset manifest, sourced from OAIZIB-CM's
own ``kneeSideInfo`` and ``subInfo`` tables. This writes it into the sidecar the
loader already looks for, rather than patching the loader or hardcoding a default.

Every OAIZIB-CM knee is a RIGHT knee, but the side is read per-record rather than
assumed, so this stays correct if a left-knee dataset is added later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kneeai.imaging.io import sidecar_path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/oaizib_60.json")
    args = ap.parse_args()

    manifest = json.loads((REPO / args.manifest).read_text())
    label_map = manifest["label_map"]

    written = 0
    for rec in manifest["records"]:
        image = REPO / rec["image_path"]
        if not image.exists():
            continue

        # Image sidecar: modality + laterality, no label map.
        payload = {
            "modality": rec["modality"],
            "side": rec["side"],
            "is_label": False,
            "source": {
                "dataset_id": rec["dataset_id"],
                "case_id": rec["case_id"],
                "patient_id": rec["patient_id"],
                "kl_grade": rec.get("kl_grade"),
                "age": rec.get("age"),
                "sex": rec.get("sex"),
                "bmi": rec.get("bmi"),
                "licence": manifest.get("licence"),
            },
        }
        sidecar_path(image).write_text(json.dumps(payload, indent=2))
        written += 1

        # Label sidecar: same, plus the label map so masks load as label volumes.
        label = REPO / rec["label_path"] if rec.get("label_path") else None
        if label and label.exists():
            lbl_payload = dict(payload)
            lbl_payload["is_label"] = True
            lbl_payload["label_map"] = label_map
            sidecar_path(label).write_text(json.dumps(lbl_payload, indent=2))
            written += 1

    print(f"sidecars written: {written}")
    print(f"modality        : MR")
    print(f"sides seen      : {sorted({r['side'] for r in manifest['records']})}")

    # Prove the loader now resolves both fields.
    from kneeai.imaging.io import read_volume

    first = REPO / manifest["records"][0]["image_path"]
    v = read_volume(first)
    print()
    print(f"verify {first.name}")
    print(f"  modality : {v.modality.value}")
    print(f"  side     : {v.side.value}")
    print(f"  spacing  : {[round(float(s), 4) for s in v.spacing]}")
    print(f"  shape    : {v.array.shape}")


if __name__ == "__main__":
    main()
