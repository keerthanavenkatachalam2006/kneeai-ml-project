"""Rebuild the manifest over every OAIZIB-CM case currently on disk.

The first training run used 40 cases; 60 were actually downloaded. The disk is
full so more data cannot be fetched — but the 20 already sitting unused are free
training signal, so this regenerates the manifest and a KL-stratified,
patient-level split across all of them.

``build_manifest`` handles scanning, splitting, integrity verification and
writing, and names its outputs after the case count: ``oaizib_60.json`` plus
``oaizib_60_{train,val}.json``.
"""

from __future__ import annotations

import json

from kneeai.datasets.oaizib import build_manifest


def main() -> None:
    summary = build_manifest(
        "data/raw/oaizib",
        val_fraction=0.20,
        seed=42,
        stratify_by_kl=True,
        verify_masks=True,
    )
    printable = {
        k: v for k, v in summary.items() if k not in ("records", "dataset_contract")
    }
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
