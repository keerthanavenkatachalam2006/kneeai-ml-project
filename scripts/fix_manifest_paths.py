"""Fix manifest paths to point to actual data location."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

for manifest_name in ("oaizib_60.json", "oaizib_40.json"):
    manifest_path = REPO / "data" / "manifests" / manifest_name
    if not manifest_path.exists():
        continue
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records", [])
    fixed = 0
    for r in records:
        if r.get("label_path") and "data/raw/oaizib/" in r["label_path"]:
            r["label_path"] = r["label_path"].replace("data/raw/oaizib/", "data/processed/extracted/")
            fixed += 1
        if r.get("image_path") and "data/raw/oaizib/" in r["image_path"]:
            r["image_path"] = r["image_path"].replace("data/raw/oaizib/", "data/processed/extracted/")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest_name}: fixed {fixed} label paths out of {len(records)} records")
    if records:
        print(f"  sample label: {records[0].get('label_path')}")
        print(f"  sample image: {records[0].get('image_path')}")
