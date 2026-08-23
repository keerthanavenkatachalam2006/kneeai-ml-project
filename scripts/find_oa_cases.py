"""Find cases with high KL grades that we can run the pipeline on."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "processed" / "extracted"

m = json.loads((REPO / "data" / "manifests" / "oaizib_60.json").read_text())
high_kl = [r for r in m["records"] if (r.get("kl_grade") or 0) >= 3]
print(f"Cases with KL >= 3: {len(high_kl)}")

available = []
for r in sorted(high_kl, key=lambda x: -x["kl_grade"]):
    cid = r["case_id"].replace("oaizib-", "oaizib_")
    # Check if image exists
    exists = False
    for subdir in ("imagesTs", "imagesTr"):
        if (DATA_DIR / subdir / f"{cid}_0000.nii.gz").exists():
            exists = True
            break
    # Check if already processed
    already = (REPO / "artifacts" / "cases" / cid / "oa.json").exists()
    status = "DONE" if already else ("AVAILABLE" if exists else "MISSING")
    print(f"  {cid}  KL={r['kl_grade']}  split={r['split']}  {status}")
    if exists:
        available.append({"case_id": cid, "kl_grade": r["kl_grade"], "already_done": already})

print(f"\nAvailable high-KL cases: {len(available)}")
print("Best to run (KL 4, not yet processed):")
for c in available:
    if c["kl_grade"] == 4 and not c["already_done"]:
        print(f"  {c['case_id']} KL={c['kl_grade']}")
