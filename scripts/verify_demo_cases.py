"""Verify the two demo cases are complete and consistent: one OA, one not."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def morphometric_score(measurements: dict) -> tuple[float, dict]:
    """Mirror of the UI's OA score, so both agree on the verdict."""
    lookup = {}
    for m in measurements.get("measurements", []):
        v = (m.get("quantity") or {}).get("value")
        if v is not None:
            lookup[m.get("name")] = float(v)

    fc = lookup.get("femoral_cartilage_mean_thickness")
    mtc = lookup.get("medial_tibial_cartilage_mean_thickness")
    ltc = lookup.get("lateral_tibial_cartilage_mean_thickness")
    jlca = lookup.get("jlca")

    score = 0.0
    max_score = 0.0
    if mtc is not None:
        max_score += 22
        score += 22 if mtc < 0.9 else 14 if mtc < 1.4 else 6 if mtc < 1.6 else 0
    if fc is not None:
        max_score += 18
        score += 18 if fc < 1.2 else 11 if fc < 1.8 else 4 if fc < 2.1 else 0
    if ltc is not None:
        max_score += 12
        score += 12 if ltc < 1.0 else 7 if ltc < 1.5 else 0
    if jlca is not None:
        max_score += 18
        a = abs(jlca)
        score += 18 if a >= 8 else 12 if a >= 5 else 5 if a >= 3 else 0
    if mtc is not None and ltc is not None and ltc > 0:
        max_score += 14
        asym = abs(mtc - ltc) / max(mtc, ltc) * 100
        score += 14 if asym >= 40 else 8 if asym >= 25 else 3 if asym >= 15 else 0

    prob = score / max_score if max_score else 0.0
    return prob, {"fc": fc, "mtc": mtc, "ltc": ltc, "jlca": jlca}


def check(case_id: str) -> dict:
    d = REPO / "artifacts" / "cases" / case_id
    out = {"case_id": case_id, "ok": True, "problems": []}

    required = ["result.json", "measurements.json", "implants.json", "plan.json"]
    for f in required:
        if not (d / f).exists():
            out["problems"].append(f"missing {f}")
            out["ok"] = False

    mdir = d / "meshes" / "manifest.json"
    if not mdir.exists():
        out["problems"].append("missing mesh manifest")
        out["ok"] = False
        return out

    manifest = json.loads(mdir.read_text())
    meshes = manifest["meshes"]
    out["n_meshes"] = len(meshes)

    # Every declared mesh must exist on disk, or the viewer shows a hole.
    for m in meshes:
        if not (d / "meshes" / m["path"]).exists():
            out["problems"].append(f"mesh file missing: {m['path']}")
            out["ok"] = False

    cats = {}
    for m in meshes:
        cats.setdefault(m["category"], []).append(m["structure"])
    out["categories"] = {k: sorted(v) for k, v in cats.items()}

    if "component" not in cats:
        out["problems"].append("no implant components exported")
        out["ok"] = False

    non_watertight = [m["structure"] for m in meshes if not m.get("is_watertight")]
    out["non_watertight"] = non_watertight

    if (d / "measurements.json").exists():
        prob, feats = morphometric_score(json.loads((d / "measurements.json").read_text()))
        out["oa_probability"] = round(prob, 3)
        out["oa_detected"] = prob >= 0.50
        out["features"] = {k: (round(v, 2) if v is not None else None) for k, v in feats.items()}

    if (d / "oa.json").exists():
        oa = json.loads((d / "oa.json").read_text())
        out["trained_model_oa"] = oa.get("oa_detected")
        out["trained_model_prob"] = oa.get("probability")

    return out


cases = sys.argv[1:] or ["oaizib_415", "oaizib_496"]
results = [check(c) for c in cases]

for r in results:
    print(f"=== {r['case_id']} ===")
    print(f"  complete        : {'YES' if r['ok'] else 'NO'}")
    print(f"  meshes          : {r.get('n_meshes')}")
    for cat, structs in (r.get("categories") or {}).items():
        print(f"    {cat:10s}: {', '.join(structs)}")
    if r.get("non_watertight"):
        print(f"  not watertight  : {', '.join(r['non_watertight'])}")
    else:
        print("  not watertight  : none")
    print(f"  OA (morphometric): {r.get('oa_probability')} -> {'DETECTED' if r.get('oa_detected') else 'not detected'}")
    print(f"  features         : {r.get('features')}")
    if r.get("problems"):
        for p in r["problems"]:
            print(f"  PROBLEM: {p}")
    print()

detected = [r["case_id"] for r in results if r.get("oa_detected")]
clean = [r["case_id"] for r in results if not r.get("oa_detected")]
print(f"OA positive : {detected or 'NONE'}")
print(f"OA negative : {clean or 'NONE'}")
if detected and clean:
    print("\nGood: you have one of each for the contrast.")
else:
    print("\nWARNING: you do not have one of each.")
