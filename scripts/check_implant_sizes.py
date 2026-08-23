"""Check what dimension fields the implant candidates actually carry."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
case_id = sys.argv[1] if len(sys.argv) > 1 else "oaizib_496"

data = json.loads((REPO / "artifacts" / "cases" / case_id / "implants.json").read_text())
cands = data.get("candidates") or []
print(f"=== {case_id}: {len(cands)} candidates ===\n")

sel = data.get("selected")
print("SELECTED keys:", sorted(sel.keys()) if sel else None)
if sel:
    for part in ("femoral_fit", "tibial_fit"):
        fit = sel.get(part)
        print(f"\n  {part}:", sorted(fit.keys()) if isinstance(fit, dict) else fit)
        if isinstance(fit, dict):
            for k in ("component_ml_mm", "component_ap_mm", "ml_mismatch_mm", "ap_mismatch_mm", "size_label", "score"):
                print(f"     {k} = {fit.get(k)}")

print("\n--- per candidate femoral/tibial dims ---")
for i, c in enumerate(cands):
    cfg = c.get("configuration") or {}
    f = c.get("femoral_fit") or {}
    t = c.get("tibial_fit") or {}
    print(
        f"  [{i}] {cfg.get('label') or cfg.get('configuration_id')}"
        f"  score={c.get('score')}"
        f"  fem_ml={f.get('component_ml_mm')} fem_ap={f.get('component_ap_mm')}"
        f"  fem_size={f.get('size_label')} tib_size={t.get('size_label')}"
    )
