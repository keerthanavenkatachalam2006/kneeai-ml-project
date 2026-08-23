"""Print the robotic readiness breakdown for a case."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
case_id = sys.argv[1] if len(sys.argv) > 1 else "oaizib_496"
p = REPO / "artifacts" / "cases" / case_id / "robotic_readiness.json"
if not p.exists():
    print(f"no robotic_readiness.json for {case_id}")
    raise SystemExit(1)

d = json.loads(p.read_text())
print(f"=== {case_id} ===")
print(f"score   : {d.get('percentage')}%  band={d.get('band')}")
print(f"raw     : {d.get('raw_weighted_percentage')}%   ceiling={d.get('ceiling_pct')}%  capped={d.get('capped')}")
print("\ncriteria:")
for c in d.get("criteria", []):
    print(
        f"  {c['name']:30s} weight={c['weight']:.2f}  "
        f"score={c['score_pct']:6.1f}%  -> {c['weighted_contribution_pct']:5.1f} pts"
    )
    print(f"      {c['detail'][:110]}")
print("\nblockers:")
for b in d.get("blockers", []):
    print(f"  {b['code']}  caps at {b['ceiling_pct']}%")
    print(f"      {b['message'][:140]}")
if not d.get("blockers"):
    print("  none")
