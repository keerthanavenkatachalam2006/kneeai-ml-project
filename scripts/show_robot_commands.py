"""Print the robot command sequence so the simulation UI can be built on real poses."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "backend"))

case_id = sys.argv[1] if len(sys.argv) > 1 else "oaizib_496"
case_dir = REPO / "artifacts" / "cases" / case_id

from kneeai.robotics import build_command_sequence

plan = json.loads((case_dir / "plan.json").read_text())
sim_path = case_dir / "simulation.json"
man_path = case_dir / "meshes" / "manifest.json"
simulation = json.loads(sim_path.read_text()) if sim_path.exists() else None
manifest = json.loads(man_path.read_text()) if man_path.exists() else None

seq = build_command_sequence(plan, mesh_manifest=manifest, simulation=simulation).to_dict()

print(f"=== {case_id} ===")
print(f"commands   : {seq['n_commands']}")
print(f"frame      : {seq['frame']}  units={seq['units']}")
print(f"executable : {seq['executable']}   hardware_control={seq['hardware_control']}")

b = seq.get("safety_boundary")
if b:
    print(f"\nsafety boundary ({b['type']}):")
    print(f"  min  {[round(v,1) for v in b['min_mm']]}")
    print(f"  max  {[round(v,1) for v in b['max_mm']]}")
    print(f"  size {[round(v,1) for v in b['size_mm']]}  margin={b['margin_mm']} mm")

print("\ncommands:")
for c in seq["commands"]:
    pose = c.get("pose")
    if pose:
        pos = [round(v, 1) for v in pose["position_mm"]]
        axis = [round(v, 3) for v in pose["axis"]]
        print(f"  {c['sequence']:2d}. {c['action']:22s} target={c['target']:24s} pos={pos} axis={axis}")
    else:
        print(f"  {c['sequence']:2d}. {c['action']:22s} target={c['target']:24s} (no pose)")

print(f"\nwarnings: {len(seq.get('warnings', []))}")
for w in seq.get("warnings", []):
    print(f"  - {w[:110]}")
