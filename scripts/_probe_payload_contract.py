"""Check that every field the dashboard reads actually exists in the API payload.

Motivation: ImplantPanel was reading ``selected.configuration_label`` and
``selected.fit_score``, neither of which the API ever emitted — the panel silently
rendered every metric as an em dash. TypeScript could not catch it because the
interfaces described the fields the panel wanted rather than the fields the API sends.

This walks the real payload for a case, collects the key paths that exist, then greps
the components for property accesses and reports the ones with no counterpart. It is a
lint, not a proof: it cannot see dynamic access, and it will flag locals that happen to
share a name with nothing. Read it as a list of things to check.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = sys.argv[1] if len(sys.argv) > 1 else "oaizib-417"
BASE = "http://127.0.0.1:8000"

with urllib.request.urlopen(f"{BASE}/api/cases/{CASE}", timeout=30) as r:
    summary = json.load(r)
with urllib.request.urlopen(f"{BASE}/api/cases/{CASE}/robot/commands", timeout=30) as r:
    summary["_robot"] = json.load(r)
with urllib.request.urlopen(f"{BASE}/api/cases/{CASE}/simulation/state", timeout=30) as r:
    summary["_state"] = json.load(r)
# The dashboard also reads three payloads that are not part of the case summary.
for name, url in (
    ("_list", "/api/cases"),
    ("_model", "/api/model"),
    ("_slices", f"/api/cases/{CASE}/slices"),
):
    with urllib.request.urlopen(f"{BASE}{url}", timeout=30) as r:
        summary[name] = json.load(r)
# And the controller dictionary returned by the approve/run POSTs, which the GET on
# /simulation/state only produces once a controller exists for the case.
try:
    from kneeai.robotics.safety import SimulationController  # noqa: E402

    sys.path.insert(0, str(ROOT / "src"))
    summary["_controller"] = SimulationController(case_id=CASE, plan_id="probe").to_dict()
except Exception as exc:  # noqa: BLE001
    print(f"(could not build a controller dict for comparison: {exc})")


def keys_of(obj: object, depth: int = 0, acc: set[str] | None = None) -> set[str]:
    """Every key name appearing anywhere in the payload."""
    acc = set() if acc is None else acc
    if depth > 6:
        return acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(k)
            keys_of(v, depth + 1, acc)
    elif isinstance(obj, list):
        for v in obj[:3]:
            keys_of(v, depth + 1, acc)
    return acc


available = keys_of(summary)
print(f"payload exposes {len(available)} distinct key names")

# Property accesses in the components, restricted to ones that look like payload
# fields: snake_case, which is the Python side's convention. React and three.js
# properties are camelCase, so this filter removes almost all the noise.
snake = re.compile(r"\.([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")

IGNORE = {
    # three.js / DOM / library members that happen to be snake_case-ish, plus our own
    # helpers and locals.
    "current_target",
    "set_precision",
}

report: dict[str, set[str]] = {}
for path in sorted((ROOT / "apps/web").glob("**/*.tsx")) + sorted(
    (ROOT / "apps/web/lib").glob("*.ts")
):
    if "node_modules" in str(path) or ".next" in str(path):
        continue
    text = path.read_text()
    used = {m.group(1) for m in snake.finditer(text)} - IGNORE
    missing = {u for u in used if u not in available}
    if missing:
        report[str(path.relative_to(ROOT))] = missing

if not report:
    print("\nno unresolved snake_case field accesses")
else:
    print("\nfields read by the UI with no counterpart in the payload:")
    for file, fields in report.items():
        print(f"  {file}")
        for f in sorted(fields):
            print(f"      {f}")

print("\n(names below are declared in types.ts but never sent by the API)")
types = (ROOT / "apps/web/lib/types.ts").read_text()
declared = {
    m.group(1)
    for m in re.finditer(r"^\s{2,}([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\??:", types, re.M)
}
orphans = sorted(declared - available)
for name in orphans:
    print(f"  {name}")
