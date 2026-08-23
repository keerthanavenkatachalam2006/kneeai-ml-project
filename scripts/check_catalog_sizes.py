"""List every femoral/tibial size the catalogue actually offers."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "backend"))

from kneeai.core.anatomy import Side
from kneeai.implants.catalog import ComponentType
from kneeai.implants.reference import build_reference_catalog

system = build_reference_catalog()
family = next(iter(system.families.values()))
for ctype in (ComponentType.FEMORAL, ComponentType.TIBIAL_BASEPLATE):
    comps = family.by_type(ctype, side=Side.RIGHT)
    print(f"\n=== {ctype.value} ({len(comps)} sizes, right) ===")
    for c in comps:
        print(
            f"  size {c.size_label:>3s}  ml={c.ml_mm:6.1f}  ap={c.ap_mm:6.1f}"
            f"  envelope_ap={c.envelope_ap_mm}  id={c.component_id}"
        )
