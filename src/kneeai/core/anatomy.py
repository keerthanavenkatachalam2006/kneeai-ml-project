"""Anatomical vocabulary: structures, sides, modalities, landmarks.

Every mask channel, mesh, landmark and measurement in the platform refers back
to the identifiers defined here. Free-form strings for anatomy are a common
source of silent mismatch between a segmentation model's label order and a
downstream geometry routine, so the label maps are declared once, here.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Side",
    "Modality",
    "StudyType",
    "Structure",
    "StructureGroup",
    "LandmarkName",
    "BONE_STRUCTURES",
    "CARTILAGE_STRUCTURES",
    "MENISCUS_STRUCTURES",
    "LIGAMENT_STRUCTURES",
    "DEFAULT_LABEL_MAP",
    "structure_group",
    "label_map_for",
    "invert_label_map",
]


class Side(str, Enum):
    """Which knee. Getting this wrong is a wrong-site-surgery class error."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"

    @property
    def sign(self) -> int:
        """+1 for right, -1 for left.

        Used to convert a side-agnostic anatomical convention (e.g. "lateral is
        +x") into a signed direction in the patient frame.
        """
        if self is Side.RIGHT:
            return 1
        if self is Side.LEFT:
            return -1
        raise ValueError("Side must be resolved to LEFT or RIGHT before use in geometry")

    def opposite(self) -> Side:
        if self is Side.LEFT:
            return Side.RIGHT
        if self is Side.RIGHT:
            return Side.LEFT
        return Side.UNKNOWN


class Modality(str, Enum):
    MR = "MR"
    CT = "CT"
    XRAY = "CR"  # DICOM uses CR/DX for radiography
    DX = "DX"
    UNKNOWN = "UNKNOWN"

    @property
    def is_volumetric(self) -> bool:
        return self in (Modality.MR, Modality.CT)


class StudyType(str, Enum):
    """Clinical intent of a study, which drives the pipeline branch taken."""

    KNEE_MR_3D = "knee_mr_3d"
    KNEE_CT = "knee_ct"
    KNEE_XRAY_AP = "knee_xray_ap"
    KNEE_XRAY_LATERAL = "knee_xray_lateral"
    LONG_LEG_XRAY = "long_leg_xray"
    UNKNOWN = "unknown"


class StructureGroup(str, Enum):
    BONE = "bone"
    CARTILAGE = "cartilage"
    MENISCUS = "meniscus"
    LIGAMENT = "ligament"
    OTHER = "other"


class Structure(str, Enum):
    """Segmentable anatomical structures (PRD §11).

    Values are stable identifiers used in filenames, DB rows and JSON. Do not
    rename them without a schema migration.
    """

    BACKGROUND = "background"

    # Bone
    FEMUR = "femur"
    TIBIA = "tibia"
    PATELLA = "patella"
    FIBULA = "fibula"

    # Cartilage
    FEMORAL_CARTILAGE = "femoral_cartilage"
    MEDIAL_TIBIAL_CARTILAGE = "medial_tibial_cartilage"
    LATERAL_TIBIAL_CARTILAGE = "lateral_tibial_cartilage"
    PATELLAR_CARTILAGE = "patellar_cartilage"

    # Meniscus
    MEDIAL_MENISCUS = "medial_meniscus"
    LATERAL_MENISCUS = "lateral_meniscus"

    # Ligament (future — PRD §11)
    ACL = "acl"
    PCL = "pcl"
    MCL = "mcl"
    LCL = "lcl"

    @property
    def group(self) -> StructureGroup:
        return structure_group(self)

    @property
    def display_name(self) -> str:
        overrides = {
            Structure.ACL: "ACL",
            Structure.PCL: "PCL",
            Structure.MCL: "MCL",
            Structure.LCL: "LCL",
        }
        if self in overrides:
            return overrides[self]
        return self.value.replace("_", " ").title()


BONE_STRUCTURES: tuple[Structure, ...] = (
    Structure.FEMUR,
    Structure.TIBIA,
    Structure.PATELLA,
    Structure.FIBULA,
)

CARTILAGE_STRUCTURES: tuple[Structure, ...] = (
    Structure.FEMORAL_CARTILAGE,
    Structure.MEDIAL_TIBIAL_CARTILAGE,
    Structure.LATERAL_TIBIAL_CARTILAGE,
    Structure.PATELLAR_CARTILAGE,
)

MENISCUS_STRUCTURES: tuple[Structure, ...] = (
    Structure.MEDIAL_MENISCUS,
    Structure.LATERAL_MENISCUS,
)

LIGAMENT_STRUCTURES: tuple[Structure, ...] = (
    Structure.ACL,
    Structure.PCL,
    Structure.MCL,
    Structure.LCL,
)


def structure_group(structure: Structure) -> StructureGroup:
    if structure in BONE_STRUCTURES:
        return StructureGroup.BONE
    if structure in CARTILAGE_STRUCTURES:
        return StructureGroup.CARTILAGE
    if structure in MENISCUS_STRUCTURES:
        return StructureGroup.MENISCUS
    if structure in LIGAMENT_STRUCTURES:
        return StructureGroup.LIGAMENT
    return StructureGroup.OTHER


#: Canonical integer label map for the platform's primary MR segmentation task.
#:
#: This ordering is a contract. A checkpoint trained against a different
#: ordering must declare its own map in its model card; inference will refuse to
#: run if the map does not match (see kneeai.ml.registry).
DEFAULT_LABEL_MAP: dict[Structure, int] = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
    Structure.PATELLA: 6,
    Structure.PATELLAR_CARTILAGE: 7,
    Structure.MEDIAL_MENISCUS: 8,
    Structure.LATERAL_MENISCUS: 9,
}


def label_map_for(structures: list[Structure] | tuple[Structure, ...]) -> dict[Structure, int]:
    """Build a contiguous label map for a subset of structures.

    Background is always 0. Order is preserved so the caller controls channel
    order explicitly rather than relying on enum declaration order.
    """
    out: dict[Structure, int] = {Structure.BACKGROUND: 0}
    next_label = 1
    for s in structures:
        if s is Structure.BACKGROUND:
            continue
        if s in out:
            raise ValueError(f"Duplicate structure in label map: {s}")
        out[s] = next_label
        next_label += 1
    return out


def invert_label_map(label_map: dict[Structure, int]) -> dict[int, Structure]:
    inverted: dict[int, Structure] = {}
    for structure, label in label_map.items():
        if label in inverted:
            raise ValueError(
                f"Label {label} assigned to both {inverted[label]} and {structure}; "
                "label maps must be injective."
            )
        inverted[label] = structure
    return inverted


class LandmarkName(str, Enum):
    """Anatomical landmarks (PRD §14).

    Each landmark is a single 3D point in the patient frame with a confidence.
    Axes and planes are *derived* from landmarks, never stored as primary data,
    so a landmark correction propagates deterministically.
    """

    # Femur — proximal
    FEMORAL_HEAD_CENTER = "femoral_head_center"

    # Femur — distal
    MEDIAL_EPICONDYLE = "medial_epicondyle"
    LATERAL_EPICONDYLE = "lateral_epicondyle"
    MEDIAL_SULCUS = "medial_sulcus"
    DISTAL_MEDIAL_CONDYLE = "distal_medial_condyle"
    DISTAL_LATERAL_CONDYLE = "distal_lateral_condyle"
    POSTERIOR_MEDIAL_CONDYLE = "posterior_medial_condyle"
    POSTERIOR_LATERAL_CONDYLE = "posterior_lateral_condyle"
    ANTERIOR_FEMORAL_CORTEX = "anterior_femoral_cortex"
    INTERCONDYLAR_NOTCH = "intercondylar_notch"
    FEMORAL_KNEE_CENTER = "femoral_knee_center"
    FEMORAL_CANAL_ENTRY = "femoral_canal_entry"

    # Tibia — proximal
    TIBIAL_KNEE_CENTER = "tibial_knee_center"
    MEDIAL_PLATEAU_LOW = "medial_plateau_low"
    LATERAL_PLATEAU_LOW = "lateral_plateau_low"
    MEDIAL_PLATEAU_EDGE = "medial_plateau_edge"
    LATERAL_PLATEAU_EDGE = "lateral_plateau_edge"
    TIBIAL_TUBEROSITY = "tibial_tuberosity"
    PCL_INSERTION = "pcl_insertion"
    ANTERIOR_TIBIAL_CORTEX = "anterior_tibial_cortex"
    POSTERIOR_TIBIAL_CORTEX = "posterior_tibial_cortex"

    # Tibia — distal
    ANKLE_CENTER = "ankle_center"
    MEDIAL_MALLEOLUS = "medial_malleolus"
    LATERAL_MALLEOLUS = "lateral_malleolus"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def bone(self) -> Structure:
        if self.value.startswith(("tibial_", "medial_plateau", "lateral_plateau")) or self in (
            LandmarkName.TIBIAL_TUBEROSITY,
            LandmarkName.PCL_INSERTION,
            LandmarkName.ANTERIOR_TIBIAL_CORTEX,
            LandmarkName.POSTERIOR_TIBIAL_CORTEX,
            LandmarkName.ANKLE_CENTER,
            LandmarkName.MEDIAL_MALLEOLUS,
            LandmarkName.LATERAL_MALLEOLUS,
        ):
            return Structure.TIBIA
        return Structure.FEMUR
