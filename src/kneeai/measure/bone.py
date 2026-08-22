"""Femoral and tibial bone measurements (PRD §13, §20).

These are the numbers implant sizing actually runs on. Two things make them
non-trivial, and both are handled explicitly here.

**Direction matters more than the mesh.** "Femoral AP size" means the extent from
the posterior condyles to the anterior cortex measured *along the anteroposterior
axis of the patient frame*. Measured along the image y-axis instead, a knee
rotated 15° in the scanner reads several millimetres short — enough to change the
recommended component size. Every extent here is projected onto an explicit
anatomical direction.

**Region matters.** The femoral ML width is the width of the *distal* femur, not
of the whole visible bone. Including the shaft or the metaphyseal flare changes
the answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import LandmarkName, Side, Structure
from kneeai.core.errors import MeasurementError
from kneeai.core.geometry import (
    Plane,
    angle_between_deg,
    bounding_extent,
    fit_plane,
    signed_angle_in_plane_deg,
    unit,
)
from kneeai.core.units import Unit
from kneeai.landmarks.frame import PatientFrame
from kneeai.landmarks.types import LandmarkSet
from kneeai.logging import get_logger
from kneeai.measure.types import Measurement, MeasurementName, make_measurement
from kneeai.recon.mesh import AnatomicalMesh, MeshSet

__all__ = ["measure_femur", "measure_tibia", "distal_femoral_region", "plateau_region"]

log = get_logger("measure.bone")


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------


def distal_femoral_region(
    femur: AnatomicalMesh,
    frame: PatientFrame,
    *,
    height_to_width_ratio: float = 0.62,
    probe_slab_mm: float = 15.0,
) -> NDArray[np.float64]:
    """Vertices of the distal (condylar) femur.

    Uses the same width-scaled definition as landmark detection, so a measurement
    and the landmark it is compared against refer to the same region.
    """
    vertices = femur.vertices
    superior = frame.superior
    proj = vertices @ superior
    distal = float(proj.min())
    slab = vertices[proj <= distal + probe_slab_mm]
    if slab.shape[0] < 20:
        raise MeasurementError("too few distal femoral vertices to size the condylar region")
    width = bounding_extent(slab, frame.right)
    height = height_to_width_ratio * width
    return vertices[proj <= distal + height]


def plateau_region(
    tibia: AnatomicalMesh,
    frame: PatientFrame,
    *,
    fraction: float = 0.14,
) -> NDArray[np.float64]:
    """Vertices of the proximal tibia (plateau)."""
    vertices = tibia.vertices
    proj = vertices @ frame.superior
    lo, hi = float(proj.min()), float(proj.max())
    span = hi - lo
    if span <= 0:  # pragma: no cover - degenerate
        raise MeasurementError("tibia mesh has zero superoinferior extent")
    selected = vertices[proj >= hi - fraction * span]
    if selected.shape[0] < 50:
        raise MeasurementError("too few proximal tibial vertices to define the plateau")
    return selected


def plateau_articular_surface(
    tibia: AnatomicalMesh,
    frame: PatientFrame,
    *,
    fraction: float = 0.16,
    normal_threshold: float = 0.30,
) -> NDArray[np.float64]:
    """Vertices of the tibial articular surface, as one connected patch.

    Selection is by *face*: proximal faces whose normal points superiorly. The
    resulting patch is then split into connected components and only the largest
    is kept.

    The connectivity step is what excludes the tibial tuberosity. The tuberosity's
    superior cap is also a proximal, upward-facing patch, and a slab-based
    selection includes it — which inflates the tibial anteroposterior depth by
    roughly a centimetre, because the tuberosity sits well anterior to the plateau
    rim. It is a *separate* patch, disconnected from the articular surface by the
    anterior cortex, so taking the largest connected component removes it without
    needing a hand-tuned anterior cutoff.
    """
    mesh = tibia.mesh
    vertices = tibia.vertices
    proj_v = vertices @ frame.superior
    lo, hi = float(proj_v.min()), float(proj_v.max())
    span = hi - lo
    if span <= 0:  # pragma: no cover - degenerate
        raise MeasurementError("tibia mesh has zero superoinferior extent")

    face_centres = np.asarray(mesh.triangles_center, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    proximal = (face_centres @ frame.superior) >= hi - fraction * span
    upward = (face_normals @ frame.superior) >= normal_threshold
    selected = np.where(proximal & upward)[0]

    if selected.size < 50:
        log.warning(
            "too few upward-facing proximal tibial faces; falling back to the "
            "vertex slab, which may include the tibial tuberosity",
            fields={"n_faces": int(selected.size)},
        )
        return plateau_region(tibia, frame, fraction=fraction)

    patch = mesh.submesh([selected], append=True)
    try:
        components = patch.split(only_watertight=False)
    except Exception:  # noqa: BLE001 - trimesh splitting can fail on odd patches
        components = []

    if len(components) > 1:
        largest = max(components, key=lambda c: float(c.area))
        log.info(
            "articular surface patch split; keeping the largest component",
            fields={
                "n_components": len(components),
                "kept_area_fraction": round(float(largest.area) / float(patch.area), 4),
            },
        )
        patch = largest

    result = np.asarray(patch.vertices, dtype=np.float64)
    if result.shape[0] < 30:
        raise MeasurementError(
            f"the tibial articular surface patch has only {result.shape[0]} vertices"
        )
    return result


def _compartment_split(
    points: NDArray[np.float64], frame: PatientFrame
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split into (medial, lateral) about the mediolateral midline of the region."""
    proj = points @ frame.medial
    midline = (float(proj.min()) + float(proj.max())) / 2.0
    medial = points[proj > midline]
    lateral = points[proj <= midline]
    if medial.shape[0] < 10 or lateral.shape[0] < 10:
        raise MeasurementError(
            f"compartment split produced {medial.shape[0]} medial / "
            f"{lateral.shape[0]} lateral vertices"
        )
    return medial, lateral


# ---------------------------------------------------------------------------
# Femur
# ---------------------------------------------------------------------------


def measure_femur(
    meshes: MeshSet,
    landmarks: LandmarkSet,
    frame: PatientFrame,
) -> list[Measurement]:
    """Femoral sizing, rotation and asymmetry measurements."""
    femur = meshes.require(Structure.FEMUR)
    out: list[Measurement] = []

    condyles = distal_femoral_region(femur, frame)

    # --- mediolateral width of the distal femur ---
    ml_width = bounding_extent(condyles, frame.right)
    out.append(
        make_measurement(
            MeasurementName.FEMORAL_ML_WIDTH,
            ml_width,
            Unit.MM,
            method=(
                "maximum extent of the distal femoral region projected onto the "
                "patient-frame mediolateral axis"
            ),
            structures=(Structure.FEMUR,),
        )
    )

    # --- transepicondylar width (landmark-based, for comparison) ---
    if landmarks.has(LandmarkName.MEDIAL_EPICONDYLE, LandmarkName.LATERAL_EPICONDYLE):
        medial_epi = landmarks.position(LandmarkName.MEDIAL_EPICONDYLE)
        lateral_epi = landmarks.position(LandmarkName.LATERAL_EPICONDYLE)
        tea_vector = medial_epi - lateral_epi
        # Projected onto the mediolateral axis rather than taken as the raw
        # distance: the epicondyles are not exactly level, and the raw distance
        # includes that superoinferior offset.
        tea_width = abs(float(np.dot(tea_vector, frame.right)))
        out.append(
            make_measurement(
                MeasurementName.TRANSEPICONDYLAR_WIDTH,
                tea_width,
                Unit.MM,
                method=(
                    "surgical transepicondylar axis, projected onto the "
                    "mediolateral axis"
                ),
                landmarks=(
                    LandmarkName.MEDIAL_EPICONDYLE,
                    LandmarkName.LATERAL_EPICONDYLE,
                ),
                structures=(Structure.FEMUR,),
            )
        )

    # --- anteroposterior size ---
    # Posterior condyle to anterior cortex, along the AP axis. This is the
    # dimension femoral component sizing is driven by.
    ap_size = bounding_extent(condyles, frame.anterior)
    out.append(
        make_measurement(
            MeasurementName.FEMORAL_AP_SIZE,
            ap_size,
            Unit.MM,
            method=(
                "extent of the distal femoral region along the anteroposterior "
                "axis (posterior condyles to anterior cortex)"
            ),
            structures=(Structure.FEMUR,),
        )
    )

    medial_condyle, lateral_condyle = _compartment_split(condyles, frame)
    out.append(
        make_measurement(
            MeasurementName.FEMORAL_MEDIAL_AP,
            bounding_extent(medial_condyle, frame.anterior),
            Unit.MM,
            method="anteroposterior extent of the medial condyle",
            structures=(Structure.FEMUR,),
        )
    )
    out.append(
        make_measurement(
            MeasurementName.FEMORAL_LATERAL_AP,
            bounding_extent(lateral_condyle, frame.anterior),
            Unit.MM,
            method="anteroposterior extent of the lateral condyle",
            structures=(Structure.FEMUR,),
        )
    )

    # --- aspect ratio ---
    # The single most useful number for implant fit: a patient whose femur is
    # narrow for its AP depth will overhang on any component sized by AP alone.
    if ap_size > 1e-6:
        out.append(
            make_measurement(
                MeasurementName.FEMORAL_ASPECT_RATIO,
                ml_width / ap_size,
                Unit.RATIO,
                method="femoral mediolateral width divided by anteroposterior size",
                structures=(Structure.FEMUR,),
            )
        )

    # --- condylar twist angle (TEA vs posterior condylar axis) ---
    if landmarks.has(
        LandmarkName.MEDIAL_EPICONDYLE,
        LandmarkName.LATERAL_EPICONDYLE,
        LandmarkName.POSTERIOR_MEDIAL_CONDYLE,
        LandmarkName.POSTERIOR_LATERAL_CONDYLE,
    ):
        tea = landmarks.position(LandmarkName.MEDIAL_EPICONDYLE) - landmarks.position(
            LandmarkName.LATERAL_EPICONDYLE
        )
        pca = landmarks.position(
            LandmarkName.POSTERIOR_MEDIAL_CONDYLE
        ) - landmarks.position(LandmarkName.POSTERIOR_LATERAL_CONDYLE)
        # Measured in the axial plane: this is a rotational quantity, and the raw
        # 3D angle between the axes conflates rotation with their height offset.
        twist = abs(signed_angle_in_plane_deg(pca, tea, frame.superior))
        out.append(
            make_measurement(
                MeasurementName.CONDYLAR_TWIST_ANGLE,
                twist,
                Unit.DEG,
                method=(
                    "angle between the posterior condylar axis and the surgical "
                    "transepicondylar axis, projected into the axial plane"
                ),
                landmarks=(
                    LandmarkName.MEDIAL_EPICONDYLE,
                    LandmarkName.LATERAL_EPICONDYLE,
                    LandmarkName.POSTERIOR_MEDIAL_CONDYLE,
                    LandmarkName.POSTERIOR_LATERAL_CONDYLE,
                ),
                structures=(Structure.FEMUR,),
                sign_convention=(
                    "reported unsigned; the transepicondylar axis is externally "
                    "rotated relative to the posterior condylar axis in the "
                    "normal knee"
                ),
            )
        )

    # --- distal condylar offset ---
    if landmarks.has(
        LandmarkName.DISTAL_MEDIAL_CONDYLE, LandmarkName.DISTAL_LATERAL_CONDYLE
    ):
        distal_medial = landmarks.position(LandmarkName.DISTAL_MEDIAL_CONDYLE)
        distal_lateral = landmarks.position(LandmarkName.DISTAL_LATERAL_CONDYLE)
        offset = float(np.dot(distal_lateral - distal_medial, frame.superior))
        out.append(
            make_measurement(
                MeasurementName.DISTAL_CONDYLAR_OFFSET,
                offset,
                Unit.MM,
                method=(
                    "superoinferior difference between the distal-most points of "
                    "the lateral and medial condyles"
                ),
                landmarks=(
                    LandmarkName.DISTAL_MEDIAL_CONDYLE,
                    LandmarkName.DISTAL_LATERAL_CONDYLE,
                ),
                structures=(Structure.FEMUR,),
                sign_convention=(
                    "positive means the lateral condyle is proximal to the medial "
                    "condyle, i.e. medial condylar prominence (valgus-producing "
                    "distal femoral geometry)"
                ),
            )
        )

    # --- posterior condylar offset ---
    if landmarks.has(
        LandmarkName.POSTERIOR_MEDIAL_CONDYLE, LandmarkName.POSTERIOR_LATERAL_CONDYLE
    ):
        post_medial = landmarks.position(LandmarkName.POSTERIOR_MEDIAL_CONDYLE)
        post_lateral = landmarks.position(LandmarkName.POSTERIOR_LATERAL_CONDYLE)
        offset = float(np.dot(post_medial - post_lateral, -frame.anterior))
        out.append(
            make_measurement(
                MeasurementName.POSTERIOR_CONDYLAR_OFFSET,
                offset,
                Unit.MM,
                method=(
                    "anteroposterior difference between the posterior-most points "
                    "of the medial and lateral condyles"
                ),
                landmarks=(
                    LandmarkName.POSTERIOR_MEDIAL_CONDYLE,
                    LandmarkName.POSTERIOR_LATERAL_CONDYLE,
                ),
                structures=(Structure.FEMUR,),
                sign_convention=(
                    "positive means the medial condyle extends further posteriorly "
                    "than the lateral condyle"
                ),
            )
        )

    # --- anatomical vs mechanical femoral axis ---
    if landmarks.has(
        LandmarkName.FEMORAL_HEAD_CENTER,
        LandmarkName.FEMORAL_CANAL_ENTRY,
        LandmarkName.FEMORAL_KNEE_CENTER,
    ):
        hip = landmarks.position(LandmarkName.FEMORAL_HEAD_CENTER)
        knee = landmarks.position(LandmarkName.FEMORAL_KNEE_CENTER)
        canal = landmarks.position(LandmarkName.FEMORAL_CANAL_ENTRY)
        mechanical = hip - knee
        anatomical = canal - knee
        angle = abs(signed_angle_in_plane_deg(anatomical, mechanical, frame.anterior))
        caveat = ""
        if float(np.linalg.norm(anatomical)) < 80.0:
            caveat = (
                "the femoral canal reference is less than 80 mm from the knee "
                "centre, so the anatomical axis is estimated over a short segment "
                "and this angle carries several degrees of uncertainty"
            )
        out.append(
            make_measurement(
                MeasurementName.FEMORAL_ANATOMICAL_MECHANICAL_ANGLE,
                angle,
                Unit.DEG,
                method=(
                    "coronal-plane angle between the femoral anatomical (shaft) "
                    "axis and the femoral mechanical axis"
                ),
                landmarks=(
                    LandmarkName.FEMORAL_HEAD_CENTER,
                    LandmarkName.FEMORAL_CANAL_ENTRY,
                    LandmarkName.FEMORAL_KNEE_CENTER,
                ),
                structures=(Structure.FEMUR,),
                caveat=caveat,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Tibia
# ---------------------------------------------------------------------------


def measure_tibia(
    meshes: MeshSet,
    landmarks: LandmarkSet,
    frame: PatientFrame,
) -> list[Measurement]:
    """Tibial sizing and plateau geometry measurements."""
    tibia = meshes.require(Structure.TIBIA)
    out: list[Measurement] = []

    # Sizing is taken on the articular surface, not on a proximal slab. The
    # tibial component sits on the resected plateau, so the plateau footprint is
    # the dimension that matters — and a slab would include the tibial tuberosity.
    plateau = plateau_articular_surface(tibia, frame)

    ml_width = bounding_extent(plateau, frame.right)
    out.append(
        make_measurement(
            MeasurementName.TIBIAL_ML_WIDTH,
            ml_width,
            Unit.MM,
            method=(
                "mediolateral extent of the tibial articular surface (largest "
                "connected upward-facing proximal patch)"
            ),
            structures=(Structure.TIBIA,),
        )
    )

    ap_depth = bounding_extent(plateau, frame.anterior)
    out.append(
        make_measurement(
            MeasurementName.TIBIAL_AP_DEPTH,
            ap_depth,
            Unit.MM,
            method=(
                "anteroposterior extent of the tibial articular surface, excluding "
                "the tibial tuberosity"
            ),
            structures=(Structure.TIBIA,),
        )
    )

    medial_plateau, lateral_plateau = _compartment_split(plateau, frame)
    out.append(
        make_measurement(
            MeasurementName.TIBIAL_MEDIAL_AP,
            bounding_extent(medial_plateau, frame.anterior),
            Unit.MM,
            method="anteroposterior extent of the medial tibial plateau",
            structures=(Structure.TIBIA,),
        )
    )
    out.append(
        make_measurement(
            MeasurementName.TIBIAL_LATERAL_AP,
            bounding_extent(lateral_plateau, frame.anterior),
            Unit.MM,
            method="anteroposterior extent of the lateral tibial plateau",
            structures=(Structure.TIBIA,),
        )
    )

    if ap_depth > 1e-6:
        out.append(
            make_measurement(
                MeasurementName.TIBIAL_ASPECT_RATIO,
                ml_width / ap_depth,
                Unit.RATIO,
                method="tibial mediolateral width divided by anteroposterior depth",
                structures=(Structure.TIBIA,),
            )
        )

    # --- posterior tibial slope ---
    plateau_plane, plane_rms = fit_articular_plane(tibia, frame)
    slope = posterior_slope_deg(plateau_plane, frame)
    caveat = ""
    if plane_rms > 2.5:
        caveat = (
            f"the plateau plane fit has an RMS residual of {plane_rms:.2f} mm, so "
            "the plateau is markedly non-planar (osteophytes or a depressed "
            "compartment); treat the slope as approximate"
        )
    out.append(
        make_measurement(
            MeasurementName.POSTERIOR_TIBIAL_SLOPE,
            slope,
            Unit.DEG,
            method=(
                "angle in the sagittal plane between the fitted proximal tibial "
                "articular plane and the plane perpendicular to the tibial "
                f"longitudinal axis (plane fit RMS {plane_rms:.2f} mm)"
            ),
            structures=(Structure.TIBIA,),
            uncertainty=round(plane_rms, 3),
            sign_convention=(
                "positive means posterior slope, i.e. the plateau falls away "
                "posteriorly, which is the normal direction"
            ),
            caveat=caveat,
        )
    )

    # --- plateau concavity depths ---
    for name, landmark, compartment in (
        (
            MeasurementName.MEDIAL_PLATEAU_DEPTH,
            LandmarkName.MEDIAL_PLATEAU_LOW,
            "medial",
        ),
        (
            MeasurementName.LATERAL_PLATEAU_DEPTH,
            LandmarkName.LATERAL_PLATEAU_LOW,
            "lateral",
        ),
    ):
        if not landmarks.has(landmark):
            continue
        low_point = landmarks.position(landmark)
        # Depth below the fitted articular plane. Signed distance is negative on
        # the far side of the normal, so the depth is its magnitude when the point
        # lies below the plane.
        distance = float(plateau_plane.signed_distance(low_point))  # type: ignore[arg-type]
        depth = abs(distance)
        out.append(
            make_measurement(
                name,
                depth,
                Unit.MM,
                method=(
                    f"perpendicular distance from the fitted plateau plane to the "
                    f"deepest point of the {compartment} compartment"
                ),
                landmarks=(landmark,),
                structures=(Structure.TIBIA,),
            )
        )

    # --- plateau height difference (compartment collapse) ---
    if landmarks.has(
        LandmarkName.MEDIAL_PLATEAU_LOW, LandmarkName.LATERAL_PLATEAU_LOW
    ):
        medial_low = landmarks.position(LandmarkName.MEDIAL_PLATEAU_LOW)
        lateral_low = landmarks.position(LandmarkName.LATERAL_PLATEAU_LOW)
        difference = float(np.dot(lateral_low - medial_low, frame.superior))
        out.append(
            make_measurement(
                MeasurementName.PLATEAU_HEIGHT_DIFFERENCE,
                difference,
                Unit.MM,
                method=(
                    "superoinferior difference between the deepest points of the "
                    "lateral and medial compartments"
                ),
                landmarks=(
                    LandmarkName.MEDIAL_PLATEAU_LOW,
                    LandmarkName.LATERAL_PLATEAU_LOW,
                ),
                structures=(Structure.TIBIA,),
                sign_convention=(
                    "positive means the lateral compartment sits proximal to the "
                    "medial, consistent with medial compartment bone loss (varus)"
                ),
            )
        )

    return out


# ---------------------------------------------------------------------------
# Shared plateau geometry
# ---------------------------------------------------------------------------


def fit_articular_plane(
    tibia: AnatomicalMesh,
    frame: PatientFrame,
    *,
    fraction: float = 0.16,
    normal_threshold: float = 0.30,
) -> tuple[Plane, float]:
    """Fit a plane to the proximal tibial articular surface.

    Restricted to upward-facing vertices. Fitting to the whole proximal region
    would include the medial and lateral cortical walls, which are roughly
    vertical and would tilt the fitted plane substantially.

    Returns ``(plane, rms_residual_mm)``. The plane normal is oriented proximally.
    """
    surface = plateau_articular_surface(
        tibia, frame, fraction=fraction, normal_threshold=normal_threshold
    )
    plane, rms = fit_plane(surface, frame=tibia.frame, label="tibial_articular_plane")
    if float(np.dot(plane.normal, frame.superior)) < 0:
        plane = plane.flipped()
    return plane, rms


def posterior_slope_deg(plateau_plane: Plane, frame: PatientFrame) -> float:
    """Posterior slope of the plateau, in degrees, positive posteriorly.

    Measured as the angle between the plateau normal and the tibial longitudinal
    axis, projected into the sagittal plane. A plateau perpendicular to the axis
    reads 0°.
    """
    normal = plateau_plane.normal
    posterior_component = float(np.dot(normal, -frame.anterior))
    superior_component = float(np.dot(normal, frame.superior))
    if abs(superior_component) < 1e-9:
        raise MeasurementError(
            "the plateau plane is parallel to the tibial axis; slope is undefined"
        )
    # The upward normal of a surface tilts towards the downhill direction. For a
    # plateau descending posteriorly, the normal therefore tilts *posteriorly*.
    #
    # Check with the 45 degree case: a surface descending posteriorly has in-plane
    # direction (posterior, superior) = (1, -1); the perpendicular pointing upward
    # is (1, 1) — posterior and superior. So the posterior component of the normal
    # carries the sign of the posterior slope.
    return float(np.degrees(np.arctan2(posterior_component, superior_component)))


def joint_line_axis(
    landmarks: LandmarkSet,
    frame: PatientFrame,
    *,
    femoral: bool,
) -> NDArray[np.float64]:
    """Coronal joint-line direction, from the lateral to the medial reference point.

    Femoral: distal condylar points. Tibial: plateau low points. Both are
    projected into the coronal plane, because mLDFA, MPTA and JLCA are
    coronal-plane angles.
    """
    if femoral:
        names = (
            LandmarkName.DISTAL_LATERAL_CONDYLE,
            LandmarkName.DISTAL_MEDIAL_CONDYLE,
        )
    else:
        names = (LandmarkName.LATERAL_PLATEAU_LOW, LandmarkName.MEDIAL_PLATEAU_LOW)
    if not landmarks.has(*names):
        raise MeasurementError(
            f"cannot derive the {'femoral' if femoral else 'tibial'} joint line: "
            f"missing {[n.value for n in names if n not in landmarks]}"
        )
    lateral, medial = (landmarks.position(n) for n in names)
    vector = medial - lateral
    # Project into the coronal plane (drop the anteroposterior component).
    vector = vector - float(np.dot(vector, frame.anterior)) * frame.anterior
    if float(np.linalg.norm(vector)) < 5.0:
        raise MeasurementError(
            "the joint-line reference points are less than 5 mm apart in the "
            "coronal plane; the joint line is undefined"
        )
    return unit(vector)
