"""Implant intelligence: catalog, resection footprints, geometric matching.

PRD §20 calls this one of the most commercially important modules, and §22 is
explicit that it starts deterministic rather than learned:

    >>> from kneeai.implants import load_catalog, match_configurations
    >>> from kneeai.implants import provisional_tibial_plane, resection_footprint
    >>>
    >>> catalog = load_catalog("builtin:reference-v0")
    >>> tibial_plane = provisional_tibial_plane(meshes, landmarks, frame)
    >>> tibial_fp = resection_footprint(meshes.tibia, tibial_plane, frame,
    ...                                 provisional=True)
    >>> fits, provenance = match_configurations(femoral_fp, tibial_fp, catalog,
    ...                                         side=frame.side)
    >>> print("\\n".join(fits[0].rationale()))

The bundled catalog is synthetic, non-clinical reference geometry. See
:mod:`kneeai.implants.reference`.
"""

from __future__ import annotations

from kneeai.implants.catalog import (
    BearingType,
    ComponentType,
    FixationType,
    ImplantCatalog,
    ImplantComponent,
    ImplantConfiguration,
    ImplantFamily,
    load_catalog,
)
from kneeai.implants.footprint import (
    DEFAULT_DISTAL_FEMORAL_RESECTION_MM,
    DEFAULT_TIBIAL_RESECTION_MM,
    BoneFootprint,
    provisional_distal_femoral_plane,
    provisional_tibial_plane,
    resection_footprint,
)
from kneeai.implants.matching import (
    ConfigurationFit,
    FemoralDimensions,
    FitConfig,
    FitMethod,
    ImplantFit,
    ScoreTerm,
    fit_component,
    fit_femoral_component,
    match_configurations,
    rank_components,
    rank_femoral_components,
)
from kneeai.implants.reference import (
    REFERENCE_FEMORAL_SIZES,
    REFERENCE_INSERT_THICKNESSES,
    REFERENCE_TIBIAL_SIZES,
    build_reference_catalog,
    femoral_footprint,
    tibial_footprint,
)

__all__ = [
    # catalog
    "ComponentType",
    "FixationType",
    "BearingType",
    "ImplantComponent",
    "ImplantConfiguration",
    "ImplantFamily",
    "ImplantCatalog",
    "load_catalog",
    # reference geometry
    "build_reference_catalog",
    "REFERENCE_FEMORAL_SIZES",
    "REFERENCE_TIBIAL_SIZES",
    "REFERENCE_INSERT_THICKNESSES",
    "femoral_footprint",
    "tibial_footprint",
    # footprints
    "BoneFootprint",
    "resection_footprint",
    "provisional_tibial_plane",
    "provisional_distal_femoral_plane",
    "DEFAULT_TIBIAL_RESECTION_MM",
    "DEFAULT_DISTAL_FEMORAL_RESECTION_MM",
    # matching
    "FitConfig",
    "FitMethod",
    "FemoralDimensions",
    "ScoreTerm",
    "ImplantFit",
    "ConfigurationFit",
    "fit_component",
    "fit_femoral_component",
    "rank_components",
    "rank_femoral_components",
    "match_configurations",
]
