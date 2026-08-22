"""Surgical simulation (PRD §28, §73, §74).

Four levels are defined in PRD §73. Two are implemented:

- **Level 1 — geometric** (:mod:`kneeai.simulation.geometric`): virtual cuts,
  component placement, static interpenetration, resected bone volume.
- **Level 2 — kinematic** (:mod:`kneeai.simulation.kinematic`): rigid-body flexion
  sweep, interference through range, joint gap trend.
- **Level 3 — biomechanical**: finite element, soft tissue, contact pressure. Not
  implemented. PRD §29 classifies it as a research module requiring validation
  before it could inform a plan, and an unvalidated FE result presented next to
  validated geometry would be read as equally trustworthy.
- **Level 4 — robotic**: workspace, trajectories, safety limits. See
  :mod:`kneeai.robotics`.

    >>> from kneeai.simulation import simulate_plan, simulate_flexion
    >>> geometric = simulate_plan(plan, meshes, configuration, frame)
    >>> print(geometric.summary())
    >>> kinematic = simulate_flexion(geometric, landmarks, frame)
    >>> print(kinematic.summary())

Both simulations report findings rather than raising, because PRD §88 requires
failing safe on impossible geometry: a report listing what is wrong is more useful
to a surgeon than an exception.
"""

from __future__ import annotations

from kneeai.simulation.geometric import (
    COLLISION_TOLERANCE_MM,
    CutResult,
    Finding,
    Severity,
    SimulationReport,
    apply_cuts,
    component_solid,
    simulate_plan,
)
from kneeai.simulation.kinematic import (
    FlexionSample,
    KinematicConfig,
    KinematicReport,
    flexion_axis,
    simulate_flexion,
)

__all__ = [
    # geometric
    "Severity",
    "Finding",
    "CutResult",
    "SimulationReport",
    "simulate_plan",
    "apply_cuts",
    "component_solid",
    "COLLISION_TOLERANCE_MM",
    # kinematic
    "KinematicConfig",
    "FlexionSample",
    "KinematicReport",
    "flexion_axis",
    "simulate_flexion",
]
