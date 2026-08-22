"""Robotic-assistance interface layer — simulation only (PRD Part X).

Three modules, deliberately separated by what they are allowed to do:

:mod:`kneeai.robotics.safety`
    The lifecycle gate. A simulation cannot start without a *named* surgeon
    approval and passing safety checks — enforced by raising, not by convention.

:mod:`kneeai.robotics.commands`
    Translates an approved plan into abstract tool poses and a keep-out boundary,
    for visualisation and review. Not joint angles, not servo setpoints.

:mod:`kneeai.robotics.plan`
    The inert, machine-readable plan document a future controller would consume.

What this package does not contain, and must not: a robot driver, a kinematic
solver, a network client, or any path from a model output to an actuator. PRD §80
and §117 require that path to be broken by a human decision. The break is
:meth:`~kneeai.robotics.safety.SimulationController.record_surgeon_approval`, and
everything downstream of it is still only data.

    >>> from kneeai.robotics import SimulationController, build_command_sequence
    >>> controller = SimulationController(case_id="c1", plan_id="p1")
    >>> controller.submit_for_approval()
    >>> controller.record_surgeon_approval(approver="dr-smith")
    >>> controller.start()          # raises until safety checks have passed
    Traceback (most recent call last):
    GuardrailViolation: simulation cannot start before safety checks pass ...
"""

from __future__ import annotations

from kneeai.robotics.commands import (
    DEFAULT_BOUNDARY_MARGIN_MM,
    DEFAULT_STANDOFF_MM,
    CommandSequence,
    RobotAction,
    RobotCommand,
    SafetyBoundary,
    ToolPose,
    build_command_sequence,
)
from kneeai.robotics.plan import RobotPlan, build_robot_plan, write_robot_plan
from kneeai.robotics.safety import (
    SIMULATION_ONLY_BANNER,
    SafetyCheck,
    SafetyReport,
    SimulationController,
    SimulationState,
    StateTransition,
    build_safety_report,
)

__all__ = [
    # safety
    "SimulationState",
    "SimulationController",
    "SafetyCheck",
    "SafetyReport",
    "StateTransition",
    "build_safety_report",
    "SIMULATION_ONLY_BANNER",
    # commands
    "RobotAction",
    "RobotCommand",
    "CommandSequence",
    "ToolPose",
    "SafetyBoundary",
    "build_command_sequence",
    "DEFAULT_STANDOFF_MM",
    "DEFAULT_BOUNDARY_MARGIN_MM",
    # plan
    "RobotPlan",
    "build_robot_plan",
    "write_robot_plan",
]
