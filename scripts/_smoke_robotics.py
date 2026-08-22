"""Smoke check for the robotics layer, run against a real case directory.

Proves three things:
1. The approval gate actually blocks — ``start()`` raises without it.
2. Command sequences derive from the real cut planes in ``plan.json``.
3. The robot plan document assembles and serialises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kneeai.core.errors import GuardrailViolation
from kneeai.robotics import (
    SIMULATION_ONLY_BANNER,
    SimulationController,
    SimulationState,
    build_command_sequence,
    build_robot_plan,
    build_safety_report,
    write_robot_plan,
)

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / (sys.argv[1] if len(sys.argv) > 1 else "artifacts/demo/cases/demo-phantom-right")


def load(name: str):
    p = CASE / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    print("=" * 76)
    print(f"ROBOTICS SMOKE — {CASE.relative_to(REPO)}")
    print(SIMULATION_ONLY_BANNER)
    print("=" * 76)

    result = load("result.json")
    plan = load("plan.json")
    if result is None or plan is None:
        raise SystemExit(f"case is missing result.json or plan.json: {CASE}")
    measurements = load("measurements.json")
    implants = load("implants.json")
    simulation = load("simulation.json")
    manifest = load("meshes/manifest.json")

    print(f"source   : {result.get('source')}")
    print(f"plan     : {plan.get('plan_id')}  v{plan.get('version')}")
    print(f"cuts     : {len(plan.get('cuts', []))}")

    # --- 1. the gate blocks --------------------------------------------------
    print("\n" + "-" * 76)
    print("1. APPROVAL GATE")
    print("-" * 76)
    controller = SimulationController(case_id=result["case_id"], plan_id=plan["plan_id"])
    print(f"  state: {controller.state.value}")

    try:
        controller.start()
        print("  FAIL: start() succeeded with no approval")
        raise SystemExit(1)
    except GuardrailViolation as exc:
        print(f"  start() correctly refused: {exc}")

    try:
        controller.submit_for_approval()
        controller.record_surgeon_approval(approver="   ")
        print("  FAIL: anonymous approval accepted")
        raise SystemExit(1)
    except GuardrailViolation as exc:
        print(f"  anonymous approval refused: {str(exc)[:90]}...")

    controller.record_surgeon_approval(
        approver="dr-smith", justification="reviewed cuts and implant fit"
    )
    print(f"  approved by {controller.approver} -> {controller.state.value}")

    try:
        controller.start()
        print("  FAIL: start() succeeded before safety checks")
        raise SystemExit(1)
    except GuardrailViolation as exc:
        print(f"  start() before safety checks refused: {str(exc)[:80]}...")

    # --- 2. safety checks ----------------------------------------------------
    print("\n" + "-" * 76)
    print("2. SAFETY CHECKS")
    print("-" * 76)
    report = build_safety_report(
        plan=plan,
        simulation=simulation,
        measurements=measurements,
        source=result.get("source", "unknown"),
    )
    controller.run_safety_checks(report)
    for c in report.checks:
        flag = "PASS" if c.passed else ("BLOCK" if c.blocking else "warn")
        print(f"  [{flag:5}] {c.name:36} {c.detail[:70]}")
    print(f"  overall: {'PASS' if report.passed else 'BLOCKED'}  state={controller.state.value}")

    # --- 3. commands ---------------------------------------------------------
    print("\n" + "-" * 76)
    print("3. COMMAND SEQUENCE")
    print("-" * 76)
    commands = build_command_sequence(plan, mesh_manifest=manifest, simulation=simulation)
    for c in commands.commands:
        pose = ""
        if c.pose:
            pose = "  @ [" + ", ".join(f"{v:7.2f}" for v in c.pose.position) + "]"
        print(f"  {c.sequence:2}. {c.action.value:28} {c.target:24}{pose}")
    if commands.boundary:
        b = commands.boundary
        print(
            f"\n  keep-out box: {[round(v,1) for v in b.min_mm]} .. "
            f"{[round(v,1) for v in b.max_mm]}  margin {b.margin_mm} mm"
        )
        print(f"  derived from: {', '.join(b.derived_from)}")
    outside = commands.poses_outside_boundary()
    print(f"  cutting poses outside boundary: {len(outside)}")
    for w in commands.warnings:
        print(f"  WARNING: {w}")

    # --- 4. run it -----------------------------------------------------------
    print("\n" + "-" * 76)
    print("4. SIMULATION RUN")
    print("-" * 76)
    if report.passed:
        controller.start()
        print(f"  started -> {controller.state.value} (allows_motion={controller.state.allows_motion})")
        controller.complete(summary=f"{commands.n_commands} commands simulated")
        print(f"  completed -> {controller.state.value}")
    else:
        print(f"  not started; state is {controller.state.value}")

    stopped = SimulationController(case_id="c", plan_id="p")
    stopped.submit_for_approval()
    stopped.record_surgeon_approval(approver="dr-jones")
    stopped.emergency_stop(actor="operator", reason="demo of E-stop")
    print(f"  e-stop test: {stopped.state.value}  terminal={stopped.state.is_terminal}")
    assert stopped.state is SimulationState.EMERGENCY_STOPPED

    # --- 5. document ---------------------------------------------------------
    print("\n" + "-" * 76)
    print("5. ROBOT PLAN DOCUMENT")
    print("-" * 76)
    robot = build_robot_plan(
        result=result,
        plan=plan,
        measurements=measurements,
        implants=implants,
        simulation=simulation,
        mesh_manifest=manifest,
        commands=commands,
        controller=controller,
    )
    written = write_robot_plan(robot, CASE, commands=commands)
    for k, p in written.items():
        print(f"  {k:18} {p.relative_to(REPO)}  ({p.stat().st_size} bytes)")

    d = robot.to_dict()
    print(f"\n  top-level keys: {list(d)}")
    print(f"  executable    : {d['executable']}")
    print(f"  implant       : {d['implant']['configuration_label']} "
          f"fem={d['implant']['femoral_size']} tib={d['implant']['tibial_size']}")
    print(f"  alignment     : strategy={d['alignment']['strategy']}")
    if d["alignment"]["unavailable"]:
        print(f"  unavailable   : {d['alignment']['unavailable']}")
        print(f"  reason        : {(d['alignment']['unavailable_reason'] or '')[:100]}...")
    print(f"  key dims mm   : {d['anatomy']['key_dimensions_mm']}")

    print("\n" + "=" * 76)
    print("ROBOTICS SMOKE: PASS")
    print("=" * 76)


if __name__ == "__main__":
    main()
