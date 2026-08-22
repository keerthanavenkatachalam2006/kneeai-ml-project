"""FastAPI surface for the surgeon dashboard.

Serves the case artefacts the CLI pipeline already writes, plus the surgeon
approval gate and the robot simulation endpoints.

Design notes worth knowing before extending this:

**It reads artefacts, it does not compute them.** ``kneeai demo run`` and
``kneeai infer`` produce a complete case directory — result, meshes, measurements,
implants, plan, simulation. This app resolves and serves those. Inference is not
triggered per request: on the dev machine a single case takes 30-40 s on CPU, and
an HTTP handler is the wrong place for that. A production deployment puts it on
the queue in ``worker/``.

**No authentication.** This binds to 127.0.0.1 and is a local review tool. It
must not be exposed beyond loopback without adding auth first — every endpoint
here would otherwise serve patient-derived geometry to anyone who can reach the
port. ``pyjwt`` and ``passlib`` are already declared in ``pyproject`` for when
that happens.

**Case ids are validated, not interpolated.** Every path segment goes through
:func:`_safe_case_id` before touching the filesystem, because ``case_id`` reaches
us from the URL and ``artifacts/cases/../../etc/passwd`` is otherwise a valid
path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from kneeai.config import get_settings
from kneeai.core.errors import GuardrailViolation, KneeAIError
from kneeai.logging import configure_logging, get_logger
from kneeai.ml.contract import ArtifactSource, validate_result_json
from kneeai.robotics import (
    SIMULATION_ONLY_BANNER,
    SimulationController,
    build_command_sequence,
    build_robot_plan,
    build_safety_report,
    write_robot_plan,
)
from kneeai.version import PLATFORM_VERSION

log = get_logger(__name__)

# Case ids appear in filesystem paths. Anything outside this alphabet is rejected.
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

#: Artefact name -> filename, so the URL vocabulary is closed rather than
#: forwarding whatever the client asks for.
ARTIFACTS: dict[str, str] = {
    "result": "result.json",
    "segmentation": "segmentation.json",
    "landmarks": "landmarks.json",
    "frame": "frame.json",
    "measurements": "measurements.json",
    "implants": "implants.json",
    "plan": "plan.json",
    "simulation": "simulation.json",
    "kinematics": "kinematics.json",
    "optimization": "optimization.json",
    "mesh_quality": "mesh_quality.json",
    "robot_plan": "robot_plan.json",
    "robot_commands": "robot_commands.json",
}

DISCLAIMER = (
    "Research and simulation prototype. Not a medical device, not clinically "
    "validated, and not for use in the care of any patient."
)

app = FastAPI(
    title="Knee Intelligence Platform",
    version=PLATFORM_VERSION,
    description=(
        "Local review API for AI knee segmentation, 3D reconstruction, "
        "measurement, implant matching, surgical planning and robot simulation. "
        + DISCLAIMER
    ),
)

# The Next.js dev server runs on 3000. Loopback only, matching the bind address.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Live simulation controllers, keyed by case id. In memory on purpose: a
# simulation run is session state, and persisting approvals belongs in db/.
_CONTROLLERS: dict[str, SimulationController] = {}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _safe_case_id(case_id: str) -> str:
    if not _CASE_ID.match(case_id):
        raise HTTPException(400, f"invalid case id: {case_id!r}")
    return case_id


def _case_roots() -> list[Path]:
    """Directories that may contain cases, in search order."""
    root = Path(get_settings().artifact_root)
    return [root / "cases", root / "demo" / "cases", root / "live" / "cases"]


def _case_dir(case_id: str) -> Path:
    cid = _safe_case_id(case_id)
    for base in _case_roots():
        candidate = base / cid
        if (candidate / "result.json").exists():
            return candidate
    raise HTTPException(404, f"case not found: {cid}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise HTTPException(404, f"artifact not found: {path.name}") from None
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"{path.name} is not valid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel):
    approver: str = Field(..., min_length=1, description="Named surgeon. Required.")
    justification: str = Field("", description="Free text recorded in the audit trail.")


class StopRequest(BaseModel):
    actor: str = Field("operator", min_length=1)
    reason: str = Field("operator pressed stop")


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


@app.on_event("startup")
def _startup() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    log.info(
        "api started",
        fields={"artifact_root": str(settings.artifact_root), "version": PLATFORM_VERSION},
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "platform_version": PLATFORM_VERSION,
        "banner": SIMULATION_ONLY_BANNER,
        "disclaimer": DISCLAIMER,
        "hardware_control": False,
        "authentication": False,
        "bind_note": "loopback only; do not expose without adding auth",
    }


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    """The registered segmentation model card, or an explicit absence.

    Reports the card's *reference* metrics. These describe the model's performance
    on its validation set and say nothing about the case currently on screen.
    """
    card_path = Path(get_settings().model_registry) / "seg_oaizib" / "model_card.json"
    if not card_path.exists():
        return {
            "available": False,
            "reason": (
                "No model card at artifacts/models/seg_oaizib/model_card.json. "
                "Cases will be served from the phantom backend."
            ),
        }
    card = _read_json(card_path)
    metrics = (card.get("metrics") or {}).get("internal_validation") or {}
    return {
        "available": True,
        "model_id": card.get("model_id"),
        "version": card.get("version"),
        "architecture": card.get("architecture"),
        "task": card.get("task"),
        "label_map": card.get("label_map"),
        "approved_for_clinical_use": card.get("approved_for_clinical_use", False),
        "reference_metrics": {
            "scope": "validation set of the training run, NOT this patient",
            "n_patients": metrics.get("n_patients"),
            "mean_dice_excl_background": metrics.get("dice_mean_excl_background"),
            "dice_per_structure": metrics.get("dice_per_structure"),
            "metric_definition": metrics.get("metric_definition"),
        },
        "dataset": {
            k: (card.get("dataset_contract") or {}).get(k)
            for k in (
                "dataset_ids",
                "n_patients",
                "split_patients",
                "split_strategy",
                "license",
                "commercial_use_permitted",
                "population_notes",
            )
        },
        "limitations": card.get("limitations"),
        "contraindications": card.get("contraindications", []),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@app.get("/api/cases")
def list_cases() -> dict[str, Any]:
    """Every case with a readable ``result.json``, newest first."""
    seen: dict[str, dict[str, Any]] = {}
    for base in _case_roots():
        if not base.is_dir():
            continue
        for directory in sorted(base.iterdir()):
            result_file = directory / "result.json"
            if not directory.is_dir() or not result_file.exists():
                continue
            if directory.name in seen:
                continue
            try:
                result = json.loads(result_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            geometry = result.get("geometry", {}) or {}
            segmentation = result.get("segmentation", {}) or {}
            source = str(result.get("source", "unknown"))
            try:
                badge = ArtifactSource(source).badge
            except ValueError:
                badge = source.upper()
            seen[directory.name] = {
                "case_id": result.get("case_id", directory.name),
                "source": source,
                "badge": badge,
                "is_ai": source == ArtifactSource.AI_PREDICTION.value,
                "side": geometry.get("side"),
                "spacing_mm": geometry.get("spacing_mm"),
                "image_shape": geometry.get("image_shape"),
                "n_structures": len(segmentation.get("structures_present", []) or []),
                "structures_missing": segmentation.get(
                    "structures_expected_but_missing", []
                ),
                "model": (result.get("model") or {}).get("model_id"),
                "has_plan": (directory / "plan.json").exists(),
                "has_meshes": (directory / "meshes" / "manifest.json").exists(),
                "has_simulation": (directory / "simulation.json").exists(),
                "n_warnings": len(result.get("warnings", []) or []),
                "modified": result_file.stat().st_mtime,
            }
    cases = sorted(seen.values(), key=lambda c: c["modified"], reverse=True)
    return {"n_cases": len(cases), "cases": cases}


@app.get("/api/cases/{case_id}")
def case_summary(case_id: str) -> dict[str, Any]:
    """Everything the dashboard needs for one case, in a single round trip."""
    directory = _case_dir(case_id)
    result = _read_json(directory / "result.json")

    def optional(name: str) -> Any:
        path = directory / ARTIFACTS[name]
        return _read_json(path) if path.exists() else None

    report = validate_result_json(directory / "result.json")
    source = str(result.get("source", "unknown"))
    try:
        badge = ArtifactSource(source).badge
    except ValueError:
        badge = source.upper()

    controller = _CONTROLLERS.get(case_id)
    return {
        "case_id": result.get("case_id", case_id),
        "source": source,
        "badge": badge,
        "is_ai": source == ArtifactSource.AI_PREDICTION.value,
        "result": result,
        "landmarks": optional("landmarks"),
        "frame": optional("frame"),
        "measurements": optional("measurements"),
        "implants": optional("implants"),
        "plan": optional("plan"),
        "simulation": optional("simulation"),
        "mesh_quality": optional("mesh_quality"),
        "meshes": (
            _read_json(directory / "meshes" / "manifest.json")
            if (directory / "meshes" / "manifest.json").exists()
            else None
        ),
        "contract_valid": report.valid,
        "contract_errors": list(report.errors),
        "simulation_state": controller.state.value if controller else "planning",
        "banner": SIMULATION_ONLY_BANNER,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/cases/{case_id}/artifacts/{name}")
def case_artifact(case_id: str, name: str) -> Any:
    if name not in ARTIFACTS:
        raise HTTPException(
            404, f"unknown artifact {name!r}; known: {sorted(ARTIFACTS)}"
        )
    return _read_json(_case_dir(case_id) / ARTIFACTS[name])


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------


@app.get("/api/cases/{case_id}/meshes")
def mesh_manifest(case_id: str) -> Any:
    return _read_json(_case_dir(case_id) / "meshes" / "manifest.json")


@app.get("/api/cases/{case_id}/meshes/{filename}")
def mesh_file(case_id: str, filename: str) -> FileResponse:
    """Serve one GLB.

    ``filename`` is checked against the manifest rather than sanitised, so only
    meshes this case actually declared can be fetched.
    """
    directory = _case_dir(case_id)
    manifest = _read_json(directory / "meshes" / "manifest.json")
    allowed = {entry["path"] for entry in manifest.get("meshes", []) or []}
    if filename not in allowed:
        raise HTTPException(404, f"{filename} is not a mesh of this case")
    path = directory / "meshes" / filename
    if not path.exists():
        raise HTTPException(404, f"mesh file missing on disk: {filename}")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/cases/{case_id}/segmentation.nii.gz")
def segmentation_volume(case_id: str) -> FileResponse:
    path = _case_dir(case_id) / "segmentation.nii.gz"
    if not path.exists():
        raise HTTPException(404, "segmentation volume not found")
    return FileResponse(path, media_type="application/gzip", filename=path.name)


# ---------------------------------------------------------------------------
# 2D slices
# ---------------------------------------------------------------------------


def _source_image(case_id: str) -> Path | None:
    """The intensity volume a case was segmented from, if it is still on disk.

    ``result.json`` carries the input path in its provenance. The phantom has no
    intensity volume at all, which is why this is optional rather than required.
    """
    directory = _case_dir(case_id)
    local = directory / "image.nii.gz"
    if local.exists():
        return local
    try:
        result = _read_json(directory / "result.json")
        provenance = result.get("provenance") or {}
        for ref in provenance.get("inputs", []) or []:
            path = ref.get("path") if isinstance(ref, dict) else None
            if path and str(path).endswith((".nii", ".nii.gz")):
                candidate = Path(path)
                if not candidate.is_absolute():
                    candidate = Path(get_settings().artifact_root).parent / candidate
                if candidate.exists():
                    return candidate
    except HTTPException:
        return None
    return None


@app.get("/api/cases/{case_id}/slices")
def slice_info(case_id: str) -> dict[str, Any]:
    """Slice counts and a sensible starting index per plane."""
    from kneeai.api.slices import slice_geometry

    directory = _case_dir(case_id)
    label = directory / "segmentation.nii.gz"
    if not label.exists():
        raise HTTPException(404, "no segmentation volume for this case")
    geometry = slice_geometry(label)
    payload = geometry.to_dict()
    source = _source_image(case_id)
    payload["has_source_image"] = source is not None
    payload["note"] = (
        "Greyscale is the source MR with segmentation contours overlaid."
        if source is not None
        else "No intensity volume for this case, so labels are drawn filled on black. "
        "The phantom is generated geometry and has no scan behind it."
    )
    return payload


@app.get("/api/cases/{case_id}/slice.png")
def slice_png(
    case_id: str,
    plane: str = "axial",
    index: int = 0,
    mode: str = "contour",
    zoom: float = 2.0,
) -> Response:
    """One rendered slice as PNG.

    Rendered server-side: slice geometry depends on the affine, and native spacing
    here is anisotropic (0.7 x 0.365 x 0.365 mm), so a slice drawn on a square pixel
    grid is visibly distorted. That correction belongs next to the code that already
    understands the volume, not reimplemented in the browser.
    """
    from kneeai.api.slices import render_slice

    directory = _case_dir(case_id)
    label = directory / "segmentation.nii.gz"
    if not label.exists():
        raise HTTPException(404, "no segmentation volume for this case")
    if plane not in ("axial", "coronal", "sagittal"):
        raise HTTPException(400, f"unknown plane {plane!r}")
    if mode not in ("contour", "fill", "image"):
        raise HTTPException(400, f"unknown mode {mode!r}")

    try:
        png = render_slice(
            label_path=label,
            plane=plane,  # type: ignore[arg-type]
            index=index,
            image_path=_source_image(case_id),
            mode=mode,
            zoom=max(0.5, min(zoom, 6.0)),
        )
    except KneeAIError as exc:
        raise HTTPException(422, str(exc)) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=600"},
    )


# ---------------------------------------------------------------------------
# Surgeon approval and robot simulation
# ---------------------------------------------------------------------------


def _controller_for(case_id: str, plan: dict[str, Any]) -> SimulationController:
    controller = _CONTROLLERS.get(case_id)
    if controller is None:
        controller = SimulationController(
            case_id=case_id, plan_id=str(plan.get("plan_id", "unknown"))
        )
        _CONTROLLERS[case_id] = controller
    return controller


@app.get("/api/cases/{case_id}/simulation/state")
def simulation_state(case_id: str) -> dict[str, Any]:
    directory = _case_dir(case_id)
    controller = _CONTROLLERS.get(case_id)
    if controller is None:
        return {
            "case_id": case_id,
            "state": "planning",
            "approver": None,
            "allows_motion": False,
            "banner": SIMULATION_ONLY_BANNER,
            "hardware_control": False,
            "has_plan": (directory / "plan.json").exists(),
        }
    return controller.to_dict()


@app.post("/api/cases/{case_id}/simulation/approve")
def approve_plan(case_id: str, body: ApprovalRequest) -> dict[str, Any]:
    """The mandatory human gate. Runs safety checks, then permits simulation.

    Approval and safety checks happen in one call because a plan approved but
    never checked is a state with no meaning — the surgeon's decision is about the
    plan, and the checks are about whether the artefacts are coherent enough to
    animate it.
    """
    directory = _case_dir(case_id)
    plan_path = directory / "plan.json"
    if not plan_path.exists():
        raise HTTPException(409, "case has no surgical plan to approve")

    plan = _read_json(plan_path)
    result = _read_json(directory / "result.json")
    simulation = (
        _read_json(directory / "simulation.json")
        if (directory / "simulation.json").exists()
        else None
    )
    measurements = (
        _read_json(directory / "measurements.json")
        if (directory / "measurements.json").exists()
        else None
    )

    controller = _controller_for(case_id, plan)
    try:
        from kneeai.robotics.safety import SimulationState

        if controller.state is SimulationState.PLANNING:
            controller.submit_for_approval(actor=body.approver)
        controller.record_surgeon_approval(
            approver=body.approver, justification=body.justification
        )
        report = build_safety_report(
            plan=plan,
            simulation=simulation,
            measurements=measurements,
            source=str(result.get("source", "unknown")),
        )
        controller.run_safety_checks(report)
    except GuardrailViolation as exc:
        raise HTTPException(409, str(exc)) from exc

    return controller.to_dict()


@app.post("/api/cases/{case_id}/simulation/run")
def run_simulation(case_id: str) -> dict[str, Any]:
    """Build the robot command sequence and mark the run complete.

    Refuses without approval. The refusal comes from
    :class:`~kneeai.robotics.safety.SimulationController`, not from a check here,
    so there is exactly one place the gate is enforced.
    """
    directory = _case_dir(case_id)
    controller = _CONTROLLERS.get(case_id)
    if controller is None:
        raise HTTPException(409, "plan has not been submitted for approval")

    plan = _read_json(directory / "plan.json")
    result = _read_json(directory / "result.json")
    simulation = (
        _read_json(directory / "simulation.json")
        if (directory / "simulation.json").exists()
        else None
    )
    measurements = (
        _read_json(directory / "measurements.json")
        if (directory / "measurements.json").exists()
        else None
    )
    implants = (
        _read_json(directory / "implants.json")
        if (directory / "implants.json").exists()
        else None
    )
    manifest = (
        _read_json(directory / "meshes" / "manifest.json")
        if (directory / "meshes" / "manifest.json").exists()
        else None
    )

    try:
        commands = build_command_sequence(
            plan, mesh_manifest=manifest, simulation=simulation
        )
        controller.start()
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
        write_robot_plan(robot, directory, commands=commands)
        controller.complete(summary=f"{commands.n_commands} commands simulated")
    except GuardrailViolation as exc:
        raise HTTPException(409, str(exc)) from exc
    except KneeAIError as exc:
        raise HTTPException(422, str(exc)) from exc

    return {
        "state": controller.to_dict(),
        "commands": commands.to_dict(),
        "robot_plan": robot.to_dict(),
    }


@app.post("/api/cases/{case_id}/simulation/stop")
def emergency_stop(case_id: str, body: StopRequest) -> dict[str, Any]:
    """Halt the simulation. Terminal for this run.

    Stops a simulation. There is no hardware attached to this system.
    """
    controller = _CONTROLLERS.get(case_id)
    if controller is None:
        raise HTTPException(404, "no simulation in progress for this case")
    controller.emergency_stop(actor=body.actor, reason=body.reason)
    return controller.to_dict()


@app.post("/api/cases/{case_id}/simulation/reset")
def reset_simulation(case_id: str) -> dict[str, Any]:
    """Discard the run so the approval flow can be demonstrated again."""
    _CONTROLLERS.pop(case_id, None)
    return {"case_id": case_id, "state": "planning", "reset": True}


@app.get("/api/cases/{case_id}/robot/commands")
def robot_commands(case_id: str) -> Any:
    """The command sequence, derived on demand without requiring approval.

    Reading a proposed sequence is not running it. Approval gates *execution*,
    and refusing to let a surgeon read what would be executed before approving it
    would invert the point of the gate.
    """
    directory = _case_dir(case_id)
    plan = _read_json(directory / "plan.json")
    simulation = (
        _read_json(directory / "simulation.json")
        if (directory / "simulation.json").exists()
        else None
    )
    manifest = (
        _read_json(directory / "meshes" / "manifest.json")
        if (directory / "meshes" / "manifest.json").exists()
        else None
    )
    try:
        return build_command_sequence(
            plan, mesh_manifest=manifest, simulation=simulation
        ).to_dict()
    except KneeAIError as exc:
        raise HTTPException(422, str(exc)) from exc


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@app.exception_handler(KneeAIError)
def _domain_error(_request: Any, exc: KneeAIError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc), "type": type(exc).__name__})


@app.get("/", include_in_schema=False)
def root() -> Response:
    return JSONResponse(
        {
            "name": "Knee Intelligence Platform API",
            "version": PLATFORM_VERSION,
            "docs": "/docs",
            "banner": SIMULATION_ONLY_BANNER,
            "disclaimer": DISCLAIMER,
        }
    )
