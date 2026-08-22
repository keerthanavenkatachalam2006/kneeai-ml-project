"""Scan upload: take a file from a browser and turn it into a case.

The API's standing rule is that it reads artefacts and does not compute them,
because inference takes 30-60 s on this machine and an HTTP handler is the wrong
place for that. Upload cannot honour that rule — the artefacts do not exist yet —
so it does the next best thing: the request validates and returns, and the work
happens on a single background worker whose progress is pollable.

The worker is deliberately **one** thread. Two concurrent 3D segmentations on a
laptop do not run twice as fast, they run four times as slow and then exhaust
memory, so uploads queue instead of competing.

Accepted inputs and what happens to each:

=================  ==========================================================
NIfTI ``.nii.gz``  the internal interchange format; used directly
DICOM ``.zip``     extracted, one series selected, converted to NIfTI
single ``.dcm``    treated as a series in its own directory
2D radiograph      stored and validated, then **stopped** with an explicit
                   reason: every model here is 3D MR, and there is no
                   radiograph model to run
=================  ==========================================================

The radiograph path is the interesting one. It would be easy to accept an X-ray,
run nothing, and show a confident-looking empty result. Instead the case is
created, the image is kept and displayed, and the OA verdict says exactly why it
is unavailable. A product that cannot answer a question should say so.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from kneeai.core.anatomy import Modality, Side
from kneeai.core.errors import KneeAIError, ValidationError
from kneeai.logging import get_logger

__all__ = [
    "MAX_UPLOAD_BYTES",
    "IngestedScan",
    "UploadJob",
    "UploadRegistry",
    "get_registry",
    "ingest_upload",
    "new_case_id",
]

log = get_logger("api.upload")

#: Hard ceiling on an uploaded file. A knee MR series is a few hundred MB at
#: most; anything far beyond that is a mistake or an attack, and streaming it to
#: disk before checking would be the bug.
MAX_UPLOAD_BYTES = 1_500_000_000  # 1.5 GB

#: Sliding-window overlap used for uploads, against the 0.5 default.
#:
#: This is a real accuracy-for-latency trade and it is made deliberately. On CPU at
#: 0.5 mm the patch count scales steeply with overlap, and 0.5 puts an upload in the
#: region of ten minutes against roughly four at 0.25. Batch evaluation
#: (``scripts/evaluate_segresnet.py``) leaves the default alone, so the numbers on
#: the model card are not the ones this shortcut produces — a gap worth remembering
#: when comparing an uploaded case against the card.
UPLOAD_INFERENCE_OVERLAP = 0.25

_NIFTI_SUFFIXES = (".nii", ".nii.gz")
_ARCHIVE_SUFFIXES = (".zip",)
_DICOM_SUFFIXES = (".dcm", ".dicom", ".ima")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: Ordered pipeline stages, used to turn the pipeline's stage labels into a
#: progress fraction. Substring matched, so wording changes downstream degrade to
#: "no progress movement" rather than a crash.
_STAGE_ORDER: tuple[tuple[str, float], ...] = (
    ("canonicalising", 0.30),
    ("post-processing", 0.35),
    ("writing segmentation", 0.40),
    ("reconstructing", 0.50),
    ("detecting landmarks", 0.60),
    ("measuring", 0.66),
    ("assessing osteoarthritis", 0.72),
    ("matching implant", 0.78),
    ("planning resections", 0.84),
    ("simulating", 0.90),
    ("exporting", 0.94),
    ("scoring robotic", 0.96),
    ("writing derived", 0.98),
)


def new_case_id(prefix: str = "upload") -> str:
    """A filesystem-safe, collision-resistant case id."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@dataclass
class IngestedScan:
    """The outcome of normalising an uploaded file."""

    #: ``"volume"`` for anything the 3D pipeline can consume, ``"radiograph"``
    #: for 2D projections it cannot.
    kind: str

    #: The NIfTI the pipeline will read. ``None`` for a radiograph.
    image_path: Path | None

    #: The file exactly as uploaded, kept for audit.
    original_path: Path

    modality: Modality = Modality.UNKNOWN
    side: Side = Side.UNKNOWN

    validation: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    #: Populated for ``kind == "radiograph"``: why the 3D pipeline was not run.
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "image_path": str(self.image_path) if self.image_path else None,
            "original_filename": self.original_path.name,
            "modality": self.modality.value,
            "side": self.side.value,
            "validation": self.validation,
            "warnings": list(self.warnings),
            "blocked_reason": self.blocked_reason,
        }


def _classify(filename: str) -> str:
    name = filename.lower()
    if name.endswith(_NIFTI_SUFFIXES):
        return "nifti"
    if name.endswith(_ARCHIVE_SUFFIXES):
        return "archive"
    if name.endswith(_DICOM_SUFFIXES):
        return "dicom"
    if name.endswith(_IMAGE_SUFFIXES):
        return "image2d"
    return "unknown"


def _looks_like_dicom(path: Path) -> bool:
    """DICOM files carry the magic ``DICM`` at byte 128."""
    try:
        with path.open("rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def _safe_extract(archive: Path, target: Path) -> None:
    """Extract a zip, refusing entries that escape the target directory."""
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        total = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            total += info.file_size
            if total > MAX_UPLOAD_BYTES:
                raise ValidationError(
                    "the archive expands to more than "
                    f"{MAX_UPLOAD_BYTES // 1_000_000} MB and was rejected"
                )
            destination = (target / info.filename).resolve()
            if not str(destination).startswith(str(target.resolve())):
                raise ValidationError(
                    f"archive entry {info.filename!r} escapes the extraction "
                    "directory and was rejected"
                )
        zf.extractall(target)


def ingest_upload(
    original: Path,
    *,
    work_dir: Path,
    side: Side = Side.UNKNOWN,
    modality_hint: Modality | None = None,
) -> IngestedScan:
    """Normalise an uploaded file into a NIfTI volume, or explain why not.

    Raises:
        ValidationError: the file is not something this platform can read, or it
            fails a blocking validation check.
    """
    from kneeai.imaging.io import dicom_to_nifti, read_volume, write_volume
    from kneeai.imaging.validation import validate_volume

    work_dir.mkdir(parents=True, exist_ok=True)
    kind = _classify(original.name)
    if kind == "unknown" and _looks_like_dicom(original):
        kind = "dicom"

    warnings: list[str] = []
    nifti_path = work_dir / "image.nii.gz"

    if kind == "image2d":
        return IngestedScan(
            kind="radiograph",
            image_path=None,
            original_path=original,
            modality=modality_hint or Modality.XRAY,
            side=side,
            blocked_reason=(
                "This is a 2D image. Every model in this platform operates on a 3D "
                "volume: the segmentation network is a 3D UNet-family model trained "
                "on 3D DESS knee MR, and the reconstruction, landmark, measurement "
                "and planning stages all need voxel geometry a projection does not "
                "have. There is no radiograph model to fall back on — no "
                "KL-graded radiograph dataset is present in data/raw. Upload an MR "
                "or CT volume (NIfTI or a DICOM zip) for the full assessment."
            ),
            warnings=[
                "A radiograph is genuinely useful to this platform for one thing the "
                "MR cannot provide: the femoral head and ankle centres needed for "
                "mechanical-axis alignment (HKA, MLDFA, MPTA). That import path is "
                "not wired to upload yet."
            ],
        )

    if kind == "nifti":
        # Copy under the platform's expected name so the slice renderer and the
        # case directory agree on where the intensity volume lives.
        suffix = ".nii.gz" if original.name.lower().endswith(".nii.gz") else ".nii"
        staged = work_dir / f"source{suffix}"
        shutil.copy2(original, staged)
        volume = read_volume(staged, modality=modality_hint, side=side or None)

    elif kind in ("archive", "dicom"):
        dicom_dir = work_dir / "dicom"
        if kind == "archive":
            _safe_extract(original, dicom_dir)
        else:
            dicom_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, dicom_dir / original.name)
        volume, _ref = dicom_to_nifti(dicom_dir, nifti_path)
        warnings.append(
            "Converted from DICOM. Verify the selected series is the 3D sequence "
            "intended for segmentation; the choice is recorded in the sidecar."
        )
    else:
        raise ValidationError(
            f"unsupported file type: {original.name!r}. Accepted: NIfTI "
            "(.nii, .nii.gz), a DICOM series as .zip, a single DICOM file, or a "
            "2D radiograph (.png, .jpg)."
        )

    # Side and modality: an explicit choice from the operator overrides whatever
    # the file claims, because a mislabelled export is more common than a
    # mistaken operator and the operator is accountable.
    if side is not Side.UNKNOWN and volume.side is not side:
        if volume.side is not Side.UNKNOWN:
            warnings.append(
                f"The file declares side '{volume.side.value}' but '{side.value}' was "
                "selected on upload. The selection was applied. Resolve this before "
                "relying on the plan — it is a wrong-site risk."
            )
        volume = replace(volume, side=side)
    if modality_hint is not None and volume.modality is not modality_hint:
        volume = replace(volume, modality=modality_hint)

    # A bare NIfTI carries no modality tag, so an untagged upload arrives as
    # UNKNOWN. That is not a soft detail: the model card's compatibility gate
    # refuses to run an MR model on an UNKNOWN volume, and rightly so. Since this
    # platform's only segmentation model is MR and the operator chose to upload
    # here, the assumption is applied explicitly and recorded loudly rather than
    # left to fail deep inside the predictor.
    assumed_modality = False
    if volume.modality is Modality.UNKNOWN:
        volume = replace(volume, modality=Modality.MR)
        assumed_modality = True

    report = validate_volume(volume)
    validation = report.to_dict()
    warnings.extend(i.message for i in report.warnings)

    if not report.passed:
        blocking = "; ".join(i.message for i in report.errors)
        raise ValidationError(
            f"the uploaded volume failed validation and was not processed: {blocking}"
        )

    if assumed_modality:
        warnings.append(
            "The file carries no modality tag, so it was ASSUMED to be MR — the only "
            "modality this platform's segmentation model is trained for. If this is "
            "actually a CT, the segmentation is unvalidated cross-modality use and "
            "its output should not be trusted. Re-upload with the modality set "
            "explicitly to remove the assumption."
        )

    # Written (or rewritten) through write_volume so the sidecar carries the
    # resolved modality, side and provenance rather than whatever the upload had.
    write_volume(volume, nifti_path)

    if volume.modality is Modality.CT:
        warnings.append(
            "This is a CT. The segmentation model was trained on 3D DESS MR only; "
            "applying it across modalities is unvalidated and its output should be "
            "treated as unreliable until measured on CT with ground truth."
        )

    return IngestedScan(
        kind="volume",
        image_path=nifti_path,
        original_path=original,
        modality=volume.modality,
        side=volume.side,
        validation=validation,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

#: Terminal states. A job in one of these will never change again.
_TERMINAL = frozenset({"complete", "failed", "rejected", "radiograph_only"})


@dataclass
class UploadJob:
    """One upload's lifecycle, pollable over HTTP."""

    case_id: str
    filename: str
    bytes_size: int
    side: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    modality: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    #: Headline outcome, filled in on completion: the OA verdict and the robotic
    #: readiness percentage, so the client gets the answer without a second call.
    outcome: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "filename": self.filename,
            "bytes_size": self.bytes_size,
            "side": self.side,
            "status": self.status,
            "stage": self.stage,
            "progress": round(float(self.progress), 3),
            "terminal": self.terminal,
            "modality": self.modality,
            "validation": self.validation,
            "warnings": list(self.warnings),
            "error": self.error,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # Measured against now while the job is live. Segmentation is a single
            # stage lasting most of the wall clock and it posts no intermediate
            # updates, so deriving elapsed time from updated_at would freeze the
            # clock during precisely the part the user is waiting through.
            "elapsed_s": round(
                (self.updated_at if self.terminal else time.time()) - self.created_at, 1
            ),
        }


class UploadRegistry:
    """In-memory job table plus the single-slot worker that drains it.

    In memory on purpose, matching the simulation controllers in ``api.main``:
    a job is session state. Persisting it belongs with the case record in
    ``db/``, not here.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._jobs: dict[str, UploadJob] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="kneeai-upload"
        )

    # --- accessors ---
    def get(self, case_id: str) -> UploadJob | None:
        with self._lock:
            return self._jobs.get(case_id)

    def all(self) -> list[UploadJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def queue_depth(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if not j.terminal)

    def create(self, job: UploadJob) -> UploadJob:
        with self._lock:
            self._jobs[job.case_id] = job
        return job

    def update(self, case_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(case_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def add_warnings(self, case_id: str, warnings: list[str]) -> None:
        with self._lock:
            job = self._jobs.get(case_id)
            if job is None:
                return
            existing = set(job.warnings)
            job.warnings.extend(w for w in warnings if w not in existing)
            job.updated_at = time.time()

    # --- submission ---
    def submit(self, job_id: str, fn: Any, /, *args: Any, **kwargs: Any) -> None:
        """Queue work for a job.

        ``job_id`` and ``fn`` are positional-only so that ``kwargs`` intended for
        ``fn`` — which will normally include its own ``case_id`` — cannot collide
        with this method's own parameters.
        """
        future = self._executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._futures[job_id] = future

        def _done(f: Future[None]) -> None:
            exc = f.exception()
            if exc is not None:
                log.error(
                    "upload job raised",
                    fields={"case_id": job_id, "error": f"{type(exc).__name__}: {exc}"},
                )
                self.update(
                    job_id,
                    status="failed",
                    stage="failed",
                    progress=1.0,
                    error=f"{type(exc).__name__}: {exc}",
                )

        future.add_done_callback(_done)

    def stage_progress(self, label: str) -> float:
        lowered = label.lower()
        for needle, fraction in _STAGE_ORDER:
            if needle in lowered:
                return fraction
        return 0.0


_REGISTRY: UploadRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> UploadRegistry:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = UploadRegistry()
        return _REGISTRY


# ---------------------------------------------------------------------------
# The worker body
# ---------------------------------------------------------------------------


def process_upload(
    *,
    case_id: str,
    original: Path,
    case_root: Path,
    side: Side,
    modality_hint: Modality | None = None,
    strict: bool = True,
) -> None:
    """Ingest, segment, plan and score one uploaded scan.

    Runs on the upload worker thread. Every failure is recorded on the job rather
    than raised at a client that has long since had its response.
    """
    from kneeai.api.adapter import AdapterConfig, MLInferenceAdapter
    from kneeai.api.pipeline import PipelineOptions, run_case_pipeline
    from kneeai.ml.contract import ArtifactSource

    registry = get_registry()
    case_dir = case_root / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. ingest ---
    registry.update(case_id, status="validating", stage="reading and validating the scan",
                    progress=0.05)
    try:
        scan = ingest_upload(
            original, work_dir=case_dir, side=side, modality_hint=modality_hint
        )
    except KneeAIError as exc:
        registry.update(
            case_id,
            status="rejected",
            stage="rejected at validation",
            progress=1.0,
            error=str(exc),
        )
        return

    registry.update(
        case_id,
        modality=scan.modality.value,
        validation=scan.validation,
        side=scan.side.value,
    )
    registry.add_warnings(case_id, scan.warnings)

    if scan.kind == "radiograph":
        registry.update(
            case_id,
            status="radiograph_only",
            stage="radiograph stored; no 3D model applies",
            progress=1.0,
            error=None,
            outcome={
                "oa_available": False,
                "oa_reason": scan.blocked_reason,
                "readiness_available": False,
                "readiness_reason": (
                    "Robotic readiness is scored from a 3D plan, which a radiograph "
                    "cannot produce."
                ),
            },
        )
        return

    # The model card declares MR and the compatibility gate enforces it. Checking
    # here turns a deep ValidationError from inside the predictor into a statement
    # about the input, which is what the operator can actually act on.
    if scan.modality is Modality.CT:
        registry.update(
            case_id,
            status="rejected",
            stage="CT stored; the segmentation model is MR-only",
            progress=1.0,
            error=(
                "This is a CT, and the only registered segmentation model was trained "
                "on 3D DESS knee MR. Running it on CT is cross-modality use with no "
                "validation behind it, so the model card's compatibility gate refuses "
                "it rather than producing a confident, unmeasured mask. The scan has "
                "been stored and the 2D views are available. To proceed you would need "
                "a model trained or validated on CT."
            ),
            outcome={
                "oa_available": False,
                "oa_reason": (
                    "The OA classifier consumes a segmentation, and no segmentation "
                    "could be produced for a CT."
                ),
                "readiness_available": False,
                "readiness_reason": "No plan exists, because no segmentation was produced.",
            },
        )
        return

    # --- 2. segment ---
    registry.update(
        case_id,
        status="segmenting",
        stage="running AI segmentation (this is the slow step)",
        progress=0.12,
    )
    config = AdapterConfig(
        backend=ArtifactSource.AI_PREDICTION,
        side=scan.side,
        image_path=scan.image_path,
        model_reference="seg_oaizib",
        allow_degradation=not strict,
        # Interactive path: the user is watching a progress bar. 0.25 overlap is
        # 2-3x faster on CPU than the 0.5 default for a marginal boundary gain.
        inference_overlap=UPLOAD_INFERENCE_OVERLAP,
    )
    try:
        outcome = MLInferenceAdapter(config).segment()
    except KneeAIError as exc:
        registry.update(
            case_id,
            status="failed",
            stage="segmentation failed",
            progress=1.0,
            error=str(exc),
        )
        return

    if outcome.degraded:
        registry.add_warnings(
            case_id,
            [
                "AI segmentation was unavailable and the result was produced by the "
                f"phantom backend instead: {outcome.degradation_reason}"
            ],
        )

    # --- 3. pipeline ---
    registry.update(case_id, status="processing", stage="building the case", progress=0.30)

    def on_stage(label: str) -> None:
        registry.update(
            case_id, stage=label, progress=max(0.30, registry.stage_progress(label))
        )

    try:
        artifacts = run_case_pipeline(
            outcome,
            case_id=case_id,
            out_root=case_root,
            options=PipelineOptions(on_stage=on_stage),
        )
    except KneeAIError as exc:
        registry.update(
            case_id, status="failed", stage="pipeline failed", progress=1.0, error=str(exc)
        )
        return

    registry.add_warnings(case_id, list(artifacts.warnings))

    # --- 4. headline outcome ---
    outcome_payload: dict[str, Any] = {}
    oa = artifacts.oa_assessment
    if oa is not None:
        outcome_payload.update(
            {
                "oa_available": bool(oa.get("available")),
                "oa_detected": oa.get("oa_detected"),
                "oa_probability": oa.get("probability"),
                "oa_severity": oa.get("severity"),
                "oa_reason": oa.get("unavailable_reason"),
            }
        )
    readiness = artifacts.readiness
    if readiness is not None:
        outcome_payload.update(
            {
                "readiness_available": True,
                "readiness_pct": readiness.get("percentage"),
                "readiness_band": readiness.get("band"),
                "n_blockers": readiness.get("n_blockers"),
            }
        )

    registry.update(
        case_id,
        status="complete",
        stage="complete",
        progress=1.0,
        outcome=outcome_payload,
        side=str(artifacts.side.value),
    )
    log.info(
        "upload processed",
        fields={
            "case_id": case_id,
            "seconds": artifacts.total_seconds,
            "planned": artifacts.planned,
        },
    )
