"""One segmentation interface, four interchangeable backends.

Everything downstream of this module — reconstruction, landmarking, measurement,
implant matching, planning, simulation, the API, the viewer — consumes a
:class:`SegmentationOutcome` and never learns which backend produced it. That is
what makes the application shippable before a model exists.

The four backends:

===============  =============================================================
``phantom``      :mod:`kneeai.phantom`. Works today; the default.
``mock``         Deterministic fixture for UI development, coarse on purpose.
``ground_truth`` Expert reference masks loaded from disk.
``ai_prediction``:mod:`kneeai.ml.registry` + :mod:`kneeai.ml.infer`.
===============  =============================================================

Two invariants:

**The AI backend never blocks on the model existing.** No model card, no
checkpoint, no torch, an out-of-memory failure — every one of them falls back to
the phantom backend and returns ``source="phantom"``.

**A fallback never keeps the requested source.** ``source`` describes what the
mask *is*, not what was asked for. A degraded run that still claimed
``ai_prediction`` would present synthetic geometry as a model result, which is
exactly what contract rule 2 exists to prevent. :attr:`SegmentationOutcome.degraded`
and :attr:`SegmentationOutcome.degradation_reason` record what happened, and the
reason is added to ``warnings`` so it reaches ``result.json`` and the UI.

The requested backend is expressed with :class:`~kneeai.ml.contract.ArtifactSource`
rather than a second enum: the four backends and the four source values are the
same four things, and a parallel enum would be one more pair of names to drift.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from kneeai.api.contract import MODEL_REGISTRY_RELATIVE_PATH, build_result_payload
from kneeai.config import Settings, get_settings
from kneeai.core.anatomy import DEFAULT_LABEL_MAP, LandmarkName, Side, Structure
from kneeai.core.errors import ValidationError
from kneeai.core.provenance import Origin, Provenance
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger
from kneeai.ml.contract import ArtifactSource
from kneeai.version import PLATFORM_VERSION

__all__ = [
    "AdapterConfig",
    "InferenceBackend",
    "MLInferenceAdapter",
    "SegmentationOutcome",
]

log = get_logger("api.adapter")

#: Alias kept because "backend" is the clearer word at call sites that are
#: choosing one. Same four values as the contract's source; deliberately not a
#: separate enum.
InferenceBackend = ArtifactSource

#: Environment variable selecting the backend, so a deployment can switch to
#: mock data for UI work without a code change.
BACKEND_ENV_VAR = "KNEEAI_INFERENCE_BACKEND"

#: Mock data is generated coarse on purpose: a blocky surface is visually
#: unmistakable, which contract rule 2 asks of the mock source, and it keeps UI
#: iteration fast.
MOCK_SPACING_MM = 2.0


@dataclass
class AdapterConfig:
    """Which backend to run and with what inputs."""

    backend: ArtifactSource = ArtifactSource.PHANTOM

    #: Operative side. Used by the synthetic backends and as an override for
    #: ground truth; the AI backend takes it from the image.
    side: Side = Side.RIGHT

    spacing_mm: float = 0.8

    #: Medial cartilage loss for the phantom (0 healthy .. 1 full thickness).
    cartilage_loss: float = 0.0

    image_path: Path | None = None
    mask_path: Path | None = None

    #: Registry reference for the AI backend, resolved under the artifact root.
    model_reference: str = "seg_oaizib"

    #: When true, a model not approved for clinical use is refused rather than
    #: run in research mode. Off by default: research use is the current state.
    require_clinical_approval: bool = False

    #: When false, an AI backend failure raises instead of degrading. Tests use
    #: this to prove the failure is real; production leaves it on.
    allow_degradation: bool = True

    #: Sliding-window overlap for inference. ``None`` uses
    #: :class:`~kneeai.ml.infer.InferenceConfig`'s default of 0.5.
    #:
    #: Exposed because the cost is steep and paid interactively. On CPU at 0.5 mm,
    #: 0.5 overlap is roughly 2-3x the wall clock of 0.25 for a marginal boundary
    #: improvement, and an upload that takes ten minutes instead of four is a
    #: materially worse product. Batch evaluation should leave it at the default.
    inference_overlap: float | None = None

    @classmethod
    def from_env(cls, **overrides: Any) -> AdapterConfig:
        """Build from ``KNEEAI_INFERENCE_BACKEND``, defaulting to phantom."""
        raw = os.environ.get(BACKEND_ENV_VAR, ArtifactSource.PHANTOM.value).strip()
        try:
            backend = ArtifactSource(raw)
        except ValueError as exc:
            valid = [s.value for s in ArtifactSource]
            raise ValidationError(
                f"{BACKEND_ENV_VAR}={raw!r} is not a known backend; choose from {valid}"
            ) from exc
        config = cls(backend=backend)
        return replace(config, **overrides) if overrides else config

    def effective_spacing_mm(self) -> float:
        if self.backend is ArtifactSource.MOCK and self.spacing_mm == 0.8:
            return MOCK_SPACING_MM
        return self.spacing_mm


@dataclass
class SegmentationOutcome:
    """A label volume plus everything the contract needs to describe it.

    Identical in shape whichever backend produced it.
    """

    mask: Volume
    source: ArtifactSource
    confidence: Mapping[Structure, float | None]
    confidence_method: str
    provenance: Provenance

    #: The intensity image, when the backend has one. ``None`` for a bare mask.
    image: Volume | None = None

    model_id: str | None = None
    model_version: str | None = None
    architecture: str | None = None
    model_card: str | None = None

    warnings: list[str] = field(default_factory=list)

    #: Analytic ground truth, phantom backends only. Used by tests and by the
    #: demo's truth table; never presented as a clinical measurement.
    truth: dict[str, float] = field(default_factory=dict)

    #: Landmarks that lie outside a knee field of view (hip, ankle) and so cannot
    #: be detected from the volume. Supplied to the landmark detector.
    external_landmarks: dict[LandmarkName, Any] = field(default_factory=dict)

    requested_backend: ArtifactSource | None = None
    degraded: bool = False
    degradation_reason: str | None = None

    @property
    def side(self) -> Side:
        return self.mask.side

    @property
    def is_model_output(self) -> bool:
        return self.source.is_model_output

    def result_payload(
        self,
        *,
        case_id: str,
        mask: Volume | None = None,
        extra_warnings: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Assemble the ``result.json`` payload through the canonical builder.

        Args:
            mask: the mask actually written to disk, when it differs from
                :attr:`mask` (canonicalised, post-processed). ``result.json``
                must describe the published volume, not an intermediate one.
        """
        published = mask if mask is not None else self.mask
        expected_but_missing = [
            s
            for s in DEFAULT_LABEL_MAP
            if s is not Structure.BACKGROUND and s not in (published.label_map or {})
        ]
        return build_result_payload(
            case_id=case_id,
            source=self.source,
            mask=published,
            confidence=self.confidence,
            confidence_method=self.confidence_method,
            provenance=self.provenance,
            model_id=self.model_id,
            model_version=self.model_version,
            architecture=self.architecture,
            model_card=self.model_card,
            structures_expected_but_missing=expected_but_missing,
            warnings=[*self.warnings, *extra_warnings],
        )


class MLInferenceAdapter:
    """Produces a :class:`SegmentationOutcome` from the configured backend."""

    def __init__(
        self, config: AdapterConfig | None = None, *, settings: Settings | None = None
    ) -> None:
        self.config = config or AdapterConfig()
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    @property
    def model_card_path(self) -> Path:
        """Where the AI backend looks for its model card.

        ``artifacts/models/seg_oaizib/model_card.json`` — the path fixed by
        ``docs/ML_CONTRACT.md`` §6.
        """
        reference = self.config.model_reference
        if reference == "seg_oaizib":
            relative = MODEL_REGISTRY_RELATIVE_PATH
        else:
            relative = f"models/{reference}"
        return self.settings.artifact_root / relative / "model_card.json"

    def model_available(self) -> bool:
        """Whether a usable model card is on disk. Never raises."""
        try:
            return self.load_model_card() is not None
        except Exception:  # noqa: BLE001 - availability probe must not throw
            return False

    def load_model_card(self) -> Any | None:
        """Load and verify the model card, or return ``None``.

        Verification is contract rule 4: the card's label ordering is checked
        against :data:`~kneeai.core.anatomy.DEFAULT_LABEL_MAP` before any weights
        are loaded. A checkpoint that calls tibia 2 produces confident,
        anatomically scrambled output and no error.
        """
        path = self.model_card_path
        if not path.exists():
            return None
        from kneeai.ml.registry import ModelCard

        card = ModelCard.from_dict(json.loads(path.read_text()))
        self._verify_label_map(card)
        return card

    @staticmethod
    def _verify_label_map(card: Any) -> None:
        """Check the card's label map against the canonical one.

        A model that segments a *subset* of the canonical structures is
        legitimate — Model 1 produces five of ten — so the subset is compared,
        not the whole map. The integers must match exactly.
        """
        declared = dict(card.label_map or {})
        if not declared:
            raise ValidationError(f"model {card.key} declares no label map")
        subset = {s: v for s, v in DEFAULT_LABEL_MAP.items() if s in declared}
        card.check_compatibility(expected_label_map=subset)
        extra = sorted(s.value for s in declared if s not in DEFAULT_LABEL_MAP)
        if extra:
            raise ValidationError(
                f"model {card.key} declares structures absent from "
                f"core.anatomy.DEFAULT_LABEL_MAP: {extra}"
            )

    # ------------------------------------------------------------------
    def segment(self) -> SegmentationOutcome:
        """Run the configured backend."""
        backend = self.config.backend
        log.info(
            "segmentation requested",
            fields={"backend": backend.value, "side": self.config.side.value},
        )
        if backend is ArtifactSource.PHANTOM:
            return self._phantom(source=ArtifactSource.PHANTOM)
        if backend is ArtifactSource.MOCK:
            return self._phantom(source=ArtifactSource.MOCK)
        if backend is ArtifactSource.GROUND_TRUTH:
            return self._ground_truth()
        if backend is ArtifactSource.AI_PREDICTION:
            return self._ai_prediction()
        raise ValidationError(f"unhandled backend: {backend!r}")

    # --- A. ground truth ----------------------------------------------
    def _ground_truth(self) -> SegmentationOutcome:
        from kneeai.imaging.io import read_volume

        path = self.config.mask_path
        if path is None:
            raise ValidationError(
                "the ground_truth backend requires mask_path (the reference "
                "annotation to load)"
            )
        path = Path(path)
        if not path.exists():
            raise ValidationError(f"ground-truth mask not found: {path}")

        mask = read_volume(path, label_map=dict(DEFAULT_LABEL_MAP), is_label=True)
        if self.config.side is not Side.UNKNOWN and mask.side is Side.UNKNOWN:
            mask = replace(mask, side=self.config.side)

        provenance = Provenance(
            component="imaging.io",
            component_version=PLATFORM_VERSION,
            origin=Origin.IMPORTED,
            parameters={"mask_path": str(path), "loader": "imaging.io.read_volume"},
            notes="Expert reference annotation loaded from disk. Not a model output.",
        )
        return SegmentationOutcome(
            mask=mask,
            source=ArtifactSource.GROUND_TRUTH,
            confidence={},
            confidence_method=(
                "not applicable: this mask is an expert reference annotation, not a "
                "model prediction, so it has no predicted probability"
            ),
            provenance=provenance,
            warnings=[
                "GROUND TRUTH: expert reference annotation. This is the performance "
                "ceiling for a model, not model output, and must never be presented "
                "as an AI result."
            ],
            requested_backend=ArtifactSource.GROUND_TRUTH,
        )

    # --- B. phantom / C. mock -----------------------------------------
    def _phantom(
        self,
        *,
        source: ArtifactSource,
        degraded_from: ArtifactSource | None = None,
        reason: str | None = None,
    ) -> SegmentationOutcome:
        """Build a phantom-backed outcome.

        Confidence is reported as unavailable rather than 1.0. The phantom mask
        is exact by construction, but "exact" is not a model probability, and a
        column of 1.00 confidences in the UI reads as an extremely confident
        model. Unavailable is the honest rendering.
        """
        from kneeai.phantom import build_phantom

        spacing = self.config.effective_spacing_mm()
        knee = build_phantom(
            side=self.config.side,
            spacing_mm=spacing,
            medial_cartilage_loss=self.config.cartilage_loss,
        )

        warnings: list[str] = []
        if source is ArtifactSource.MOCK:
            warnings.append(
                "MOCK DATA: deterministic fixture for user-interface development. "
                "Not a patient, not a model output, and carries no anatomical "
                f"meaning. Generated coarse ({spacing:.1f} mm voxels) so it is "
                "visually unmistakable."
            )
        else:
            warnings.append(
                "PHANTOM DATA: synthetic knee generated by kneeai.phantom with "
                "analytic ground truth. Not a patient and not a model output."
            )
        if degraded_from is not None and reason:
            warnings.append(
                f"A {degraded_from.value} result was requested but could not be "
                f"produced ({reason}). This output is {source.badge} — synthetic "
                "geometry, not a model prediction — and source has been set "
                "accordingly."
            )
        if self.config.cartilage_loss > 0:
            warnings.append(
                f"Phantom generated with {self.config.cartilage_loss:.0%} medial "
                "cartilage loss to represent a varus osteoarthritic knee."
            )

        provenance = Provenance(
            component="phantom.generator",
            component_version=PLATFORM_VERSION,
            origin=Origin.DETERMINISTIC,
            parameters={
                "generator": "kneeai.phantom.build_phantom",
                "side": self.config.side.value,
                "spacing_mm": spacing,
                "medial_cartilage_loss": self.config.cartilage_loss,
                "requested_backend": (degraded_from or source).value,
            },
            notes="Synthetic phantom with analytic ground truth. No patient data.",
        )
        return SegmentationOutcome(
            mask=knee.mask,
            image=knee.image,
            source=source,
            confidence={},
            confidence_method=(
                "not applicable: the phantom mask is analytic ground truth "
                "generated from closed-form geometry, not a model prediction, so no "
                "probabilistic confidence exists. Reported as unavailable rather "
                "than 1.0."
            ),
            provenance=provenance,
            warnings=warnings,
            truth=dict(knee.truth),
            external_landmarks={
                LandmarkName.FEMORAL_HEAD_CENTER: knee.landmark(
                    LandmarkName.FEMORAL_HEAD_CENTER
                ),
                LandmarkName.ANKLE_CENTER: knee.landmark(LandmarkName.ANKLE_CENTER),
            },
            requested_backend=degraded_from or source,
            degraded=degraded_from is not None,
            degradation_reason=reason,
        )

    # --- D. ai prediction ---------------------------------------------
    def _ai_prediction(self) -> SegmentationOutcome:
        def degrade(reason: str) -> SegmentationOutcome:
            if not self.config.allow_degradation:
                raise ValidationError(
                    f"AI inference is unavailable and degradation is disabled: {reason}"
                )
            log.warning(
                "AI backend unavailable; falling back to phantom",
                fields={"reason": reason},
            )
            return self._phantom(
                source=ArtifactSource.PHANTOM,
                degraded_from=ArtifactSource.AI_PREDICTION,
                reason=reason,
            )

        try:
            card = self.load_model_card()
        except Exception as exc:  # noqa: BLE001 - a bad card must degrade, not crash
            return degrade(
                f"the model card at {self.model_card_path} was rejected: {exc}"
            )
        if card is None:
            return degrade(f"no model card at {self.model_card_path}")

        if self.config.require_clinical_approval and not card.approved_for_clinical_use:
            return degrade(
                f"model {card.key} is not approved for clinical use and clinical "
                "approval was required"
            )

        image_path = self.config.image_path
        if image_path is None:
            return degrade("no input image was given (image_path is required)")
        image_path = Path(image_path)
        if not image_path.exists():
            return degrade(f"input image not found: {image_path}")

        try:
            from kneeai.imaging.io import read_volume
            from kneeai.ml.infer import SegmentationPredictor

            from kneeai.ml.infer import InferenceConfig

            image = read_volume(image_path)
            inference_config = (
                InferenceConfig(overlap=self.config.inference_overlap)
                if self.config.inference_overlap is not None
                else None
            )
            predictor = SegmentationPredictor(
                card,
                config=inference_config,
                require_clinical_approval=self.config.require_clinical_approval,
            )
            result = predictor.predict(image)
        except Exception as exc:  # noqa: BLE001 - torch absent, no checkpoint, OOM
            return degrade(f"{type(exc).__name__}: {exc}")

        warnings = list(result.postprocess_report.warnings)
        if not card.approved_for_clinical_use:
            warnings.append(
                f"Model {card.key} is NOT approved for clinical use "
                "(model_card.approved_for_clinical_use is false). Research use only."
            )
        missing = [s for s in DEFAULT_LABEL_MAP if s not in (card.label_map or {})]
        if missing:
            log.info(
                "model segments a subset of the canonical structures",
                fields={"not_segmented": [s.value for s in missing]},
            )

        try:
            relative_card = str(self.model_card_path.relative_to(self.settings.artifact_root))
        except ValueError:
            relative_card = str(self.model_card_path)

        return SegmentationOutcome(
            mask=result.mask,
            image=image,
            source=ArtifactSource.AI_PREDICTION,
            confidence=dict(result.confidence),
            confidence_method=(
                "mean predicted softmax probability over the voxels assigned to each "
                "structure, computed before post-processing"
            ),
            provenance=result.provenance,
            model_id=card.model_id,
            model_version=card.version,
            architecture=card.architecture.value,
            model_card=relative_card,
            warnings=warnings,
            requested_backend=ArtifactSource.AI_PREDICTION,
        )
