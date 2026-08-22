"""Segmentation inference (PRD §12, §31).

The predictor does four things beyond running the network:

1. **Verifies compatibility** against the model card before loading weights
   (label map, modality, clinical approval).
2. **Applies the model's own preprocessing spec**, taken from the card rather
   than from a global default. A preprocessing mismatch between training and
   inference degrades accuracy silently.
3. **Reports per-structure confidence**, which the AI Findings panel displays
   (PRD §31). Confidence is the mean softmax probability over the voxels assigned
   to a structure — a genuinely computed quantity, not a fixed number.
4. **Returns the mask on the original acquisition grid**, so measurements are made
   in the geometry the image was acquired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import Structure, invert_label_map
from kneeai.core.errors import SegmentationError, ValidationError
from kneeai.core.provenance import Origin, Provenance
from kneeai.imaging.preprocess import PreprocessingSpec, preprocess, resample_like
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger
from kneeai.ml.models import ModelConfig, build_model, require_torch
from kneeai.ml.postprocess import PostProcessConfig, PostProcessReport, postprocess_mask
from kneeai.ml.registry import Architecture, ModelCard, ModelRegistry

__all__ = ["InferenceConfig", "SegmentationResult", "SegmentationPredictor"]

log = get_logger("ml.infer")


@dataclass
class InferenceConfig:
    sw_batch_size: int = 4
    overlap: float = 0.5

    #: Gaussian window weighting reduces the seam artefacts that a constant
    #: window leaves at patch borders — those seams land as small step
    #: discontinuities on the reconstructed bone surface.
    blend_mode: str = "gaussian"

    amp: bool = True

    #: Test-time augmentation by axis flipping. Off by default: the models are
    #: side-specific and flip-invariance is not a property we want to encourage
    #: (a mirrored right knee is a left knee).
    tta_flips: bool = False

    #: Voxels whose top-class probability is below this are left as background.
    #: Prevents a low-confidence halo from inflating structure volumes.
    min_confidence: float = 0.0

    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)

    #: Return the mask on the original acquisition grid rather than the
    #: preprocessed grid.
    return_on_source_grid: bool = True

    device: str | None = None


@dataclass
class SegmentationResult:
    """A predicted mask plus everything needed to judge and audit it."""

    mask: Volume
    confidence: dict[Structure, float]
    volumes_mm3: dict[Structure, float]
    model_card: ModelCard
    postprocess_report: PostProcessReport
    preprocessed_shape: tuple[int, int, int]
    provenance: Provenance
    probability_map: NDArray[np.float32] | None = None

    @property
    def structures(self) -> list[Structure]:
        return sorted(self.volumes_mm3, key=lambda s: s.value)

    def findings(self) -> list[dict[str, Any]]:
        """Structured AI findings for the surgeon dashboard (PRD §31)."""
        return [
            {
                "structure": s.value,
                "display_name": s.display_name,
                "segmentation": "complete" if self.volumes_mm3[s] > 0 else "not detected",
                "confidence": round(self.confidence.get(s, float("nan")), 4),
                "volume_mm3": round(self.volumes_mm3[s], 1),
                "model_id": self.model_card.model_id,
                "model_version": self.model_card.version,
            }
            for s in self.structures
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_card.key,
            "mask": self.mask.summary(),
            "preprocessed_shape": list(self.preprocessed_shape),
            "findings": self.findings(),
            "postprocessing": self.postprocess_report.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


class SegmentationPredictor:
    """Loads a registered segmentation model and predicts masks."""

    def __init__(
        self,
        card: ModelCard,
        *,
        checkpoint_path: str | Path | None = None,
        registry: ModelRegistry | None = None,
        config: InferenceConfig | None = None,
        require_clinical_approval: bool = False,
    ) -> None:
        torch, _monai = require_torch()
        self.torch = torch
        self.card = card
        self.config = config or InferenceConfig()

        if card.label_map is None:
            raise ValidationError(f"model {card.key} has no label map")
        card.check_compatibility(require_clinical_approval=require_clinical_approval)

        if card.architecture is Architecture.NNUNET:
            raise ValidationError(
                "nnU-Net checkpoints are executed through the nnU-Net predictor, "
                "not this one. See kneeai.ml.nnunet.predict_with_nnunet."
            )

        self.label_map: dict[Structure, int] = dict(card.label_map)
        self.inverse_label_map = invert_label_map(self.label_map)
        self.spec = (
            PreprocessingSpec.from_dict(dict(card.preprocessing))
            if card.preprocessing
            else PreprocessingSpec()
        )

        self.device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.amp = bool(self.config.amp and self.device.type == "cuda")

        path = Path(checkpoint_path) if checkpoint_path else None
        if path is None:
            reg = registry or ModelRegistry()
            path = reg.checkpoint_for(card.model_id, card.version)

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # The checkpoint's own label map is authoritative about what the output
        # channels mean. If it disagrees with the card, one of them was edited.
        ckpt_label_map = checkpoint.get("label_map")
        if ckpt_label_map:
            as_structures = {Structure(k): int(v) for k, v in ckpt_label_map.items()}
            if as_structures != self.label_map:
                raise ValidationError(
                    f"checkpoint label map disagrees with the model card for {card.key}.\n"
                    f"  checkpoint: {ckpt_label_map}\n"
                    f"  card      : {{{', '.join(f'{s.value}: {v}' for s, v in self.label_map.items())}}}"
                )

        model_config = ModelConfig(
            architecture=card.architecture,
            in_channels=card.input_channels,
            out_channels=len(self.label_map),
            patch_size=tuple(card.patch_size or (96, 96, 96)),  # type: ignore[arg-type]
        )
        if checkpoint.get("model_config"):
            saved = dict(checkpoint["model_config"])
            saved.pop("architecture", None)
            for key, value in saved.items():
                if hasattr(model_config, key) and value is not None:
                    setattr(
                        model_config,
                        key,
                        tuple(value) if isinstance(value, list) else value,
                    )
        self.model_config = model_config

        self.model = build_model(model_config).to(self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SegmentationError(
                f"checkpoint does not match the architecture recorded on card "
                f"{card.key}: {len(missing)} missing and {len(unexpected)} "
                f"unexpected parameter tensors. First missing: {list(missing)[:3]}"
            )
        self.model.eval()

        log.info(
            "predictor ready",
            fields={
                "model": card.key,
                "architecture": card.architecture.value,
                "device": str(self.device),
                "patch_size": list(model_config.patch_size),
                "structures": len(self.label_map) - 1,
                "clinical_approved": card.approved_for_clinical_use,
            },
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_registry(
        cls,
        reference: str,
        *,
        registry: ModelRegistry | None = None,
        **kwargs: Any,
    ) -> SegmentationPredictor:
        reg = registry or ModelRegistry()
        return cls(reg.resolve(reference), registry=reg, **kwargs)

    # ------------------------------------------------------------------
    def predict(
        self,
        volume: Volume,
        *,
        return_probabilities: bool = False,
    ) -> SegmentationResult:
        """Segment a volume. Returns the mask on the source grid by default."""
        torch = self.torch

        if volume.is_label:
            raise ValidationError("predict() expects an image volume, not a mask")
        self.card.check_compatibility(expected_modality=volume.modality)

        pre = preprocess(volume, self.spec)
        working = pre.volume

        tensor = (
            torch.from_numpy(np.ascontiguousarray(working.array, dtype=np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self._sliding_window(tensor)
            probabilities = torch.softmax(logits.float(), dim=1)[0]  # (C, i, j, k)

        prob_np = probabilities.cpu().numpy().astype(np.float32)
        del logits, probabilities, tensor
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        labels = prob_np.argmax(axis=0).astype(np.uint8)
        top_prob = prob_np.max(axis=0)

        if self.config.min_confidence > 0.0:
            suppressed = int(np.count_nonzero((labels != 0) & (top_prob < self.config.min_confidence)))
            labels[top_prob < self.config.min_confidence] = 0
            if suppressed:
                log.info(
                    "low-confidence voxels suppressed",
                    fields={
                        "voxels": suppressed,
                        "threshold": self.config.min_confidence,
                    },
                )

        raw_mask = Volume.from_labels(labels, self.label_map, reference=working)

        # Confidence is computed on the pre-post-processing mask: it should
        # describe what the network believed, not what cleanup left behind.
        confidence = self._structure_confidence(prob_np, labels)

        cleaned, report = postprocess_mask(raw_mask, self.config.postprocess)

        if self.config.return_on_source_grid:
            final = resample_like(cleaned, volume)
        else:
            final = cleaned

        volumes = {
            s: final.structure_volume_mm3(s)
            for s in self.label_map
            if s is not Structure.BACKGROUND
        }

        provenance = Provenance.for_component(
            "recon.surface",
            origin=Origin.AI,
            model_id=self.card.model_id,
            model_version=self.card.version,
            parameters={
                "operation": "segmentation_inference",
                "preprocessing_spec": self.spec.name,
                "sw_overlap": self.config.overlap,
                "blend_mode": self.config.blend_mode,
                "tta_flips": self.config.tta_flips,
                "min_confidence": self.config.min_confidence,
                "returned_grid": "source" if self.config.return_on_source_grid else "preprocessed",
                "source_volume_hash": volume.content_hash(),
            },
            notes=f"segmented by {self.card.key}",
        )
        final = final.with_metadata(model=self.card.key)

        result = SegmentationResult(
            mask=final,
            confidence=confidence,
            volumes_mm3=volumes,
            model_card=self.card,
            postprocess_report=report,
            preprocessed_shape=working.shape,
            provenance=provenance,
            probability_map=prob_np if return_probabilities else None,
        )
        log.info(
            "segmentation complete",
            fields={
                "model": self.card.key,
                "structures_found": sum(1 for v in volumes.values() if v > 0),
                "structures_expected": len(volumes),
                "mean_confidence": round(
                    float(np.nanmean(list(confidence.values()))) if confidence else float("nan"),
                    4,
                ),
            },
        )
        return result

    # ------------------------------------------------------------------
    def _sliding_window(self, tensor: Any) -> Any:
        torch = self.torch
        from monai.inferers import sliding_window_inference

        def run(x: Any) -> Any:
            with torch.amp.autocast("cuda", enabled=self.amp):
                return sliding_window_inference(
                    inputs=x,
                    roi_size=self.model_config.patch_size,
                    sw_batch_size=self.config.sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.overlap,
                    mode=self.config.blend_mode,
                    padding_mode="constant",
                )

        logits = run(tensor)
        if not self.config.tta_flips:
            return logits

        # Average logits over axis flips, flipping each prediction back first.
        count = 1
        for axes in ((2,), (3,), (4,)):
            flipped = run(torch.flip(tensor, dims=axes))
            logits = logits + torch.flip(flipped, dims=axes)
            count += 1
        return logits / count

    def _structure_confidence(
        self, probabilities: NDArray[np.float32], labels: NDArray[np.uint8]
    ) -> dict[Structure, float]:
        """Mean predicted probability over each structure's assigned voxels.

        Reported as NaN when a structure has no voxels: there is no meaningful
        confidence for something that was not found, and reporting 0.0 would be
        read as "found, with low confidence".
        """
        out: dict[Structure, float] = {}
        for structure, label in self.label_map.items():
            if structure is Structure.BACKGROUND:
                continue
            selected = labels == label
            if not selected.any():
                out[structure] = float("nan")
                continue
            out[structure] = float(probabilities[label][selected].mean())
        return out

    # ------------------------------------------------------------------
    def predict_batch(
        self, volumes: Sequence[Volume]
    ) -> list[SegmentationResult]:
        return [self.predict(v) for v in volumes]
