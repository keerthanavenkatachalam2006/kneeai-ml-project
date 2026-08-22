"""Segmentation training loop (PRD §35, §66).

Runs on Google Colab with a CUDA runtime. Memory-constrained-GPU techniques
called for in PRD §66 are all here and on by default where safe: mixed
precision, gradient accumulation, patch-based 3D training, and gradient
checkpointing for the transformer architectures.

Validation runs sliding-window inference over whole volumes rather than random
patches, so the reported Dice is measured the same way the model will be used.

On completion the trainer emits a :class:`~kneeai.ml.registry.ModelCard`. It
requires a :class:`~kneeai.ml.registry.DatasetContract` up front, so a run cannot
be started without one (PRD §113).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from kneeai.core.anatomy import Modality, Structure
from kneeai.core.errors import ConfigurationError
from kneeai.core.provenance import new_id, now_utc
from kneeai.logging import get_logger
from kneeai.ml.models import (
    ModelConfig,
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
    count_parameters,
    describe_device,
    require_torch,
)
from kneeai.ml.registry import (
    Architecture,
    DatasetContract,
    ModelCard,
    ModelMetrics,
    ModelRegistry,
    ModelTask,
)

__all__ = ["TrainConfig", "TrainingResult", "SegmentationTrainer"]

log = get_logger("ml.train")


@dataclass
class TrainConfig:
    """Run-level training configuration."""

    run_name: str = ""
    output_dir: Path = Path("./artifacts/runs")

    max_epochs: int = 300
    val_interval: int = 5

    #: Mixed precision. Roughly halves activation memory, which is what makes a
    #: 96³ patch fit alongside a SwinUNETR on a 16 GB Colab GPU.
    amp: bool = True

    #: Accumulate gradients to reach a larger effective batch without the memory.
    grad_accumulation_steps: int = 1

    #: Gradient clipping. 3D segmentation losses spike when a patch happens to
    #: contain almost no foreground.
    grad_clip_norm: float | None = 12.0

    #: Stop when validation Dice has not improved for this many validations.
    early_stopping_patience: int | None = 20

    #: Sliding-window inference settings used during validation.
    sw_batch_size: int = 2
    sw_overlap: float = 0.5

    seed: int = 42
    deterministic: bool = False
    save_every_epoch: bool = False
    log_every_n_steps: int = 20

    #: Optional MLflow tracking (PRD §68).
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "knee-segmentation"

    def __post_init__(self) -> None:
        if self.grad_accumulation_steps < 1:
            raise ConfigurationError("grad_accumulation_steps must be >= 1")
        if not 0.0 <= self.sw_overlap < 1.0:
            raise ConfigurationError("sw_overlap must be in [0, 1)")
        self.output_dir = Path(self.output_dir)
        if not self.run_name:
            self.run_name = f"run-{now_utc().strftime('%Y%m%d-%H%M%S')}-{new_id()[:6]}"


@dataclass
class TrainingResult:
    run_id: str
    run_dir: Path
    best_metric: float
    best_epoch: int
    epochs_trained: int
    history: list[dict[str, Any]] = field(default_factory=list)
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    per_structure_dice: dict[str, float] = field(default_factory=dict)
    model_card: ModelCard | None = None
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "epochs_trained": self.epochs_trained,
            "per_structure_dice": self.per_structure_dice,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "wall_time_s": round(self.wall_time_s, 1),
            "history": self.history,
        }


class SegmentationTrainer:
    """Trains a 3D segmentation model and registers it with a model card."""

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        *,
        label_map: dict[Structure, int],
        dataset_contract: DatasetContract,
        preprocessing: dict[str, Any] | None = None,
        modality: Modality = Modality.MR,
        device: str | None = None,
    ) -> None:
        torch, _monai = require_torch()
        self.torch = torch

        if model_config.out_channels != len(label_map):
            raise ConfigurationError(
                f"model has {model_config.out_channels} output channels but the "
                f"label map defines {len(label_map)} classes. These must match, "
                "or output channels will be attributed to the wrong anatomy."
            )

        self.model_config = model_config
        self.config = train_config
        self.label_map = dict(label_map)
        self.dataset_contract = dataset_contract
        self.preprocessing = dict(preprocessing or {})
        self.modality = modality

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.run_dir = self.config.output_dir / self.config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.config.deterministic:
            from monai.utils import set_determinism

            set_determinism(seed=self.config.seed, use_deterministic_algorithms=True)
        else:
            from monai.utils import set_determinism

            set_determinism(seed=self.config.seed)

        self.model = build_model(model_config).to(self.device)
        self.loss_fn = build_loss(model_config)
        self.optimizer = build_optimizer(self.model, model_config)
        self.scheduler = build_scheduler(self.optimizer, model_config)

        self.amp = bool(self.config.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.history: list[dict[str, Any]] = []
        self._mlflow: Any | None = None

        device_info = describe_device(str(self.device))
        params = count_parameters(self.model)
        log.info(
            "trainer initialised",
            fields={
                "run": self.config.run_name,
                "architecture": model_config.architecture.value,
                "params_m": round(params["trainable"] / 1e6, 2),
                "amp": self.amp,
                **{k: v for k, v in device_info.items() if k != "monai"},
            },
        )
        if self.config.amp and self.device.type != "cuda":
            log.warning(
                "mixed precision requested but no CUDA device is available; "
                "training on CPU in full precision and will be very slow"
            )
        self._write_run_config(device_info, params)

    # ------------------------------------------------------------------
    def _write_run_config(self, device_info: dict[str, Any], params: dict[str, int]) -> None:
        payload = {
            "run_name": self.config.run_name,
            "model": self.model_config.to_dict(),
            "train": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in self.config.__dict__.items()
            },
            "label_map": {s.value: v for s, v in self.label_map.items()},
            "dataset_contract": self.dataset_contract.to_dict(),
            "preprocessing": self.preprocessing,
            "device": device_info,
            "parameters": params,
        }
        (self.run_dir / "run_config.json").write_text(json.dumps(payload, indent=2, default=str))

    def _init_mlflow(self) -> None:
        uri = self.config.mlflow_tracking_uri
        if not uri:
            return
        try:
            import mlflow
        except ImportError:
            log.warning("MLflow tracking requested but mlflow is not installed")
            return
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.config.mlflow_experiment)
        mlflow.start_run(run_name=self.config.run_name)
        mlflow.log_params(
            {
                "architecture": self.model_config.architecture.value,
                "patch_size": "x".join(str(p) for p in self.model_config.patch_size),
                "lr": self.model_config.learning_rate,
                "loss": self.model_config.loss,
                "amp": self.amp,
                "n_classes": self.model_config.out_channels,
                "n_patients": self.dataset_contract.n_patients,
                "datasets": ",".join(self.dataset_contract.dataset_ids),
            }
        )
        self._mlflow = mlflow

    # ------------------------------------------------------------------
    def fit(self, train_loader: Any, val_loader: Any) -> TrainingResult:
        torch = self.torch
        started = time.time()
        self._init_mlflow()

        best_metric = -1.0
        best_epoch = -1
        best_checkpoint: Path | None = None
        best_per_structure: dict[str, float] = {}
        validations_without_improvement = 0
        epochs_trained = 0

        try:
            for epoch in range(1, self.config.max_epochs + 1):
                epochs_trained = epoch
                train_loss = self._train_one_epoch(train_loader, epoch)
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 5),
                    "lr": round(self.optimizer.param_groups[0]["lr"], 8),
                }

                should_validate = (
                    epoch % self.config.val_interval == 0
                    or epoch == self.config.max_epochs
                )
                if should_validate:
                    mean_dice, per_structure = self._validate(val_loader)
                    record["val_mean_dice"] = round(mean_dice, 5)
                    record["val_per_structure"] = {
                        k: round(v, 4) for k, v in per_structure.items()
                    }

                    if mean_dice > best_metric:
                        best_metric = mean_dice
                        best_epoch = epoch
                        best_per_structure = per_structure
                        best_checkpoint = self._save_checkpoint("best", epoch, mean_dice)
                        validations_without_improvement = 0
                        record["best"] = True
                    else:
                        validations_without_improvement += 1

                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        if should_validate:
                            self.scheduler.step(record.get("val_mean_dice", 0.0))
                    else:
                        self.scheduler.step()

                self.history.append(record)
                self._log_epoch(record)

                if self.config.save_every_epoch:
                    self._save_checkpoint(f"epoch{epoch:04d}", epoch, record.get("val_mean_dice", 0.0))

                patience = self.config.early_stopping_patience
                if patience is not None and validations_without_improvement >= patience:
                    log.info(
                        "early stopping",
                        fields={
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_mean_dice": round(best_metric, 4),
                            "validations_without_improvement": validations_without_improvement,
                        },
                    )
                    break
        except KeyboardInterrupt:  # pragma: no cover - interactive
            log.warning("training interrupted; saving current state")
        finally:
            last_checkpoint = self._save_checkpoint("last", epochs_trained, best_metric)
            (self.run_dir / "history.json").write_text(
                json.dumps(self.history, indent=2, default=str)
            )
            if self._mlflow is not None:
                self._mlflow.end_run()

        result = TrainingResult(
            run_id=self.config.run_name,
            run_dir=self.run_dir,
            best_metric=best_metric,
            best_epoch=best_epoch,
            epochs_trained=epochs_trained,
            history=self.history,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            per_structure_dice=best_per_structure,
            wall_time_s=time.time() - started,
        )
        log.info("training complete", fields=result.to_dict() | {"history": "<omitted>"})
        return result

    # ------------------------------------------------------------------
    def _train_one_epoch(self, loader: Any, epoch: int) -> float:
        torch = self.torch
        self.model.train()
        total_loss = 0.0
        n_steps = 0
        accum = self.config.grad_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                if not torch.isfinite(loss):
                    # A non-finite loss poisons the weights the moment it is
                    # backpropagated. Skip the step and say so, rather than
                    # discovering NaN weights several epochs later.
                    log.warning(
                        "non-finite loss; step skipped",
                        fields={"epoch": epoch, "step": step, "loss": float(loss)},
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                scaled = loss / accum

            self.scaler.scale(scaled).backward()

            if step % accum == 0:
                if self.config.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach())
            n_steps += 1
            if n_steps % self.config.log_every_n_steps == 0:
                log.info(
                    "training step",
                    fields={
                        "epoch": epoch,
                        "step": n_steps,
                        "loss": round(total_loss / n_steps, 5),
                    },
                )

        if n_steps == 0:
            raise ConfigurationError(
                "no training steps completed; the dataloader yielded nothing"
            )
        return total_loss / n_steps

    # ------------------------------------------------------------------
    def _validate(self, loader: Any) -> tuple[float, dict[str, float]]:
        torch = self.torch
        from monai.data import decollate_batch
        from monai.inferers import sliding_window_inference
        from monai.metrics import DiceMetric
        from monai.transforms import AsDiscrete, Compose

        self.model.eval()
        n_classes = self.model_config.out_channels
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=n_classes)])
        post_label = Compose([AsDiscrete(to_onehot=n_classes)])
        # include_background=False: a 99% correct background inflates mean Dice and
        # tells us nothing about the structures being measured.
        metric = DiceMetric(include_background=False, reduction="mean_batch")

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                with torch.amp.autocast("cuda", enabled=self.amp):
                    logits = sliding_window_inference(
                        inputs=images,
                        roi_size=self.model_config.patch_size,
                        sw_batch_size=self.config.sw_batch_size,
                        predictor=self.model,
                        overlap=self.config.sw_overlap,
                        mode="gaussian",
                    )
                preds = [post_pred(p) for p in decollate_batch(logits.float())]
                truths = [post_label(t) for t in decollate_batch(labels)]
                metric(y_pred=preds, y=truths)

        per_class = metric.aggregate()
        metric.reset()
        values = [float(v) for v in per_class]

        inverted = {v: k for k, v in self.label_map.items()}
        per_structure: dict[str, float] = {}
        # DiceMetric with include_background=False returns classes 1..N-1 in order.
        for offset, value in enumerate(values, start=1):
            structure = inverted.get(offset)
            name = structure.value if structure else f"class_{offset}"
            per_structure[name] = value if math.isfinite(value) else float("nan")

        finite = [v for v in values if math.isfinite(v)]
        mean_dice = float(sum(finite) / len(finite)) if finite else 0.0
        return mean_dice, per_structure

    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag: str, epoch: int, metric: float) -> Path:
        path = self.run_dir / f"{tag}.pt"
        self.torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "metric": metric,
                "model_config": self.model_config.to_dict(),
                "label_map": {s.value: v for s, v in self.label_map.items()},
                "preprocessing": self.preprocessing,
                "run_name": self.config.run_name,
            },
            path,
        )
        return path

    def _log_epoch(self, record: dict[str, Any]) -> None:
        log.info("epoch", fields=record)
        if self._mlflow is not None:
            flat = {
                k: v for k, v in record.items() if isinstance(v, (int, float)) and k != "epoch"
            }
            for structure, value in (record.get("val_per_structure") or {}).items():
                flat[f"dice_{structure}"] = value
            self._mlflow.log_metrics(flat, step=int(record["epoch"]))

    # ------------------------------------------------------------------
    def build_model_card(
        self,
        result: TrainingResult,
        *,
        model_id: str,
        version: str,
        intended_use: str,
        limitations: str = "",
        contraindications: Sequence[str] = (),
        created_by: str = "",
        external_test_metrics: dict[str, Any] | None = None,
    ) -> ModelCard:
        """Assemble the model card for a completed run."""
        torch, monai = require_torch()
        return ModelCard(
            model_id=model_id,
            version=version,
            task=ModelTask.SEGMENTATION,
            architecture=self.model_config.architecture,
            dataset_contract=self.dataset_contract,
            label_map=self.label_map,
            preprocessing=self.preprocessing,
            modality=self.modality,
            input_channels=self.model_config.in_channels,
            patch_size=self.model_config.patch_size,
            metrics=ModelMetrics(
                internal_validation={
                    "mean_dice": round(result.best_metric, 4),
                    "per_structure_dice": result.per_structure_dice,
                    "best_epoch": result.best_epoch,
                    "epochs_trained": result.epochs_trained,
                },
                external_test=external_test_metrics or {},
            ),
            intended_use=intended_use,
            limitations=limitations,
            contraindications=tuple(contraindications),
            framework="pytorch",
            framework_version=f"torch {torch.__version__} / monai {monai.__version__}",
            created_by=created_by,
            training_run_id=result.run_id,
            git_commit=_git_commit(),
            notes=f"Trained with {self.model_config.loss} loss, amp={self.amp}.",
        )

    def register(
        self,
        result: TrainingResult,
        *,
        model_id: str,
        version: str,
        intended_use: str,
        registry: ModelRegistry | None = None,
        **card_kwargs: Any,
    ) -> ModelCard:
        """Build the card and register the best checkpoint."""
        card = self.build_model_card(
            result,
            model_id=model_id,
            version=version,
            intended_use=intended_use,
            **card_kwargs,
        )
        reg = registry or ModelRegistry()
        return reg.register(card, checkpoint_path=result.best_checkpoint)


def _git_commit() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - git may be unavailable (e.g. in Colab)
        return ""
