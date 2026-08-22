"""Pretrained encoder weight loading, with explicit match reporting.

The failure this module exists to prevent: a "pretrained" model that is actually
randomly initialised. It happens when a checkpoint's key names do not line up with
the target module's — ``module.`` prefixes, a ``state_dict`` wrapper, a renamed
submodule — and the load is done with ``strict=False``. PyTorch silently loads
nothing, training proceeds, and the run is reported as fine-tuned when it was trained
from scratch. Nothing in the loss curve gives it away.

So every load here **reports exactly which tensors matched**, and refuses by default
when the matched fraction is implausibly low.

Torch is imported lazily. The descriptors and the report are torch-free so a manifest
can be inspected without the training stack.

Generic by design
-----------------
:func:`load_encoder_weights` knows nothing about SwinUNETR specifically. It unwraps
common checkpoint containers, strips common prefixes, applies an optional caller
remap, then classifies every target parameter. A different SSL encoder — a future
fastMRI-pretrained one, say — drops in by supplying a different
:class:`PretrainedSource`, with no change here.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from kneeai.core.errors import ConfigurationError
from kneeai.logging import get_logger

__all__ = [
    "PretrainedSource",
    "EncoderLoadReport",
    "MONAI_SSL_SWINUNETR",
    "PRETRAINED_SOURCES",
    "DEFAULT_PRETRAINED_DIR",
    "download_pretrained",
    "load_encoder_weights",
    "sha256_of",
]

log = get_logger("ml.pretrained")

DEFAULT_PRETRAINED_DIR = Path("artifacts/pretrained")

#: Container keys a checkpoint may hide its tensors under.
_CONTAINER_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "network_weights",
    "student",
    "teacher",
    "weights",
)

#: Prefixes to strip. ``module.`` comes from DataParallel/DDP; the others from
#: wrapping the encoder inside a larger SSL head during pretraining.
_STRIP_PREFIXES = ("module.", "backbone.", "net.", "encoder.", "model.")


def sha256_of(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PretrainedSource:
    """A named pretrained encoder, and how to obtain and apply it."""

    identifier: str
    url: str
    filename: str

    #: Which architecture this checkpoint is for. Loading it into anything else is
    #: refused, because a partial match across architectures is exactly the silent
    #: failure this module guards against.
    architecture: str

    description: str = ""

    #: Expected SHA256, once verified. ``None`` means "record on first download";
    #: the actual digest is always written into the sidecar and the model card.
    expected_sha256: str | None = None

    #: Submodule prefix the weights belong to in the target model, e.g. ``swinViT.``
    #: Used to report an encoder-specific match fraction rather than a whole-model
    #: one, which would look poor simply because the decoder is never pretrained.
    target_prefix: str = ""

    #: Optional key rewrite applied after container unwrapping and prefix stripping.
    key_remap: Mapping[str, str] = field(default_factory=dict)

    citation: str = ""
    license: str = ""

    def local_path(self, root: str | os.PathLike[str] | None = None) -> Path:
        return Path(root or DEFAULT_PRETRAINED_DIR) / self.filename

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "url": self.url,
            "filename": self.filename,
            "architecture": self.architecture,
            "description": self.description,
            "expected_sha256": self.expected_sha256,
            "target_prefix": self.target_prefix,
            "citation": self.citation,
            "license": self.license,
        }


#: MONAI's self-supervised pretrained SwinUNETR encoder.
#:
#: Pretrained on ~5k unlabelled 3D CT volumes with a combined contrastive /
#: inpainting / rotation objective (Tang et al., CVPR 2022). It is the encoder the
#: MONAI SwinUNETR tutorials fine-tune from.
#:
#: Note the domain gap, which is stated rather than glossed: this is a **CT**-heavy
#: pretraining corpus and our target is knee **MR**. Transfer is still expected to
#: help at a ≤40-case budget — low-level 3D structure is largely modality-agnostic —
#: but the gap is real and belongs on the model card, not hidden.
MONAI_SSL_SWINUNETR = PretrainedSource(
    identifier="monai-ssl-swinunetr-btcv",
    url=(
        "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/"
        "0.8.1/model_swinvit.pt"
    ),
    filename="monai_ssl_swinvit.pt",
    architecture="swin_unetr",
    target_prefix="swinViT.",
    description=(
        "MONAI self-supervised pretrained Swin UNETR encoder (swinViT). Pretrained on "
        "unlabelled 3D CT with contrastive + inpainting + rotation objectives."
    ),
    citation=(
        "Tang Y, Yang D, Li W, Roth HR, Landman B, Xu D, Nath V, Hatamizadeh A. "
        "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image "
        "Analysis. CVPR 2022."
    ),
    license="Apache-2.0 (MONAI project)",
)

PRETRAINED_SOURCES: dict[str, PretrainedSource] = {
    MONAI_SSL_SWINUNETR.identifier: MONAI_SSL_SWINUNETR,
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class EncoderLoadReport:
    """Exactly what happened during a weight load."""

    source_identifier: str
    checkpoint_path: str
    sha256: str
    architecture: str

    matched: list[str] = field(default_factory=list)
    shape_mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list
    )
    missing_in_checkpoint: list[str] = field(default_factory=list)
    unused_in_checkpoint: list[str] = field(default_factory=list)

    n_target_params: int = 0
    n_checkpoint_params: int = 0
    target_prefix: str = ""
    container_key: str | None = None
    stripped_prefix: str | None = None
    warnings: list[str] = field(default_factory=list)

    # --- fractions ---
    @property
    def matched_fraction(self) -> float:
        """Matched tensors as a fraction of all target tensors."""
        return len(self.matched) / self.n_target_params if self.n_target_params else 0.0

    @property
    def encoder_matched_fraction(self) -> float:
        """Matched tensors within the encoder submodule only.

        Reported separately because the decoder is never pretrained: a whole-model
        fraction of ~0.5 can be a perfect encoder load, and judging the load on it
        would either raise false alarms or mask real failures.
        """
        if not self.target_prefix:
            return self.matched_fraction
        encoder_targets = [
            k
            for k in self.matched + self.missing_in_checkpoint
            if k.startswith(self.target_prefix)
        ]
        if not encoder_targets:
            return 0.0
        encoder_matched = [k for k in self.matched if k.startswith(self.target_prefix)]
        return len(encoder_matched) / len(encoder_targets)

    @property
    def loaded_anything(self) -> bool:
        return bool(self.matched)

    def summary(self) -> str:
        lines = [
            f"Pretrained encoder load: {self.source_identifier}",
            f"  checkpoint      : {self.checkpoint_path}",
            f"  sha256          : {self.sha256}",
            f"  architecture    : {self.architecture}",
            f"  container key   : {self.container_key or '<none, top-level tensors>'}",
            f"  stripped prefix : {self.stripped_prefix or '<none>'}",
            f"  target tensors  : {self.n_target_params}",
            f"  checkpoint tens.: {self.n_checkpoint_params}",
            f"  MATCHED         : {len(self.matched)} "
            f"({self.matched_fraction * 100:.1f}% of model, "
            f"{self.encoder_matched_fraction * 100:.1f}% of encoder)",
            f"  shape mismatch  : {len(self.shape_mismatched)}",
            f"  missing         : {len(self.missing_in_checkpoint)}",
            f"  unused          : {len(self.unused_in_checkpoint)}",
        ]
        if self.shape_mismatched:
            lines.append("  shape mismatches (first 10):")
            for name, target, source in self.shape_mismatched[:10]:
                lines.append(f"    {name}: model {target} vs checkpoint {source}")
        if self.matched:
            lines.append(f"  matched examples: {self.matched[:3]}")
        if self.missing_in_checkpoint:
            lines.append(f"  missing examples: {self.missing_in_checkpoint[:3]}")
        if self.unused_in_checkpoint:
            lines.append(f"  unused examples : {self.unused_in_checkpoint[:3]}")
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_identifier": self.source_identifier,
            "checkpoint_path": self.checkpoint_path,
            "sha256": self.sha256,
            "architecture": self.architecture,
            "container_key": self.container_key,
            "stripped_prefix": self.stripped_prefix,
            "n_target_params": self.n_target_params,
            "n_checkpoint_params": self.n_checkpoint_params,
            "n_matched": len(self.matched),
            "n_shape_mismatched": len(self.shape_mismatched),
            "n_missing_in_checkpoint": len(self.missing_in_checkpoint),
            "n_unused_in_checkpoint": len(self.unused_in_checkpoint),
            "matched_fraction": round(self.matched_fraction, 4),
            "encoder_matched_fraction": round(self.encoder_matched_fraction, 4),
            "target_prefix": self.target_prefix,
            "matched": self.matched,
            "shape_mismatched": [
                {"name": n, "model_shape": list(t), "checkpoint_shape": list(s)}
                for n, t, s in self.shape_mismatched
            ],
            "missing_in_checkpoint": self.missing_in_checkpoint,
            "unused_in_checkpoint": self.unused_in_checkpoint,
            "warnings": self.warnings,
        }

    def write(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def download_pretrained(
    source: PretrainedSource | str = MONAI_SSL_SWINUNETR,
    *,
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> tuple[Path, str]:
    """Download a pretrained checkpoint and record its SHA256.

    Returns ``(path, sha256)``. Cached: an existing file with a matching digest is
    not re-downloaded.
    """
    if isinstance(source, str):
        if source not in PRETRAINED_SOURCES:
            raise ConfigurationError(
                f"unknown pretrained source {source!r}; known: "
                f"{sorted(PRETRAINED_SOURCES)}"
            )
        source = PRETRAINED_SOURCES[source]

    target = source.local_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        digest = sha256_of(target)
        log.info(
            "pretrained checkpoint already present",
            fields={"path": str(target), "sha256": digest[:16]},
        )
    else:
        log.info("downloading pretrained checkpoint", fields={"url": source.url})
        with urllib.request.urlopen(source.url, timeout=600) as response:
            payload = response.read()
        target.write_bytes(payload)
        digest = sha256_of(target)
        log.info(
            "pretrained checkpoint downloaded",
            fields={
                "path": str(target),
                "bytes": len(payload),
                "sha256": digest[:16],
            },
        )

    if source.expected_sha256 and digest != source.expected_sha256:
        raise ConfigurationError(
            f"{source.identifier} SHA256 mismatch.\n"
            f"  expected: {source.expected_sha256}\n"
            f"  actual  : {digest}\n"
            "Refusing to use a checkpoint whose bytes differ from the recorded "
            "digest — the provenance recorded on the model card would be false."
        )

    # The digest travels with the file so the model card can cite it even if the
    # download step is not re-run.
    (target.with_suffix(target.suffix + ".json")).write_text(
        json.dumps({**source.to_dict(), "sha256": digest, "bytes": target.stat().st_size}, indent=2)
    )
    return target, digest


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _unwrap(checkpoint: Any) -> tuple[Mapping[str, Any], str | None]:
    """Find the tensor dict inside a checkpoint, returning the container key used."""
    if not isinstance(checkpoint, Mapping):
        raise ConfigurationError(
            f"checkpoint is a {type(checkpoint).__name__}, not a dict of tensors"
        )
    # A dict whose values are already tensors is the tensor dict itself.
    if any(hasattr(v, "shape") for v in checkpoint.values()):
        return checkpoint, None
    for key in _CONTAINER_KEYS:
        inner = checkpoint.get(key)
        if isinstance(inner, Mapping) and inner:
            return inner, key
    raise ConfigurationError(
        f"could not find tensors in the checkpoint. Top-level keys: "
        f"{sorted(checkpoint)[:12]}"
    )


def _strip_common_prefix(
    tensors: Mapping[str, Any], target_keys: Iterable[str]
) -> tuple[dict[str, Any], str | None]:
    """Strip a wrapper prefix if doing so improves overlap with the target keys.

    Chosen by measuring overlap rather than assuming: a prefix that *reduces* the
    match is not applied.
    """
    targets = set(target_keys)
    baseline = sum(1 for k in tensors if k in targets)

    best_prefix: str | None = None
    best_overlap = baseline
    for prefix in _STRIP_PREFIXES:
        if not any(k.startswith(prefix) for k in tensors):
            continue
        candidate = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in tensors.items()
        }
        overlap = sum(1 for k in candidate if k in targets)
        if overlap > best_overlap:
            best_overlap = overlap
            best_prefix = prefix

    if best_prefix is None:
        return dict(tensors), None
    return {
        (k[len(best_prefix) :] if k.startswith(best_prefix) else k): v
        for k, v in tensors.items()
    }, best_prefix


def load_encoder_weights(
    model: Any,
    source: PretrainedSource | str = MONAI_SSL_SWINUNETR,
    *,
    checkpoint_path: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    architecture: str | None = None,
    min_encoder_match: float = 0.5,
    strict_architecture: bool = True,
    key_remap: Mapping[str, str] | Callable[[str], str] | None = None,
    allow_download: bool = True,
) -> EncoderLoadReport:
    """Load pretrained encoder weights into ``model``, reporting every tensor.

    Args:
        min_encoder_match: refuse the load when the fraction of matched *encoder*
            tensors falls below this. The default of 0.5 is deliberately blunt: a
            genuine prefix/architecture mismatch produces a fraction near zero, and
            silently continuing would mean training a randomly-initialised model and
            reporting it as fine-tuned.
        strict_architecture: refuse when the source's architecture does not match.

    Raises:
        ConfigurationError: on architecture mismatch, or when too little matched.
    """
    from kneeai.ml.models import require_torch

    torch, _monai = require_torch()

    if isinstance(source, str):
        if source not in PRETRAINED_SOURCES:
            raise ConfigurationError(
                f"unknown pretrained source {source!r}; known: "
                f"{sorted(PRETRAINED_SOURCES)}"
            )
        source = PRETRAINED_SOURCES[source]

    if architecture and strict_architecture and architecture != source.architecture:
        raise ConfigurationError(
            f"{source.identifier} provides weights for {source.architecture!r} but the "
            f"model being built is {architecture!r}. A cross-architecture load matches "
            "a handful of tensors by coincidence and leaves the rest random, which "
            "looks like a successful pretrained load and is not one. Pass "
            "strict_architecture=False only if you have a specific reason."
        )

    # --- obtain the checkpoint ---
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise ConfigurationError(f"checkpoint not found: {path}")
        digest = sha256_of(path)
    else:
        path = source.local_path(root)
        if path.exists():
            digest = sha256_of(path)
        elif allow_download:
            path, digest = download_pretrained(source, root=root)
        else:
            raise ConfigurationError(
                f"{path} does not exist and downloading is disabled. Run "
                "download_pretrained() first."
            )

    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    tensors, container_key = _unwrap(raw)

    target_state = model.state_dict()
    tensors, stripped = _strip_common_prefix(tensors, target_state.keys())

    # --- caller remap, applied last so it can override everything above ---
    if key_remap is not None:
        if callable(key_remap):
            tensors = {key_remap(k): v for k, v in tensors.items()}
        else:
            tensors = {key_remap.get(k, k): v for k, v in tensors.items()}
    if source.key_remap:
        tensors = {source.key_remap.get(k, k): v for k, v in tensors.items()}

    report = EncoderLoadReport(
        source_identifier=source.identifier,
        checkpoint_path=str(path),
        sha256=digest,
        architecture=source.architecture,
        n_target_params=len(target_state),
        n_checkpoint_params=len(tensors),
        target_prefix=source.target_prefix,
        container_key=container_key,
        stripped_prefix=stripped,
    )

    # --- classify every target tensor ---
    loadable: dict[str, Any] = {}
    for name, target_tensor in target_state.items():
        if name not in tensors:
            report.missing_in_checkpoint.append(name)
            continue
        source_tensor = tensors[name]
        target_shape = tuple(target_tensor.shape)
        source_shape = tuple(getattr(source_tensor, "shape", ()))
        if target_shape != source_shape:
            report.shape_mismatched.append((name, target_shape, source_shape))
            continue
        loadable[name] = source_tensor
        report.matched.append(name)

    report.unused_in_checkpoint = sorted(set(tensors) - set(target_state))

    # --- decide before mutating the model ---
    if not report.loaded_anything:
        raise ConfigurationError(
            "pretrained load matched ZERO tensors — the model would be randomly "
            "initialised while being reported as fine-tuned.\n"
            f"{report.summary()}\n"
            f"Checkpoint key examples : {sorted(tensors)[:5]}\n"
            f"Model key examples      : {sorted(target_state)[:5]}\n"
            "Supply key_remap= to reconcile the naming."
        )

    if report.encoder_matched_fraction < min_encoder_match:
        raise ConfigurationError(
            f"only {report.encoder_matched_fraction * 100:.1f}% of encoder tensors "
            f"matched, below the {min_encoder_match * 100:.0f}% threshold. This is a "
            "partial load, which trains mostly-random weights under a 'pretrained' "
            f"label.\n{report.summary()}"
        )

    missing_keys, unexpected_keys = model.load_state_dict(loadable, strict=False)
    # Sanity: load_state_dict's own view must agree with ours.
    if len(missing_keys) != len(target_state) - len(loadable):
        report.warnings.append(
            f"load_state_dict reported {len(missing_keys)} missing keys but "
            f"{len(target_state) - len(loadable)} were expected; investigate before "
            "trusting this load"
        )

    if report.shape_mismatched:
        report.warnings.append(
            f"{len(report.shape_mismatched)} tensor(s) matched by name but not by "
            "shape and were skipped. This usually means a different patch size, "
            "feature width or input channel count than the checkpoint was trained "
            "with."
        )

    log.info(
        "pretrained encoder loaded",
        fields={
            "source": source.identifier,
            "sha256": digest[:16],
            "matched": len(report.matched),
            "encoder_matched_pct": round(report.encoder_matched_fraction * 100, 1),
            "shape_mismatched": len(report.shape_mismatched),
            "missing": len(report.missing_in_checkpoint),
            "unused": len(report.unused_in_checkpoint),
        },
    )
    print(report.summary(), flush=True)
    return report
