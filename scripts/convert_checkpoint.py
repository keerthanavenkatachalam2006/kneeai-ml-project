"""Re-save a training checkpoint in the format ``kneeai.ml.infer`` expects.

The training scripts save a compact dict of their own. The platform's predictor
expects three specific keys, and it is strict about all of them on purpose:

``model_state_dict``
    Looked up by name, falling back to the whole file. A nested ``state_dict`` key
    is read as though the metadata entries were parameter tensors, which surfaces
    as "63 missing and 8 unexpected".

``label_map``
    Authoritative over the model card. Must include background, because
    ``infer.py`` derives ``out_channels`` from ``len(label_map)`` — omit it and the
    predictor builds a 5-channel head for a 6-channel checkpoint.

``model_config``
    A serialised :class:`ModelConfig`. Without it the predictor falls back to
    defaults — ``features=(32, 64, 128, 256, 320)`` — and every tensor shape
    mismatches. ``features`` here must match what training actually built.

No weights are altered. This is a container change only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kneeai.ml.models import ModelConfig
from kneeai.ml.registry import Architecture

REPO = Path(__file__).resolve().parents[1]

LABEL_MAP_WITH_BACKGROUND = {
    "background": 0,
    "femur": 1,
    "femoral_cartilage": 2,
    "tibia": 3,
    "medial_tibial_cartilage": 4,
    "lateral_tibial_cartilage": 5,
}

# What scripts/train_oaizib_local.py and train_oaizib_v2.py actually construct.
TRAINED_FEATURES = (16, 32, 64, 128, 256)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/models/seg_oaizib/checkpoint.pt")
    ap.add_argument("--out", default=None, help="defaults to overwriting --checkpoint")
    args = ap.parse_args()

    src = REPO / args.checkpoint
    dst = REPO / (args.out or args.checkpoint)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise SystemExit(f"no state dict found in {src}; keys: {list(ckpt)}")

    patch = int(ckpt.get("patch", 64))
    n_classes = int(ckpt.get("n_classes", len(LABEL_MAP_WITH_BACKGROUND)))
    if n_classes != len(LABEL_MAP_WITH_BACKGROUND):
        raise SystemExit(
            f"checkpoint has {n_classes} output classes but the label map declares "
            f"{len(LABEL_MAP_WITH_BACKGROUND)}"
        )

    config = ModelConfig(
        architecture=Architecture.UNET,
        in_channels=1,
        out_channels=n_classes,
        patch_size=(patch, patch, patch),
        features=TRAINED_FEATURES,
        dropout=0.0,
        loss="dice_ce",
        include_background_in_loss=False,
        learning_rate=1e-3,
        weight_decay=1e-5,
        scheduler="cosine",
        max_epochs=int(ckpt.get("epoch", 0)),
    )

    payload = {
        "model_state_dict": state,
        "label_map": LABEL_MAP_WITH_BACKGROUND,
        "model_config": config.to_dict(),
        # Carry the training metadata forward so the file stays self-describing.
        "epoch": ckpt.get("epoch"),
        "val_dice_mean": ckpt.get("val_dice_mean"),
        "val_dice_per_structure": ckpt.get("val_dice_per_structure"),
        "structures": ckpt.get("structures"),
        "n_classes": n_classes,
        "spacing_mm": ckpt.get("spacing_mm"),
        "patch": patch,
        "run": ckpt.get("run", "v1"),
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)

    print(f"source      : {src.relative_to(REPO)}")
    print(f"written     : {dst.relative_to(REPO)}")
    print(f"tensors     : {len(state)}")
    print(f"out_channels: {n_classes}   patch: {patch}^3")
    print(f"features    : {TRAINED_FEATURES}")
    print(f"epoch       : {payload['epoch']}   val_dice: {payload['val_dice_mean']}")

    # Prove the predictor's own builder produces a shape-compatible model.
    from kneeai.ml.models import build_model

    probe = build_model(config)
    missing, unexpected = probe.load_state_dict(state, strict=False)
    print()
    print(f"load into build_model(config): missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print("  MISMATCH — first missing:", list(missing)[:3])
        print("  first unexpected:", list(unexpected)[:3])
        raise SystemExit(1)
    print("  PASS — architecture round-trips cleanly")


if __name__ == "__main__":
    main()
