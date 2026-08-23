"""
Priority 1.1 — Run AI segmentation inference on a test case.

Usage:
    python scripts/run_inference.py                    # Default: case 405
    python scripts/run_inference.py --case 410
    python scripts/run_inference.py --nifti path/to/image.nii.gz

Outputs:
    - Predicted label volume saved as NIfTI
    - Mid-slice visualization PNG (axial, sagittal, coronal)
    - Per-class voxel counts printed
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, r'E:\KneeAI\backend')

from app.config import MODELS_DIR, PROCESSED_DATA, LABELS_MAP
from app.segmentation.inference import KneeSegmentationModel
from app.data_loader import OAIZIBDataLoader


def save_slice_visualization(image: np.ndarray, prediction: np.ndarray, case_id: int, output_dir: Path):
    """Save mid-slice overlay PNGs for visual QC."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("  matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Color map for labels
    colors = ['black', '#E8D9BF', '#3399FF', '#D9CC80', '#FF6666', '#66FF66']
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Case {case_id} — AI Segmentation', fontsize=14)

    # Mid slices
    mid_sag = image.shape[0] // 2
    mid_cor = image.shape[1] // 2
    mid_axi = image.shape[2] // 2

    # Row 1: MRI
    axes[0, 0].imshow(image[mid_sag, :, :], cmap='gray', aspect='auto')
    axes[0, 0].set_title(f'Sagittal (slice {mid_sag})')
    axes[0, 1].imshow(image[:, mid_cor, :], cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Coronal (slice {mid_cor})')
    axes[0, 2].imshow(image[:, :, mid_axi], cmap='gray', aspect='auto')
    axes[0, 2].set_title(f'Axial (slice {mid_axi})')

    # Row 2: Prediction overlay
    axes[1, 0].imshow(image[mid_sag, :, :], cmap='gray', aspect='auto')
    axes[1, 0].imshow(prediction[mid_sag, :, :], cmap=cmap, alpha=0.4, vmin=0, vmax=5, aspect='auto')
    axes[1, 0].set_title('Sagittal + Segmentation')

    axes[1, 1].imshow(image[:, mid_cor, :], cmap='gray', aspect='auto')
    axes[1, 1].imshow(prediction[:, mid_cor, :], cmap=cmap, alpha=0.4, vmin=0, vmax=5, aspect='auto')
    axes[1, 1].set_title('Coronal + Segmentation')

    axes[1, 2].imshow(image[:, :, mid_axi], cmap='gray', aspect='auto')
    axes[1, 2].imshow(prediction[:, :, mid_axi], cmap=cmap, alpha=0.4, vmin=0, vmax=5, aspect='auto')
    axes[1, 2].set_title('Axial + Segmentation')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    png_path = output_dir / f"case_{case_id:03d}_segmentation.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization: {png_path}")


def run_inference_on_case(case_id: int, model: KneeSegmentationModel, output_dir: Path):
    """Run inference on an OAIZIB-CM case by ID."""
    loader = OAIZIBDataLoader()
    case = loader.load_case(case_id)

    image = case['image']
    voxel_size = case['voxel_size']
    print(f"  Image shape: {image.shape}")
    print(f"  Voxel size: {voxel_size}")

    # Run AI prediction
    print("  Running inference (sliding window)...")
    prediction = model.predict(image, voxel_size)

    # Print per-class stats
    print(f"\n  Prediction summary:")
    for label_id in range(1, 6):
        count = (prediction == label_id).sum()
        vol_mm3 = count * np.prod(voxel_size)
        print(f"    {LABELS_MAP[label_id]:<25}: {count:>8} voxels ({vol_mm3/1000:.2f} cm³)")

    # Save prediction as NIfTI
    output_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = output_dir / f"case_{case_id:03d}_prediction.nii.gz"
    affine = case.get('affine', np.eye(4))
    nib.save(nib.Nifti1Image(prediction.astype(np.int16), affine), str(nifti_path))
    print(f"\n  Saved prediction: {nifti_path}")

    # Visualization
    save_slice_visualization(image, prediction, case_id, output_dir)

    return prediction, case


def run_inference_on_nifti(nifti_path: Path, model: KneeSegmentationModel, output_dir: Path):
    """Run inference on an arbitrary NIfTI file."""
    print(f"  Loading: {nifti_path}")
    img = nib.load(str(nifti_path))
    image = img.get_fdata()
    voxel_size = img.header.get_zooms()[:3]

    print(f"  Image shape: {image.shape}")
    print(f"  Voxel size: {voxel_size}")

    print("  Running inference (sliding window)...")
    prediction = model.predict(image, voxel_size)

    # Stats
    print(f"\n  Prediction summary:")
    for label_id in range(1, 6):
        count = (prediction == label_id).sum()
        vol_mm3 = count * np.prod(voxel_size)
        print(f"    {LABELS_MAP[label_id]:<25}: {count:>8} voxels ({vol_mm3/1000:.2f} cm³)")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
    nifti_out = output_dir / f"{stem}_prediction.nii.gz"
    nib.save(nib.Nifti1Image(prediction.astype(np.int16), img.affine), str(nifti_out))
    print(f"\n  Saved prediction: {nifti_out}")

    save_slice_visualization(image, prediction, 0, output_dir)

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Run KneeAI segmentation inference")
    parser.add_argument("--case", type=int, default=405, help="OAIZIB-CM case ID (default: 405)")
    parser.add_argument("--nifti", type=str, default=None, help="Path to arbitrary NIfTI file")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else PROCESSED_DATA / "predictions"

    print("=" * 60)
    print("KneeAI — AI Segmentation Inference")
    print("=" * 60)

    # Load model
    model = KneeSegmentationModel()
    print(f"\nLoading model from: {model.model_path}")
    if not model.load():
        print("ERROR: Model not found. Run training first.")
        sys.exit(1)
    print("  Model loaded successfully.")

    if args.nifti:
        print(f"\n--- Inference on: {args.nifti} ---")
        run_inference_on_nifti(Path(args.nifti), model, output_dir)
    else:
        print(f"\n--- Inference on case {args.case} ---")
        run_inference_on_case(args.case, model, output_dir)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
