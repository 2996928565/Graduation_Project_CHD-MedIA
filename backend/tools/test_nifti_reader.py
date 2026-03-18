"""
NIfTI image + label reading test script (MM-WHS 2017)
Usage:
    # With label (segmentation ground truth):
    python backend/tools/test_nifti_reader.py \
        --image path/to/mr_train_1001_image.nii.gz \
        --label path/to/mr_train_1001_label.nii.gz \
        --out_dir backend/tools/output_slices

    # Without label (image-only):
    python backend/tools/test_nifti_reader.py \
        --image path/to/patient_scan.nii.gz \
        --out_dir backend/tools/output_slices
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ── optional: matplotlib for slice visualization ─────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # works without a display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed, skipping slice visualization.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core I/O & info printing
# ─────────────────────────────────────────────────────────────────────────────

def load_nifti(path: str) -> sitk.Image:
    """Load a NIfTI file (.nii / .nii.gz)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    img = sitk.ReadImage(str(path))
    return img


def print_info(img: sitk.Image, name: str, is_label: bool = False) -> None:
    """Print basic metadata for an image or label volume."""
    arr = sitk.GetArrayFromImage(img)  # shape: (D, H, W)
    print(f"\n{'─'*60}")
    print(f"  [{name}]")
    print(f"  Shape  (D x H x W)   : {arr.shape}")
    print(f"  Dtype                : {arr.dtype}")
    print(f"  Spacing (mm)         : {img.GetSpacing()}")
    print(f"  Origin               : {img.GetOrigin()}")
    print(f"  Direction            : {img.GetDirection()}")
    print(f"  Intensity range      : min={arr.min():.2f}  max={arr.max():.2f}")
    if is_label:
        unique = np.unique(arr)
        print(f"  Label values (unique): {unique}")
        for v in unique:
            count = (arr == v).sum()
            pct = count / arr.size * 100
            anat = MMWHS_LABEL_MAP.get(int(v), (0, 0, 0, f"unknown {int(v)}"))[3]
            remap_id = MMWHS_REMAP.get(int(v), "?")
            print(f"    {anat:<24s} orig={int(v):4d}  remap->{remap_id}  "
                  f"{count:>10,} voxels  ({pct:.2f}%)")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# MM-WHS 2017 label definitions
# ─────────────────────────────────────────────────────────────────────────────

# Official MM-WHS 2017 label pixel values (NOT consecutive 0-7 — must remap before training!)
MMWHS_LABEL_MAP = {
    0:   (0.00, 0.00, 0.00, "Background        (BG)"),
    500: (0.90, 0.20, 0.20, "LV blood pool     (LV)"),
    600: (0.20, 0.60, 0.90, "RV blood pool     (RV)"),
    420: (0.20, 0.90, 0.40, "LA blood pool     (LA)"),
    550: (0.90, 0.70, 0.10, "RA blood pool     (RA)"),
    205: (0.50, 0.50, 0.90, "LV myocardium     (Myo)"),
    820: (0.80, 0.20, 0.90, "Ascending aorta   (AO)"),
    850: (0.90, 0.50, 0.10, "Pulmonary artery  (PA)"),
}

# Remapping table: original value -> consecutive class id 0-7
# i.e. {0:0, 500:1, 600:2, 420:3, 550:4, 205:5, 820:6, 850:7}
MMWHS_REMAP = {v: i for i, v in enumerate(MMWHS_LABEL_MAP.keys())}

# Color lookup for label_to_rgb (keyed by original label value)
LABEL_COLORS = {k: v[:3] for k, v in MMWHS_LABEL_MAP.items()}


def remap_mmwhs_labels(label_arr: np.ndarray) -> np.ndarray:
    """
    Remap MM-WHS 2017 original label values (0,205,420,500,550,600,820,850)
    to consecutive integers (0-7) required for model training.
    Unknown values are left as-is and a warning is printed.
    """
    out = np.zeros_like(label_arr, dtype=np.uint8)
    for orig, new in MMWHS_REMAP.items():
        out[label_arr == orig] = new
    unknown = {int(v) for v in np.unique(label_arr) if int(v) not in MMWHS_REMAP}
    if unknown:
        print(f"[WARN] Unknown label values found (kept as-is): {unknown}")
    return out


def label_to_rgb(label_slice: np.ndarray) -> np.ndarray:
    """Convert an integer label slice to an RGB colour image for overlay."""
    h, w = label_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for val, color in LABEL_COLORS.items():
        rgb[label_slice == val] = color
    # Unknown classes rendered in white
    known = set(LABEL_COLORS.keys())
    for val in np.unique(label_slice):
        if int(val) not in known:
            rgb[label_slice == val] = (1.0, 1.0, 1.0)
    return rgb


def save_sample_slices(
    img_arr: np.ndarray,
    lbl_arr: np.ndarray | None,
    out_dir: Path,
    n_slices: int = 5,
) -> None:
    """
    Sample n_slices axial slices at equal intervals and save panels:
      - If lbl_arr is provided: raw grayscale, colour-coded label, overlay
      - If lbl_arr is None: only raw grayscale
    """
    if not HAS_MPL:
        print("[WARN] Skipping slice export (matplotlib not available)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    depth = img_arr.shape[0]

    # Determine which slices to sample
    if lbl_arr is not None:
        # Only sample slices that actually contain labelled structures (non-background)
        labelled_slices = [i for i in range(depth) if np.any(lbl_arr[i] > 0)]
        if len(labelled_slices) == 0:
            print("[WARN] No labelled slices found, falling back to full range.")
            labelled_slices = list(range(depth))
        print(f"  {len(labelled_slices)}/{depth} slices contain labels "
              f"(slice {labelled_slices[0]} ~ {labelled_slices[-1]})")
        sample_pool = labelled_slices
    else:
        # No label, sample from middle 80% of volume to skip empty edge slices
        start = int(depth * 0.1)
        end = int(depth * 0.9)
        sample_pool = list(range(start, end))
        print(f"  Sampling from slices {start} ~ {end} ({len(sample_pool)} slices)")

    # Evenly sample n_slices from the pool
    pick = np.linspace(0, len(sample_pool) - 1, n_slices, dtype=int)
    indices = [sample_pool[p] for p in pick]

    # Percentile clip normalisation (avoids extreme values dominating window)
    p1, p99 = np.percentile(img_arr, 1), np.percentile(img_arr, 99)
    img_norm = np.clip((img_arr - p1) / (p99 - p1 + 1e-8), 0, 1)

    for idx in indices:
        img_slice = img_norm[idx]  # (H, W) float [0,1]

        if lbl_arr is not None:
            # With label: show 3 panels (grayscale, label, overlay)
            lbl_slice = lbl_arr[idx]  # (H, W) original MM-WHS values
            lbl_remap = remap_mmwhs_labels(lbl_slice)  # (H, W) 0-7
            lbl_rgb = label_to_rgb(lbl_slice)

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            axes[0].imshow(img_slice, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title(f"Grayscale  slice={idx}")
            axes[0].axis("off")

            axes[1].imshow(lbl_rgb)
            axes[1].set_title(f"Label (colour)  slice={idx}")
            axes[1].axis("off")

            axes[2].imshow(img_slice, cmap="gray", vmin=0, vmax=1)
            axes[2].imshow(lbl_rgb, alpha=0.4)
            axes[2].set_title(f"Overlay  slice={idx}")
            axes[2].axis("off")

            # Legend with anatomical names
            unique_vals = np.unique(lbl_slice)
            patches = []
            for v in unique_vals:
                color = LABEL_COLORS.get(int(v), (1, 1, 1))
                anat = MMWHS_LABEL_MAP.get(int(v), (0, 0, 0, f"unknown {int(v)}"))[3]
                patches.append(mpatches.Patch(color=color, label=anat))
            if patches:
                axes[1].legend(handles=patches, loc="lower right",
                               fontsize=7, framealpha=0.6)
        else:
            # Without label: show only grayscale image
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"MRI Slice {idx}/{depth-1}")
            ax.axis("off")

        plt.tight_layout()
        out_path = out_dir / f"slice_{idx:04d}.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nSlice images saved to: {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIfTI image & label reading test (MM-WHS 2017)")
    parser.add_argument("--image", required=True, help="Path to *_image.nii.gz")
    parser.add_argument("--label", default=None, help="Path to *_label.nii.gz (optional)")
    parser.add_argument("--out_dir", default="backend/tools/output_slices",
                        help="Output directory for slice images (default: backend/tools/output_slices)")
    parser.add_argument("--n_slices", type=int, default=5,
                        help="Number of representative slices to visualise (default: 5)")
    args = parser.parse_args()

    print("\n=== Loading volumes ===")
    img_sitk = load_nifti(args.image)
    print_info(img_sitk, "Image", is_label=False)

    lbl_sitk = None
    lbl_arr = None
    if args.label:
        lbl_sitk = load_nifti(args.label)
        print_info(lbl_sitk, "Label", is_label=True)

        # Shape consistency check
        img_arr = sitk.GetArrayFromImage(img_sitk)
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk)
        if img_arr.shape[:3] != lbl_arr.shape[:3]:
            print(f"\n[ERROR] Shape mismatch: image={img_arr.shape}  label={lbl_arr.shape}")
            sys.exit(1)
        else:
            print(f"\n[OK] Shapes match: {img_arr.shape[:3]}")

        # Show remapped labels
        lbl_remap_full = remap_mmwhs_labels(lbl_arr)
        print(f"\n[Remapped unique values (for training)]: {np.unique(lbl_remap_full)}")
    else:
        print("\n[INFO] No label file provided, processing image only.")
        img_arr = sitk.GetArrayFromImage(img_sitk)

    print("\n=== Saving representative axial slices ===")
    save_sample_slices(img_arr, lbl_arr, Path(args.out_dir), args.n_slices)

    print("\n=== Done ===")
    if args.label:
        print("\n─── MM-WHS 2017 usage notes ─────────────────────────────────")
        print(" Training : call remap_mmwhs_labels() to convert labels to 0-7")
        print(" Eval     : model outputs 0-7 -> reverse-lookup MMWHS_REMAP for original values")
        print(" Dataset  : 20 MR + 20 CT training cases, 7 cardiac structures")
        print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
