#!/usr/bin/env python
"""
Generate crowd density heatmap images from ShanghaiTech annotations.

- Scans Part A and Part B splits (train/test) if present.
- Loads matching ground-truth .mat files for each image.
- Builds smooth Gaussian density maps from head points.
- Saves heatmap-over-image outputs to outputs/density_outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.io import loadmat

from src.density_estimation import build_gaussian_density_map
from src.density_estimation.annotation_reader import get_head_points

# Optional progress bar.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\pingm\Personal Folder\Development\College Projects\HS202_G18\datasets\Shanghai Tech"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\pingm\Personal Folder\Development\College Projects\HS202_G18\outputs\density_outputs"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_shanghaitech_splits(dataset_root: Path) -> list[tuple[str, str, Path, Path]]:
    """Return available (part, split, images_dir, gt_dir) entries."""
    entries: list[tuple[str, str, Path, Path]] = []

    for part in ("A", "B"):
        part_dir = dataset_root / f"part_{part}_final"
        for split in ("train", "test"):
            split_dir = part_dir / f"{split}_data"
            images_dir = split_dir / "images"
            gt_dir = split_dir / "ground_truth"
            if images_dir.is_dir() and gt_dir.is_dir():
                entries.append((part, split, images_dir, gt_dir))

    return entries


def list_images(images_dir: Path) -> list[Path]:
    """List images in deterministic order."""
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def find_annotation_path(image_path: Path, gt_dir: Path) -> Path | None:
    """Resolve ShanghaiTech-style GT annotation filename for an image."""
    frame_id = image_path.stem
    candidates = [
        gt_dir / f"GT_{frame_id}.mat",
        gt_dir / f"{frame_id}.mat",
    ]
    for ann_path in candidates:
        if ann_path.exists():
            return ann_path
    return None


def load_head_coordinates(annotation_path: Path) -> list[tuple[float, float]]:
    """Load head points from a .mat annotation file."""
    mat_data = loadmat(str(annotation_path))
    return get_head_points(mat_data)


def density_map_to_heatmap_image(density_map: np.ndarray) -> np.ndarray:
    """Convert a density map to a false-color BGR heatmap image."""
    if density_map.size == 0:
        raise ValueError("density_map is empty")

    peak = float(np.max(density_map))
    if peak > 0:
        normalized = np.clip(density_map / peak * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros(density_map.shape, dtype=np.uint8)

    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend color heatmap with the original image."""
    if image.shape[:2] != heatmap.shape[:2]:
        raise ValueError("image and heatmap dimensions must match")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)


def build_output_path(output_dir: Path, image_path: Path, part: str, split: str) -> Path:
    """Build output path that keeps part/split folders and original frame stem."""
    split_dir = output_dir / f"part_{part}_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f"{image_path.stem}_density.png"


def iter_with_progress(items: Iterable[Path], desc: str):
    """Yield iterable optionally wrapped in tqdm."""
    if tqdm is None:
        return items
    items_list = list(items)
    return tqdm(items_list, total=len(items_list), desc=desc, unit="img")


def process_split(
    part: str,
    split: str,
    images_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    sigma: float,
    adaptive_sigma: bool,
    min_sigma: float,
    max_sigma: float,
    overlay_alpha: float,
) -> tuple[int, int, int]:
    """Process one split and return (processed, missing_annotations, failures)."""
    processed = 0
    missing_annotations = 0
    failures = 0

    images = list_images(images_dir)
    desc = f"Part {part} {split}"

    for image_path in iter_with_progress(images, desc=desc):
        ann_path = find_annotation_path(image_path, gt_dir)
        if ann_path is None:
            missing_annotations += 1
            continue

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                failures += 1
                print(f"[WARN] Cannot read image: {image_path}")
                continue

            points = load_head_coordinates(ann_path)

            density_map = build_gaussian_density_map(
                points=points,
                shape=image.shape[:2],
                sigma=sigma,
                adaptive_sigma=adaptive_sigma,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
            )

            # build_gaussian_density_map already preserves total mass ~= number of points.
            heatmap_img = density_map_to_heatmap_image(density_map)
            output_img = overlay_heatmap_on_image(image, heatmap_img, alpha=overlay_alpha)

            out_path = build_output_path(output_dir, image_path, part, split)
            ok = cv2.imwrite(str(out_path), output_img)
            if not ok:
                failures += 1
                print(f"[WARN] Failed to write output: {out_path}")
                continue

            processed += 1

        except Exception as exc:
            failures += 1
            print(f"[WARN] Failed {image_path.name}: {exc}")

    return processed, missing_annotations, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate smooth ShanghaiTech density heatmap frames from .mat head annotations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="ShanghaiTech root containing part_A_final and/or part_B_final.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where heatmap images are saved.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=12.0,
        help="Gaussian sigma for density spread.",
    )
    parser.add_argument(
        "--adaptive-sigma",
        action="store_true",
        help="Use nearest-neighbor-based adaptive sigma.",
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=10.0,
        help="Minimum sigma when adaptive mode is enabled.",
    )
    parser.add_argument(
        "--max-sigma",
        type=float,
        default=15.0,
        help="Maximum sigma when adaptive mode is enabled.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Blend factor for heatmap overlay on original image (0.0-1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    output_dir = args.output_dir

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = find_shanghaitech_splits(dataset_root)
    if not splits:
        raise FileNotFoundError(
            f"No valid ShanghaiTech splits found under: {dataset_root}"
        )

    print("Generating ShanghaiTech density heatmaps")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir  : {output_dir}")
    print(f"Sigma       : {args.sigma:.2f}")
    print(f"Adaptive    : {args.adaptive_sigma}")
    print(f"Overlay a   : {args.overlay_alpha:.2f}")
    if args.adaptive_sigma:
        print(f"Sigma range : [{args.min_sigma:.2f}, {args.max_sigma:.2f}]")
    print("-" * 60)

    total_processed = 0
    total_missing = 0
    total_failures = 0

    for part, split, images_dir, gt_dir in splits:
        processed, missing, failures = process_split(
            part=part,
            split=split,
            images_dir=images_dir,
            gt_dir=gt_dir,
            output_dir=output_dir,
            sigma=args.sigma,
            adaptive_sigma=args.adaptive_sigma,
            min_sigma=args.min_sigma,
            max_sigma=args.max_sigma,
            overlay_alpha=args.overlay_alpha,
        )
        total_processed += processed
        total_missing += missing
        total_failures += failures

    print("\n" + "=" * 60)
    print("Density Heatmap Generation Complete")
    print("=" * 60)
    print(f"Processed images    : {total_processed}")
    print(f"Missing annotations : {total_missing}")
    print(f"Failures            : {total_failures}")
    print("=" * 60)


if __name__ == "__main__":
    main()
