from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .annotation_reader import get_head_points
from .yolo_detector import get_centers_from_boxes, run_yolo_detection
from .zone_mapper import assign_points_to_grid, normalize_zone_density


DEFAULT_SHANGHAITECH_ROOT = (
	r"C:\Users\pingm\Personal Folder\Development\College Projects\Datasets for HS202\Shanghai Tech"
)


def load_image(path: str | Path) -> np.ndarray:
	"""Load an image using OpenCV."""
	image = cv2.imread(str(path))
	if image is None:
		raise FileNotFoundError(f"Unable to read image: {path}")
	return image


def _estimate_adaptive_sigma(points: list[tuple[float, float]], min_sigma: float, max_sigma: float) -> float:
	"""Estimate a density sigma from nearest-neighbor spacing."""
	if len(points) < 2:
		return float(max(min_sigma, min(max_sigma, (min_sigma + max_sigma) / 2.0)))

	points_arr = np.asarray(points, dtype=np.float32)
	diffs = points_arr[:, None, :] - points_arr[None, :, :]
	distances = np.sqrt(np.sum(diffs * diffs, axis=2))
	# Ignore self-distance zeros on the diagonal.
	distances[distances == 0] = np.inf
	nearest = np.min(distances, axis=1)
	median_distance = float(np.median(nearest[np.isfinite(nearest)]))
	if not np.isfinite(median_distance):
		return float(max(min_sigma, min(max_sigma, (min_sigma + max_sigma) / 2.0)))

	sigma = median_distance / 2.0
	return float(max(min_sigma, min(max_sigma, sigma)))


def build_gaussian_density_map(
	points: list[tuple[float, float]],
	shape: tuple[int, int],
	sigma: float = 18.0,
	adaptive_sigma: bool = False,
	min_sigma: float = 10.0,
	max_sigma: float = 15.0,
) -> np.ndarray:
	"""Create a smooth crowd density map from head points.

	Each point contributes a Gaussian kernel whose truncated mass is renormalized
	so that the total sum of the final heatmap equals the number of people.
	"""
	h, w = shape
	density_map = np.zeros((h, w), dtype=np.float32)
	if not points:
		return density_map

	if adaptive_sigma:
		sigma = _estimate_adaptive_sigma(points, min_sigma=min_sigma, max_sigma=max_sigma)
	else:
		sigma = float(sigma)

	if sigma <= 0:
		raise ValueError("sigma must be greater than zero")

	radius = max(1, int(np.ceil(3.0 * sigma)))

	for x, y in points:
		if not np.isfinite(x) or not np.isfinite(y):
			continue

		x1 = max(0, int(np.floor(x)) - radius)
		x2 = min(w, int(np.floor(x)) + radius + 1)
		y1 = max(0, int(np.floor(y)) - radius)
		y2 = min(h, int(np.floor(y)) + radius + 1)

		if x1 >= x2 or y1 >= y2:
			continue

		x_coords = np.arange(x1, x2, dtype=np.float32) - float(x)
		y_coords = np.arange(y1, y2, dtype=np.float32) - float(y)
		kernel = np.exp(-(y_coords[:, None] ** 2 + x_coords[None, :] ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
		kernel_sum = float(kernel.sum())
		if kernel_sum <= 0:
			continue

		density_map[y1:y2, x1:x2] += kernel / kernel_sum

	return density_map


def compute_density_map(
	points: list[tuple[float, float]],
	grid: dict[str, tuple[float, float, float, float]],
	frame_id: str,
	normalize: bool = False,
	normalization_mode: str = "area",
) -> dict[str, Any]:
	"""Compute zone-wise density counts for a frame."""
	zone_density = assign_points_to_grid(points, grid)

	result: dict[str, Any] = {
		"frame_id": frame_id,
		"zone_density": zone_density,
	}

	if normalize:
		result["zone_density_normalized"] = normalize_zone_density(
			zone_density, grid, mode=normalization_mode
		)

	return result


def draw_density_debug(
	image: np.ndarray,
	grid: dict[str, tuple[float, float, float, float]],
	points: list[tuple[float, float]],
	zone_density: dict[str, int] | None = None,
	show_zone_counts: bool = True,
) -> np.ndarray:
	"""Draw grid zones, points, and optional zone counts for debugging."""
	out = image.copy()

	for zone_name, (x1, y1, x2, y2) in grid.items():
		p1 = (int(round(x1)), int(round(y1)))
		p2 = (int(round(x2)), int(round(y2)))
		cv2.rectangle(out, p1, p2, (0, 255, 255), 1)

		if show_zone_counts and zone_density is not None:
			label = f"{zone_name}: {zone_density.get(zone_name, 0)}"
			text_pt = (p1[0] + 4, max(15, p1[1] + 15))
			cv2.putText(
				out,
				label,
				text_pt,
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				(0, 255, 0),
				1,
				cv2.LINE_AA,
			)

	for x, y in points:
		cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

	return out


def _candidate_annotation_names(frame_id: str) -> list[str]:
	return [f"GT_{frame_id}.mat", f"{frame_id}.mat"]


def _find_annotation_path(frame_id: str, annotations_dir: Path) -> Path:
	for name in _candidate_annotation_names(frame_id):
		candidate = annotations_dir / name
		if candidate.exists():
			return candidate
	raise FileNotFoundError(
		f"Annotation file not found for frame '{frame_id}' in {annotations_dir}"
	)


def process_frame_density(
	image_path: str | Path,
	grid: dict[str, tuple[float, float, float, float]],
	annotation_path: str | Path | None = None,
	use_yolo: bool = False,
	yolo_kwargs: dict[str, Any] | None = None,
	normalize: bool = False,
	normalization_mode: str = "area",
) -> dict[str, Any]:
	"""Process one frame and return zone density output payload."""
	image_path = Path(image_path)
	frame_id = image_path.stem

	if use_yolo:
		image = load_image(image_path)
		boxes = run_yolo_detection(image, **(yolo_kwargs or {}))
		points = get_centers_from_boxes(boxes)
	else:
		if annotation_path is None:
			raise ValueError("annotation_path is required when use_yolo=False")
		points = get_head_points(annotation_path)

	return compute_density_map(
		points=points,
		grid=grid,
		frame_id=frame_id,
		normalize=normalize,
		normalization_mode=normalization_mode,
	)


def process_image_directory(
	images_dir: str | Path,
	grid: dict[str, tuple[float, float, float, float]],
	annotations_dir: str | Path | None = None,
	use_yolo: bool = False,
	yolo_kwargs: dict[str, Any] | None = None,
	normalize: bool = False,
	normalization_mode: str = "area",
	image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> dict[str, dict[str, int] | dict[str, float]]:
	"""
	Batch-process all images in a directory.

	Returns:
		{
		  "IMG_1": {"z0": 4, "z1": 9, ...},
		  ...
		}
	"""
	images_dir = Path(images_dir)
	if not images_dir.exists():
		raise FileNotFoundError(f"Images directory not found: {images_dir}")

	if not use_yolo:
		if annotations_dir is None:
			raise ValueError("annotations_dir is required when use_yolo=False")
		annotations_dir = Path(annotations_dir)
		if not annotations_dir.exists():
			raise FileNotFoundError(
				f"Annotations directory not found: {annotations_dir}"
			)

	image_paths = sorted(
		p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions
	)

	output: dict[str, dict[str, int] | dict[str, float]] = {}
	for image_path in image_paths:
		frame_id = image_path.stem
		ann_path = None
		if not use_yolo:
			ann_path = _find_annotation_path(frame_id, annotations_dir)  # type: ignore[arg-type]

		frame_result = process_frame_density(
			image_path=image_path,
			annotation_path=ann_path,
			grid=grid,
			use_yolo=use_yolo,
			yolo_kwargs=yolo_kwargs,
			normalize=normalize,
			normalization_mode=normalization_mode,
		)

		output[frame_id] = frame_result["zone_density"]

	return output


class DensityCalculator:
	"""Convenience wrapper for density estimation operations."""

	def __init__(
		self,
		grid: dict[str, tuple[float, float, float, float]],
		normalize: bool = False,
		normalization_mode: str = "area",
	) -> None:
		self.grid = grid
		self.normalize = normalize
		self.normalization_mode = normalization_mode

	def process_frame(
		self,
		image_path: str | Path,
		annotation_path: str | Path | None = None,
		use_yolo: bool = False,
		yolo_kwargs: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		return process_frame_density(
			image_path=image_path,
			annotation_path=annotation_path,
			grid=self.grid,
			use_yolo=use_yolo,
			yolo_kwargs=yolo_kwargs,
			normalize=self.normalize,
			normalization_mode=self.normalization_mode,
		)

	def process_directory(
		self,
		images_dir: str | Path,
		annotations_dir: str | Path | None = None,
		use_yolo: bool = False,
		yolo_kwargs: dict[str, Any] | None = None,
	) -> dict[str, dict[str, int] | dict[str, float]]:
		return process_image_directory(
			images_dir=images_dir,
			grid=self.grid,
			annotations_dir=annotations_dir,
			use_yolo=use_yolo,
			yolo_kwargs=yolo_kwargs,
			normalize=self.normalize,
			normalization_mode=self.normalization_mode,
		)


def get_shanghaitech_split_paths(
	dataset_root: str | Path = DEFAULT_SHANGHAITECH_ROOT,
	part: str = "A",
	split: str = "test",
) -> tuple[Path, Path]:
	"""Return (images_dir, ground_truth_dir) for ShanghaiTech Part A/B split."""
	part = part.upper()
	if part not in {"A", "B"}:
		raise ValueError("part must be 'A' or 'B'")

	split = split.lower()
	if split not in {"train", "test"}:
		raise ValueError("split must be 'train' or 'test'")

	part_dir = f"part_{part}_final"
	split_dir = f"{split}_data"

	base = Path(dataset_root) / part_dir / split_dir
	images_dir = base / "images"
	ground_truth_dir = base / "ground_truth"
	return images_dir, ground_truth_dir

