from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def load_mat_annotations(path: str | Path) -> dict[str, Any]:
	"""Load a ShanghaiTech-style .mat annotation file."""
	return loadmat(str(path))


def _to_points_array(candidate: Any) -> np.ndarray | None:
	"""Convert a candidate object to an Nx2 numeric points array if possible."""
	if not isinstance(candidate, np.ndarray):
		return None
	if candidate.dtype.names is not None:
		# Structured arrays (common in MATLAB .mat) are handled elsewhere.
		return None
	if candidate.dtype == object:
		return None

	arr = np.asarray(candidate, dtype=float)
	if arr.ndim != 2:
		return None

	if arr.shape[1] == 2:
		points = arr
	elif arr.shape[0] == 2:
		points = arr.T
	else:
		return None

	if points.size == 0:
		return np.empty((0, 2), dtype=float)

	mask = np.isfinite(points).all(axis=1)
	points = points[mask]
	return points


def _extract_struct_children(obj: Any) -> list[Any]:
	"""Extract nested values from numpy structured arrays/records."""
	children: list[Any] = []

	if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
		for idx in np.ndindex(obj.shape):
			rec = obj[idx]
			for field_name in obj.dtype.names:
				children.append(rec[field_name])
	elif isinstance(obj, np.void) and obj.dtype.names is not None:
		for field_name in obj.dtype.names:
			children.append(obj[field_name])

	return children


def _find_numeric_points_recursive(obj: Any) -> np.ndarray | None:
	"""Walk nested MATLAB structures and return the best Nx2 points array found."""
	best: np.ndarray | None = None

	direct = _to_points_array(obj)
	if direct is not None:
		best = direct

	if isinstance(obj, dict):
		candidates = list(obj.values())
	elif isinstance(obj, (np.ndarray, np.void)) and getattr(obj.dtype, "names", None):
		candidates = _extract_struct_children(obj)
	elif isinstance(obj, np.ndarray) and obj.dtype == object:
		candidates = [obj[idx] for idx in np.ndindex(obj.shape)]
	elif isinstance(obj, (list, tuple)):
		candidates = list(obj)
	else:
		candidates = []

	for child in candidates:
		child_points = _find_numeric_points_recursive(child)
		if child_points is None:
			continue
		if best is None or child_points.shape[0] > best.shape[0]:
			best = child_points

	return best


def get_head_points(mat_file: str | Path | dict[str, Any]) -> list[tuple[float, float]]:
	"""
	Extract head annotation points from a ShanghaiTech .mat file.

	Returns a list of (x, y) tuples.
	"""
	if isinstance(mat_file, (str, Path)):
		mat_data = load_mat_annotations(mat_file)
	else:
		mat_data = mat_file

	if "image_info" in mat_data:
		points = _find_numeric_points_recursive(mat_data["image_info"])
	else:
		points = _find_numeric_points_recursive(mat_data)

	if points is None:
		raise ValueError("Could not find head annotation points in .mat content")

	return [(float(x), float(y)) for x, y in points]

