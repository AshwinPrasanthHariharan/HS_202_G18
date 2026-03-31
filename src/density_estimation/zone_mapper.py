from __future__ import annotations

from typing import Iterable

import numpy as np


GridDict = dict[str, tuple[float, float, float, float]]


def assign_points_to_grid(
	points: Iterable[tuple[float, float]],
	grid: GridDict,
) -> dict[str, int]:
	"""
	Count how many points fall in each rectangular zone.

	Zone tuple format: (x1, y1, x2, y2)
	"""
	zone_names = list(grid.keys())
	if not zone_names:
		return {}

	zone_boxes = np.asarray([grid[name] for name in zone_names], dtype=float)
	counts = np.zeros(len(zone_names), dtype=int)

	points_arr = np.asarray(list(points), dtype=float)
	if points_arr.size == 0:
		return {name: 0 for name in zone_names}

	if points_arr.ndim == 1:
		points_arr = points_arr.reshape(-1, 2)

	px = points_arr[:, 0][None, :]
	py = points_arr[:, 1][None, :]

	x1 = zone_boxes[:, 0][:, None]
	y1 = zone_boxes[:, 1][:, None]
	x2 = zone_boxes[:, 2][:, None]
	y2 = zone_boxes[:, 3][:, None]

	inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
	counts = inside.sum(axis=1).astype(int)

	return {name: int(counts[i]) for i, name in enumerate(zone_names)}


def normalize_zone_density(
	zone_density: dict[str, int],
	grid: GridDict,
	mode: str = "area",
) -> dict[str, float]:
	"""
	Normalize zone density.

	Supported modes:
	- "area": count / zone_area
	- "total": count / total_count
	"""
	if mode not in {"area", "total"}:
		raise ValueError("mode must be one of: 'area', 'total'")

	if mode == "total":
		total = float(sum(zone_density.values()))
		if total == 0:
			return {k: 0.0 for k in zone_density}
		return {k: float(v) / total for k, v in zone_density.items()}

	normalized: dict[str, float] = {}
	for zone_name, count in zone_density.items():
		x1, y1, x2, y2 = grid[zone_name]
		area = max((x2 - x1) * (y2 - y1), 1.0)
		normalized[zone_name] = float(count) / float(area)
	return normalized

