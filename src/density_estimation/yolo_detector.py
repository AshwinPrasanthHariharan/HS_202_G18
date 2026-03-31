from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _ensure_image_array(image: str | Path | np.ndarray) -> np.ndarray:
	if isinstance(image, np.ndarray):
		return image
	image_arr = cv2.imread(str(image))
	if image_arr is None:
		raise FileNotFoundError(f"Unable to read image: {image}")
	return image_arr


def run_yolo_detection(
	image: str | Path | np.ndarray,
	model_path: str = "yolov8n.pt",
	conf: float = 0.25,
	iou: float = 0.45,
	classes: tuple[int, ...] | None = (0,),
	device: str | None = None,
	model: Any | None = None,
) -> np.ndarray:
	"""
	Run YOLOv8 person detection and return boxes in xyxy format.

	Returns:
		np.ndarray with shape (N, 4), columns: x1, y1, x2, y2
	"""
	image_arr = _ensure_image_array(image)

	if model is None:
		try:
			from ultralytics import YOLO
		except ImportError as exc:
			raise ImportError(
				"ultralytics is required for YOLO detection. "
				"Install with `pip install ultralytics`."
			) from exc
		model = YOLO(model_path)

	results = model.predict(
		source=image_arr,
		conf=conf,
		iou=iou,
		classes=list(classes) if classes is not None else None,
		device=device,
		verbose=False,
	)

	if not results or results[0].boxes is None:
		return np.empty((0, 4), dtype=float)

	boxes = results[0].boxes.xyxy
	if boxes is None or len(boxes) == 0:
		return np.empty((0, 4), dtype=float)

	return boxes.detach().cpu().numpy().astype(float)


def get_centers_from_boxes(boxes: np.ndarray) -> list[tuple[float, float]]:
	"""Convert xyxy bounding boxes to center points (cx, cy)."""
	if boxes.size == 0:
		return []

	boxes = np.asarray(boxes, dtype=float)
	cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
	cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
	centers = np.stack((cx, cy), axis=1)
	return [(float(x), float(y)) for x, y in centers]

