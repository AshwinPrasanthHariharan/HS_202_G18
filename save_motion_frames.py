#!/usr/bin/env python
"""
Simple script to save motion analysis visualization frames with density heatmap overlay.
Method 1: Save frames without display windows.
"""

import scipy.io
import numpy as np
from pathlib import Path

from src.motion_analysis import process


def _to_numeric_points(candidate) -> np.ndarray | None:
    """Return Nx2 numeric points from array-like data when possible."""
    if not isinstance(candidate, np.ndarray):
        return None
    if candidate.dtype.names is not None or candidate.dtype == object:
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
    return points[mask]


def _extract_points_recursive(obj) -> np.ndarray | None:
    """Walk nested MATLAB containers and return the largest Nx2 points array found."""
    best = _to_numeric_points(obj)

    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        children = []
        for idx in np.ndindex(obj.shape):
            rec = obj[idx]
            for field_name in obj.dtype.names:
                children.append(rec[field_name])
    elif isinstance(obj, np.void) and obj.dtype.names is not None:
        children = [obj[field_name] for field_name in obj.dtype.names]
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        children = [obj[idx] for idx in np.ndindex(obj.shape)]
    elif isinstance(obj, (list, tuple)):
        children = list(obj)
    else:
        children = []

    for child in children:
        child_points = _extract_points_recursive(child)
        if child_points is None:
            continue
        if best is None or child_points.shape[0] > best.shape[0]:
            best = child_points

    return best

print("Motion Analysis with Density Heatmap - Frame Saving")
print("=" * 60)

DEFAULT_DENSITY_SIGMA = 18.0

# Get user input
try:
    end_frame = int(input("How many frames to process? (default 50): ") or "50")
except ValueError:
    end_frame = 50

print(f"\nProcessing first {end_frame} frames...")
print("(This generates motion analysis visualizations as PNG images)")
print("-" * 60)

# Load mall dataset density data
print("Loading mall dataset density data...")
mall_gt_path = Path('mall_dataset/mall_gt.mat')
density_data = {}
if mall_gt_path.exists():
    try:
        mall_gt = scipy.io.loadmat(str(mall_gt_path))
        if 'frame' in mall_gt:
            frame_data = mall_gt['frame']
            if frame_data.size > 0:
                actual_frames = frame_data.ravel()
                print(f"actual_frames shape: {actual_frames.shape}, dtype: {actual_frames.dtype}")
                for i in range(min(len(actual_frames), end_frame)):
                    frame_struct = actual_frames[i]
                    loc = None

                    if isinstance(frame_struct, np.void) and frame_struct.dtype.names and 'loc' in frame_struct.dtype.names:
                        loc = frame_struct['loc']
                    elif hasattr(frame_struct, 'dtype') and frame_struct.dtype.names and 'loc' in frame_struct.dtype.names:
                        loc = frame_struct['loc']

                    points_arr = _extract_points_recursive(loc)
                    if points_arr is not None and points_arr.size > 0:
                        density_data[i] = [(float(x), float(y)) for x, y in points_arr]
                print(f"Loaded density data for {len(density_data)} frames")
            else:
                print("Frame data is empty")
        else:
            print("No 'frame' key in mall_gt.mat")
    except Exception as e:
        print(f"Error loading density data: {e}")
else:
    print("mall_gt.mat not found, proceeding without density overlay")

# Process and save frames
stats = process(
    'mall_dataset/frames',
    end_idx=end_frame,
    display=False,           # No windows
    save_frames=True,        # Save as PNG images
    start_idx=0,
    skip_frames=1,
    density_data=density_data if density_data else None,
    detect_density_from_frame=True,
    use_yolo_for_density=True,
    density_sigma=DEFAULT_DENSITY_SIGMA,
    adaptive_density_sigma=False,
)

# Show results
print("\n" + "=" * 60)
print("SAVED FRAMES LOCATION: output_frames/")
print("Each frame contains 4 panels: density heatmap | flow arrows")
print("                           zone overlay    | motion heatmap")
print("=" * 60)

# Count saved files
output_dir = Path('output_frames')
saved_files = list(output_dir.glob('frame_*.png'))
if saved_files:
    print(f"✓ {len(saved_files)} visualization frames saved")
    print(f"✓ First frame: {output_dir / saved_files[0].name}")
    print(f"✓ Last frame: {output_dir / saved_files[-1].name}")
    print("\nYou can now:")
    print("  1. Open output_frames/ folder in Windows Explorer")
    print("  2. View PNG images (double-click to open)")
    print("  3. Use Windows Photo Viewer to browse through them")
else:
    print("✗ No frames were saved")

print("\nStatistics:")
print(f"  Total pairs: {stats['total_pairs']}")
print(f"  Processed: {stats['processed']}")
print(f"  Mean speed: {stats.get('mean_avg_speed', 0):.2f} px/frame")
print(f"  Max speed: {stats.get('mean_max_speed', 0):.2f} px/frame")
