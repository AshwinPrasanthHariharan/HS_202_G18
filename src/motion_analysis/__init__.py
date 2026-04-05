"""
Motion Analysis Module

This module provides crowd motion estimation and speed analysis using optical flow.

Key Components:
- FramePairLoader: Load frame pairs from video or image directory
- compute_optical_flow: Compute optical flow using Farneback algorithm
- flow_to_magnitude: Convert flow to per-pixel magnitude
- build_zone_speed_map: Partition image into zones and compute mean speed
- Visualization functions: draw_zone_overlay, draw_magnitude_heatmap, draw_flow_arrows
- process: Main pipeline for processing data
- run_synthetic_demo: Generate synthetic data for testing
"""

from .motion_utils import (
    FramePairLoader,
    flow_to_magnitude,
    draw_zone_overlay,
    draw_magnitude_heatmap,
    draw_flow_arrows,
    overlay_heatmap_on_frame,
)

from .optical_flow import compute_optical_flow

from .speed_estimator import (
    build_zone_speed_map,
    process,
    run_synthetic_demo,
    GRID_ROWS,
    GRID_COLS,
    SPEED_MIN,
    SPEED_MAX,
    SAVE_FRAMES,
    OUTPUT_DIR,
)

__all__ = [
    # Frame loading
    "FramePairLoader",
    # Optical flow
    "compute_optical_flow",
    # Magnitude and zones
    "flow_to_magnitude",
    "build_zone_speed_map",
    # Visualization
    "draw_zone_overlay",
    "draw_magnitude_heatmap",
    "draw_flow_arrows",
    # Processing pipelines
    "process",
    "run_synthetic_demo",
    # Configuration
    "GRID_ROWS",
    "GRID_COLS",
    "SPEED_MIN",
    "SPEED_MAX",
    "SAVE_FRAMES",
    "OUTPUT_DIR",
]

