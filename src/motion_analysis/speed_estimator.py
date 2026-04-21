# Configuration Constants
GRID_ROWS = 4
GRID_COLS = 4
SPEED_MIN = 0.0
SPEED_MAX = 10.0
SAVE_FRAMES = True
OUTPUT_DIR = "output_frames"

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .optical_flow import compute_optical_flow
from .motion_utils import (
    FramePairLoader, flow_to_magnitude, draw_zone_overlay,
    draw_flow_arrows, overlay_heatmap_on_frame
)

# Import density detector helpers
try:
    from ..density_estimation import (
        run_yolo_detection,
        get_centers_from_boxes,
        build_gaussian_density_map,
    )
except ImportError:
    run_yolo_detection = None
    get_centers_from_boxes = None
    build_gaussian_density_map = None


def _coerce_points(points: Optional[list], frame_shape: tuple[int, int, int]) -> list[tuple[float, float]]:
    """Normalize arbitrary point payloads into valid (x, y) points within frame bounds."""
    if not points:
        return []

    h, w = frame_shape[:2]
    clean_points: list[tuple[float, float]] = []
    for item in points:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue

        if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h:
            clean_points.append((x, y))

    return clean_points


def _detect_density_points_from_frame(curr_frame: np.ndarray, use_yolo_for_density: bool = True) -> list[tuple[float, float]]:
    """Detect crowd points directly from frame using YOLO when available, with CV fallback."""
    if use_yolo_for_density and run_yolo_detection and get_centers_from_boxes:
        try:
            boxes = run_yolo_detection(curr_frame, classes=(0,), conf=0.20, iou=0.45)
            return get_centers_from_boxes(boxes)
        except Exception:
            # Fall through to OpenCV fallback detector.
            pass

    gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    blur_small = cv2.GaussianBlur(gray, (5, 5), 0)
    blur_large = cv2.GaussianBlur(gray, (0, 0), 11)
    fg = cv2.absdiff(blur_small, blur_large)
    _, mask = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    points: list[tuple[float, float]] = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if 20 <= area <= 1500:
            cx, cy = centroids[label]
            points.append((float(cx), float(cy)))

    return points


def _render_density_heatmap_panel(
    curr_frame: np.ndarray,
    points: list[tuple[float, float]],
    alpha: float = 0.55,
    density_sigma: float = 18.0,
    adaptive_density_sigma: bool = False,
) -> np.ndarray:
    """Render a smooth point-density heatmap over the frame for panel display."""
    h, w = curr_frame.shape[:2]

    if build_gaussian_density_map is not None:
        density = build_gaussian_density_map(
            points,
            (h, w),
            sigma=density_sigma,
            adaptive_sigma=adaptive_density_sigma,
        )
    else:
        density = np.zeros((h, w), dtype=np.float32)
        for x, y in points:
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                density[iy, ix] += 1.0

    peak = float(np.max(density))
    if peak > 0:
        normalized = np.clip(density / peak * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros((h, w), dtype=np.uint8)

    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    density_heatmap = cv2.applyColorMap(normalized, cmap)
    panel = cv2.addWeighted(curr_frame, 1.0 - alpha, density_heatmap, alpha, 0)

    cv2.putText(
        panel,
        f"Density points: {len(points)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def build_zone_speed_map(magnitude: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Partition magnitude image into (rows × cols) zones and compute mean speed per zone.
    
    Args:
        magnitude: Magnitude array of shape (H, W)
        rows: Number of row divisions
        cols: Number of column divisions
    
    Returns:
        Zone speed map of shape (rows, cols) as float32
    """
    h, w = magnitude.shape
    
    # Use linspace to get zone edges for compatibility with grid system
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)
    
    zone_speed = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            y1, y2 = row_edges[i], row_edges[i + 1]
            x1, x2 = col_edges[j], col_edges[j + 1]
            
            zone_magnitude = magnitude[y1:y2, x1:x2]
            zone_speed[i, j] = np.mean(zone_magnitude)
    
    return zone_speed


def process(source: str, display: bool = True, save_frames: bool = None,
            output_dir: str = None, start_idx: int = 0, end_idx: int = None,
            skip_frames: int = 1, density_data: Optional[Dict[int, list]] = None,
            detect_density_from_frame: bool = False,
            use_yolo_for_density: bool = True,
            density_sigma: float = 18.0,
            adaptive_density_sigma: bool = False) -> dict:
    """
    Optimized processing loop for frame sequences.
    
    Computes optical flow, magnitude, zone speeds, and generates visualization panels.
    Saves 2×2 combined panel (original, heatmap, flow arrows, zone overlay) to output directory.
    If density_data provided, includes density heatmap in the visualization.
    
    Args:
        source: Path to image directory
        display: If True, display results in a window (slower)
        save_frames: If True, save output frames (defaults to SAVE_FRAMES constant)
        output_dir: Output directory (defaults to OUTPUT_DIR constant)
        start_idx: Start processing from this frame index (0-based)
        end_idx: End processing at this frame index (inclusive)
        skip_frames: Process every nth frame (default 1 = all frames)
        density_data: Optional dict mapping frame index to list of (x,y) points for density overlay
        detect_density_from_frame: If True, estimate density points directly from each frame
        use_yolo_for_density: If True, prefer YOLO person detection when frame-based density is enabled
        density_sigma: Gaussian sigma used to smooth point annotations
        adaptive_density_sigma: If True, estimate sigma from nearest-neighbor spacing
    
    Returns:
        Dictionary with processing statistics
    """
    save_frames = save_frames if save_frames is not None else SAVE_FRAMES
    output_dir = output_dir if output_dir is not None else OUTPUT_DIR
    
    # Create output directory
    output_path = Path(output_dir)
    if save_frames:
        output_path.mkdir(exist_ok=True)
    
    # Initialize loader
    loader = FramePairLoader(source, start_idx=start_idx, end_idx=end_idx, 
                           show_progress=not display)
    
    frame_count = 0
    stats = {
        'total_pairs': len(loader),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'avg_speed': [],
        'max_speed': []
    }
    
    for i, (prev_frame, curr_frame) in enumerate(loader):
        # Skip frames if requested
        if i % skip_frames != 0:
            stats['skipped'] += 1
            continue
        
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow
            flow = compute_optical_flow(prev_gray, curr_gray)
            
            # Compute magnitude
            magnitude = flow_to_magnitude(flow)
            
            # Build zone speed map
            zone_speed = build_zone_speed_map(magnitude, GRID_ROWS, GRID_COLS)
            
            # Collect statistics
            stats['avg_speed'].append(float(np.mean(magnitude)))
            stats['max_speed'].append(float(np.max(magnitude)))
            
            # Generate visualizations (skip if not saving and not displaying)
            if save_frames or display:
                # Create visualizations with overlaid heatmap
                heatmap_overlaid = overlay_heatmap_on_frame(curr_frame.copy(), magnitude, SPEED_MAX, alpha=0.4)
                flow_viz = draw_flow_arrows(curr_frame.copy(), flow)
                zone_viz = draw_zone_overlay(curr_frame.copy(), zone_speed, SPEED_MIN, SPEED_MAX)

                points_from_annotations = []
                if density_data and frame_count in density_data:
                    points_from_annotations = _coerce_points(density_data[frame_count], curr_frame.shape)

                points = points_from_annotations
                if not points and detect_density_from_frame:
                    points = _detect_density_points_from_frame(
                        curr_frame,
                        use_yolo_for_density=use_yolo_for_density,
                    )

                density_overlay = _render_density_heatmap_panel(
                    curr_frame.copy(),
                    points,
                    density_sigma=density_sigma,
                    adaptive_density_sigma=adaptive_density_sigma,
                )
                
                # Create 2×2 panel: density overlay | flow arrows
                #                   zone overlay     | motion heatmap
                top_row = np.hstack([density_overlay, flow_viz])
                bottom_row = np.hstack([zone_viz, heatmap_overlaid])
                panel = np.vstack([top_row, bottom_row])
                
                # if requested
                if display:
                    cv2.namedWindow("Motion Analysis Pipeline")
                    cv2.imshow("Motion Analysis Pipeline", panel)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                
                # Save if enabled
                if save_frames:
                    output_file = output_path / f"frame_{frame_count:06d}.png"
                    cv2.imwrite(str(output_file), panel)
            
            frame_count += 1
            stats['processed'] += 1
        
        except Exception as e:
            stats['errors'] += 1
            print(f"Error processing frame {i}: {e}")
            continue
    
    if display:
        cv2.destroyAllWindows()
    
    # Compute final statistics
    if stats['avg_speed']:
        stats['mean_avg_speed'] = float(np.mean(stats['avg_speed']))
        stats['mean_max_speed'] = float(np.mean(stats['max_speed']))
    
    print(f"\n{'='*50}")
    print(f"Processing Complete")
    print(f"{'='*50}")
    print(f"Total pairs available: {stats['total_pairs']}")
    print(f"Processed: {stats['processed']} pairs")
    print(f"Skipped: {stats['skipped']} pairs")
    print(f"Errors: {stats['errors']}")
    if stats['mean_avg_speed']:
        print(f"Mean average speed: {stats['mean_avg_speed']:.2f} px/frame")
        print(f"Mean max speed: {stats['mean_max_speed']:.2f} px/frame")
    print(f"{'='*50}\n")
    
    return stats


def run_synthetic_demo(n_frames: int = 30) -> None:
    """
    Generate synthetic moving Gaussian blobs for pipeline validation.
    
    Creates n_frames with n=12 blobs of radius 12-30px, velocity ±3px/frame.
    Useful for testing without a real dataset.
    
    Args:
        n_frames: Number of frames to generate
    """
    # Frame dimensions
    h, w = 480, 640
    
    # Initialize synthetic video
    frame_h, frame_w = h, w
    n_blobs = 12
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    if SAVE_FRAMES:
        output_path.mkdir(exist_ok=True)
    
    # Initialize blob properties: (x, y, radius, vx, vy)
    np.random.seed(42)
    blobs = []
    for _ in range(n_blobs):
        x = np.random.uniform(50, frame_w - 50)
        y = np.random.uniform(50, frame_h - 50)
        radius = np.random.uniform(12, 30)
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-3, 3)
        blobs.append([x, y, radius, vx, vy])
    
    blobs = np.array(blobs, dtype=np.float32)
    
    prev_frame = None
    
    for frame_idx in range(n_frames):
        # Create frame
        frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 50  # Dark background
        
        # Draw blobs
        for i in range(n_blobs):
            x, y, radius, vx, vy = blobs[i]
            cx, cy = int(x), int(y)
            r = int(radius)
            
            # Draw Gaussian blob
            cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1)
            
            # Update position
            blobs[i, 0] += vx
            blobs[i, 1] += vy
            
            # Bounce off walls
            if blobs[i, 0] < blobs[i, 2]:
                blobs[i, 0] = blobs[i, 2]
                blobs[i, 3] = -blobs[i, 3]
            elif blobs[i, 0] > frame_w - blobs[i, 2]:
                blobs[i, 0] = frame_w - blobs[i, 2]
                blobs[i, 3] = -blobs[i, 3]
            
            if blobs[i, 1] < blobs[i, 2]:
                blobs[i, 1] = blobs[i, 2]
                blobs[i, 4] = -blobs[i, 4]
            elif blobs[i, 1] > frame_h - blobs[i, 2]:
                blobs[i, 1] = frame_h - blobs[i, 2]
                blobs[i, 4] = -blobs[i, 4]
        
        # Process with motion analysis
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow
            flow = compute_optical_flow(prev_gray, curr_gray)
            
            # Compute magnitude
            magnitude = flow_to_magnitude(flow)
            
            # Build zone speed map
            zone_speed = build_zone_speed_map(magnitude, GRID_ROWS, GRID_COLS)
            
            # Generate visualizations with overlaid heatmap
            heatmap_overlaid = overlay_heatmap_on_frame(frame.copy(), magnitude, SPEED_MAX, alpha=0.4)
            flow_viz = draw_flow_arrows(frame.copy(), flow)
            zone_viz = draw_zone_overlay(frame.copy(), zone_speed, SPEED_MIN, SPEED_MAX)
            
            # Create 2×2 panel: heatmap overlay | flow arrows
            #                   zone overlay      | original
            top_row = np.hstack([heatmap_overlaid, flow_viz])
            bottom_row = np.hstack([zone_viz, frame])
            panel = np.vstack([top_row, bottom_row])
            
            # Display
            cv2.namedWindow("Synthetic Motion Analysis")
            cv2.imshow("Synthetic Motion Analysis", panel)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            
            # Save if enabled
            if SAVE_FRAMES:
                output_file = output_path / f"synthetic_{frame_idx:05d}.png"
                cv2.imwrite(str(output_file), panel)
                print(f"Saved: {output_file}")
        
        prev_frame = frame
    
    cv2.destroyAllWindows()
    print(f"Synthetic demo complete. {n_frames} frames generated.")


if __name__ == "__main__":
    # Example usage
    print("Motion Analysis Module")
    print("Run synthetic demo: run_synthetic_demo()")
    print("Process real data: process('path/to/data')")

