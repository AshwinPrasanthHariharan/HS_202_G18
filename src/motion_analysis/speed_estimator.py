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
from typing import Tuple, Dict, Optional

from .optical_flow import compute_optical_flow
from .motion_utils import (
    FramePairLoader, flow_to_magnitude, draw_zone_overlay,
    draw_magnitude_heatmap, draw_flow_arrows, overlay_heatmap_on_frame
)


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
            skip_frames: int = 1) -> dict:
    """
    Optimized processing loop for frame sequences.
    
    Computes optical flow, magnitude, zone speeds, and generates visualization panels.
    Saves 2×2 combined panel (original, heatmap, flow arrows, zone overlay) to output directory.
    
    Args:
        source: Path to image directory
        display: If True, display results in a window (slower)
        save_frames: If True, save output frames (defaults to SAVE_FRAMES constant)
        output_dir: Output directory (defaults to OUTPUT_DIR constant)
        start_idx: Start processing from this frame index (0-based)
        end_idx: End processing at this frame index (inclusive)
        skip_frames: Process every nth frame (default 1 = all frames)
    
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
                
                # Create 2×2 panel: heatmap overlay | flow arrows
                #                   zone overlay      | original
                top_row = np.hstack([heatmap_overlaid, flow_viz])
                bottom_row = np.hstack([zone_viz, curr_frame])
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

