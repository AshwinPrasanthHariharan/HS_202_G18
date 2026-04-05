import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Iterator, Optional, List

# Try to import tqdm for progress bars, fall back if not available
try:
    from tqdm import tqdm as _tqdm_class
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    _tqdm_class = None

def _make_progress_iterator(iterable, total, desc, unit):
    """Wraps iterable with progress bar if tqdm available, otherwise returns as-is."""
    if _HAS_TQDM and _tqdm_class is not None:
        try:
            # Wrap tqdm with iter() to ensure it returns an iterator properly
            return iter(_tqdm_class(iterable, total=total, desc=desc, unit=unit))
        except Exception:
            # Fall back if tqdm fails for any reason
            return iter(iterable)
    return iter(iterable)


class FramePairLoader:
    """
    Optimized loader for consecutive frame pairs from image directory.
    Efficiently handles large frame sequences (tested with 2000+ frames).
    """
    
    def __init__(self, source_path: str, start_idx: int = 0, 
                 end_idx: Optional[int] = None, show_progress: bool = True):
        """
        Initialize the FramePairLoader for image directory.
        
        Args:
            source_path: Path to image directory
            start_idx: Start frame index (0-based)
            end_idx: End frame index (inclusive, None for all frames)
            show_progress: Display progress bar during iteration
        
        Raises:
            ValueError: If source doesn't exist or fewer than 2 images found
        """
        self.source_path = Path(source_path)
        self.show_progress = show_progress
        
        if not self.source_path.is_dir():
            raise ValueError(f"Source path must be a directory: {source_path}")
        
        # Find all image files with supported extensions
        supported_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.frames = sorted([
            f for f in self.source_path.iterdir()
            if f.suffix.lower() in supported_exts
        ])
        
        if len(self.frames) < 2:
            raise ValueError(f"Fewer than 2 images found in {source_path}")
        
        # Handle frame range
        self.total_frames = len(self.frames)
        self.start_idx = max(0, start_idx)
        self.end_idx = min(self.total_frames - 1, end_idx if end_idx is not None else self.total_frames - 1)
        
        if self.start_idx >= self.total_frames - 1:
            raise ValueError(f"Start index {start_idx} out of range (0-{self.total_frames - 1})")
        
        self.frame_range = self.end_idx - self.start_idx
    
    def __len__(self) -> int:
        """Return number of frame pairs available."""
        return self.frame_range
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over consecutive frame pairs.
        
        Yields:
            Tuple of (prev_frame, curr_frame) in BGR format
        """
        iterator = self._image_pairs()
        
        if self.show_progress:
            iterator = _make_progress_iterator(iterator, total=self.frame_range, 
                                              desc="Processing frames", unit="pair")
        
        return iterator
    
    def _image_pairs(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield frame pairs from image directory with range support."""
        for i in range(self.start_idx, self.end_idx):
            prev_frame = cv2.imread(str(self.frames[i]))
            curr_frame = cv2.imread(str(self.frames[i + 1]))
            
            if prev_frame is not None and curr_frame is not None:
                yield prev_frame, curr_frame
    
    def get_frame(self, idx: int) -> np.ndarray:
        """
        Load a single frame by index.
        
        Args:
            idx: Frame index (0-based)
        
        Returns:
            BGR frame as numpy array
        """
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(f"Frame index {idx} out of range (0-{self.total_frames - 1})")
        
        frame = cv2.imread(str(self.frames[idx]))
        if frame is None:
            raise IOError(f"Failed to load frame: {self.frames[idx]}")
        
        return frame
    
    def get_frame_path(self, idx: int) -> Path:
        """Get the file path of a frame by index."""
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(f"Frame index {idx} out of range (0-{self.total_frames - 1})")
        return self.frames[idx]


def flow_to_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to per-pixel magnitude.
    
    Args:
        flow: Optical flow array of shape (H, W, 2)
    
    Returns:
        Magnitude array of shape (H, W) as float32
    """
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.sqrt(fx**2 + fy**2).astype(np.float32)
    return magnitude


def draw_zone_overlay(frame: np.ndarray, zone_speed: np.ndarray, 
                      speed_min: float = 0.0, speed_max: float = 10.0) -> np.ndarray:
    """
    Draw semi-transparent colored rectangles per zone with numeric speed labels.
    
    Args:
        frame: BGR image
        zone_speed: Zone speed map of shape (rows, cols)
        speed_min: Minimum speed for color mapping
        speed_max: Maximum speed for color mapping
    
    Returns:
        Frame with zone overlay
    """
    rows, cols = zone_speed.shape
    frame_h, frame_w = frame.shape[:2]
    
    # Create overlay
    overlay = frame.copy()
    
    # Get row and column edge positions
    row_edges = np.linspace(0, frame_h, rows + 1, dtype=int)
    col_edges = np.linspace(0, frame_w, cols + 1, dtype=int)
    
    # Normalize speeds for color mapping
    normalized = np.clip((zone_speed - speed_min) / (speed_max - speed_min + 1e-6), 0, 1)
    
    for i in range(rows):
        for j in range(cols):
            y1, y2 = row_edges[i], row_edges[i + 1]
            x1, x2 = col_edges[j], col_edges[j + 1]
            
            # Color based on speed (red = high, blue = low)
            norm_speed = normalized[i, j]
            color = (int(255 * (1 - norm_speed)), 100, int(255 * norm_speed))  # BGR
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Add speed label
            text = f"{zone_speed[i, j]:.1f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - 0.35, overlay, 0.35, 0)
    return result


def draw_magnitude_heatmap(magnitude: np.ndarray, speed_max: float = 10.0) -> np.ndarray:
    """
    Generate false-color heatmap of magnitude using JET colormap.
    
    Args:
        magnitude: Magnitude array of shape (H, W)
        speed_max: Maximum speed for normalization
    
    Returns:
        False-color BGR image
    """
    # Normalize magnitude to 0-255 for colormap
    normalized = np.clip(magnitude / speed_max, 0, 1) * 255
    normalized = normalized.astype(np.uint8)
    
    # Apply JET colormap
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return heatmap


def overlay_heatmap_on_frame(frame: np.ndarray, magnitude: np.ndarray, 
                            speed_max: float = 10.0, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay heatmap directly on the original frame using alpha blending.
    
    Args:
        frame: Original BGR frame
        magnitude: Magnitude array of shape (H, W)
        speed_max: Maximum speed for normalization
        alpha: Blend factor (0.0-1.0, higher = more heatmap visible)
    
    Returns:
        Frame with heatmap overlaid
    """
    heatmap = draw_magnitude_heatmap(magnitude, speed_max)
    # Blend frame with heatmap
    result = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return result


def draw_flow_arrows(frame: np.ndarray, flow: np.ndarray, 
                     step: int = 16, min_magnitude: float = 0.5) -> np.ndarray:
    """
    Draw optical flow arrows at regular grid positions.
    
    Args:
        frame: BGR image
        flow: Optical flow array of shape (H, W, 2)
        step: Grid spacing for arrow sampling
        min_magnitude: Skip vectors with magnitude below this threshold
    
    Returns:
        Frame with flow arrows
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            magnitude = np.sqrt(fx**2 + fy**2)
            
            if magnitude < min_magnitude:
                continue
            
            # Scale arrows to fit within step size
            scale = min(step / 2, step / (magnitude + 1e-6))
            dx = int(fx * scale)
            dy = int(fy * scale)
            
            # Draw arrow
            pt1 = (x, y)
            pt2 = (x + dx, y + dy)
            
            cv2.arrowedLine(result, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
    
    return result

