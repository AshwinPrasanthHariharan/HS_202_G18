import cv2
import numpy as np
from typing import Tuple


def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Compute optical flow between two grayscale frames using Farneback algorithm.
    
    Args:
        prev_gray: Previous grayscale frame
        curr_gray: Current grayscale frame
    
    Returns:
        Optical flow array of shape (H, W, 2) with flow vectors (fx, fy)
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    return flow
