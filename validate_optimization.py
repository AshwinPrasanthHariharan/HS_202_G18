"""
Quick validation script for optimized motion analysis module.
Verifies module structure and dataset compatibility.
"""

import sys
from pathlib import Path

print("="*60)
print("Motion Analysis Module - Validation Script")
print("="*60)

# Check 1: Module imports
print("\n[1] Checking module imports...")
try:
    from src.motion_analysis import (
        FramePairLoader, 
        compute_optical_flow,
        flow_to_magnitude,
        build_zone_speed_map,
        process,
        GRID_ROWS, GRID_COLS
    )
    print("    ✓ All core modules imported successfully")
except ImportError as e:
    print(f"    ✗ Import failed: {e}")
    sys.exit(1)

# Check 2: Dataset validation
print("\n[2] Checking mall dataset...")
dataset_path = Path("mall_dataset/frames")
if not dataset_path.exists():
    print(f"    ✗ Dataset not found at {dataset_path}")
    sys.exit(1)

frame_files = sorted([f for f in dataset_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
print(f"    ✓ Found {len(frame_files)} frame files in dataset")
if len(frame_files) >= 2:
    print(f"    ✓ First frame: {frame_files[0].name}")
    print(f"    ✓ Last frame: {frame_files[-1].name}")

# Check 3: FramePairLoader instantiation
print("\n[3] Testing FramePairLoader...")
try:
    loader = FramePairLoader(str(dataset_path), show_progress=False)
    print(f"    ✓ Loader initialized successfully")
    print(f"    ✓ Total frames available: {loader.total_frames}")
    print(f"    ✓ Frame pairs in full range: {len(loader)}")
except Exception as e:
    print(f"    ✗ Loader failed: {e}")
    sys.exit(1)

# Check 4: Frame range validation
print("\n[4] Testing frame range support...")
try:
    loader_range = FramePairLoader(str(dataset_path), start_idx=10, end_idx=59, show_progress=False)
    print(f"    ✓ Range loader (10-59) created")
    print(f"    ✓ Frame pairs in range: {len(loader_range)}")
except Exception as e:
    print(f"    ✗ Range loader failed: {e}")
    sys.exit(1)

# Check 5: Direct frame access
print("\n[5] Testing direct frame access...")
try:
    frame_0 = loader.get_frame(0)
    print(f"    ✓ Frame 0 loaded: {frame_0.shape}")
    
    frame_path = loader.get_frame_path(0)
    print(f"    ✓ Frame 0 path: {frame_path.name}")
except Exception as e:
    print(f"    ✗ Direct access failed: {e}")
    sys.exit(1)

# Check 6: Configuration
print("\n[6] Checking configuration...")
print(f"    ✓ Grid configuration: {GRID_ROWS}x{GRID_COLS}")

# Check 7: OpenCV and NumPy
print("\n[7] Checking dependencies...")
try:
    import cv2
    import numpy as np
    print(f"    ✓ OpenCV version: {cv2.__version__}")
    print(f"    ✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"    ✗ Dependency missing: {e}")
    sys.exit(1)

# Check 8: tqdm optional import
print("\n[8] Checking optional dependencies...")
try:
    from tqdm import tqdm
    print(f"    ✓ tqdm available (progress bars enabled)")
except ImportError:
    print(f"    ⚠ tqdm not available (progress bars will be silent)")

# Summary
print("\n" + "="*60)
print("VALIDATION COMPLETE - All checks passed!")
print("="*60)

print("""
Next steps:
1. Run demo: python demo_motion_analysis.py
2. Process quick sample: 
   from src.motion_analysis import process
   stats = process('mall_dataset/frames', end_idx=99, save_frames=False)
3. Check MOTION_ANALYSIS_OPTIMIZATION.md for usage guide
4. Review OPTIMIZATION_SUMMARY.md for detailed changes
""")
