#!/usr/bin/env python
"""
Simple script to save motion analysis visualization frames.
Method 1: Save frames without display windows.
"""

from src.motion_analysis import process
from pathlib import Path

print("Motion Analysis - Frame Saving (Method 1)")
print("=" * 60)

# Get user input
try:
    end_frame = int(input("How many frames to process? (default 50): ") or "50")
except ValueError:
    end_frame = 50

print(f"\nProcessing first {end_frame} frames...")
print("(This generates motion analysis visualizations as PNG images)")
print("-" * 60)

# Process and save frames
stats = process(
    'mall_dataset/frames',
    end_idx=end_frame,
    display=False,           # No windows
    save_frames=True,        # Save as PNG images
    start_idx=0,
    skip_frames=1
)

# Show results
print("\n" + "=" * 60)
print("SAVED FRAMES LOCATION: output_frames/")
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

if __name__ == "__main__":
    source_path = r"C:\Users\Lokesh Kumar\Downloads\mall_dataset\mall_dataset\frames"
    process(source_path, display=True)