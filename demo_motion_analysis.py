"""
Demo script for Motion Analysis on Mall Dataset

This script demonstrates various use cases for the optimized motion analysis module
with your 2000-frame mall dataset.
"""

from src.motion_analysis import process, run_synthetic_demo, FramePairLoader, GRID_ROWS, GRID_COLS
import time

def demo_quick_test():
    """Process first 50 frames for quick validation."""
    print("\n" + "="*60)
    print("DEMO 1: Quick Test (First 50 frames)")
    print("="*60)
    
    start_time = time.time()
    
    # Process only frames 0-49 (50 frames = 49 pairs)
    stats = process(
        source="mall_dataset/frames",
        display=False,
        save_frames=False,  # Don't save to keep it fast
        start_idx=0,
        end_idx=49,
        skip_frames=1
    )
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    return stats


def demo_sampling():
    """Process every 10th frame to reduce computation."""
    print("\n" + "="*60)
    print("DEMO 2: Sampling (Process every 10th frame)")
    print("="*60)
    
    start_time = time.time()
    
    # Process every 10th frame (200 frames = 20 processing operations)
    stats = process(
        source="mall_dataset/frames",
        display=False,
        save_frames=False,
        skip_frames=10  # Process every 10th frame
    )
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    return stats


def demo_full_processing():
    """Process all frames with output saving."""
    print("\n" + "="*60)
    print("DEMO 3: Full Processing (All frames with output)")
    print("="*60)
    
    start_time = time.time()
    
    # Process all frames and save outputs
    stats = process(
        source="mall_dataset/frames",
        display=False,
        save_frames=True,
        output_dir="output_frames",
        skip_frames=1
    )
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    return stats


def demo_frame_loader():
    """Show usage of the FramePairLoader directly."""
    print("\n" + "="*60)
    print("DEMO 4: Direct FramePairLoader Usage")
    print("="*60)
    
    # Create loader for frames 0-99
    loader = FramePairLoader("mall_dataset/frames", start_idx=0, end_idx=99)
    print(f"Total frames in source: {loader.total_frames}")
    print(f"Frame pairs available: {len(loader)}")
    
    # Get specific frame
    frame_0 = loader.get_frame(0)
    print(f"Frame 0 shape: {frame_0.shape}")
    print(f"Frame 0 path: {loader.get_frame_path(0)}")
    
    # Iterate through pairs
    print(f"\nIterating through {len(loader)} frame pairs:")
    for i, (prev_frame, curr_frame) in enumerate(loader):
        if i < 3:
            print(f"  Pair {i}: prev={prev_frame.shape}, curr={curr_frame.shape}")
        if i >= 2:
            print(f"  ... (and {len(loader) - 3} more pairs)")
            break


def demo_synthetic_validation():
    """Generate synthetic data for testing without real data."""
    print("\n" + "="*60)
    print("DEMO 5: Synthetic Data Validation")
    print("="*60)
    
    start_time = time.time()
    
    # Generate synthetic moving blob sequence
    run_synthetic_demo(n_frames=30)
    
    elapsed = time.time() - start_time
    print(f"Synthetic demo completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Motion Analysis Module - Demo Suite")
    print("="*60)
    print(f"Grid Configuration: {GRID_ROWS}x{GRID_COLS}")
    
    # Run demonstrations
    choice = input("""
Choose a demo to run:
1. Quick test (first 50 frames)
2. Sampling (every 10th frame)
3. Full processing (all frames)
4. FramePairLoader usage
5. Synthetic validation
6. Run all demos

Enter choice (1-6): """).strip()
    
    if choice == "1":
        demo_quick_test()
    elif choice == "2":
        demo_sampling()
    elif choice == "3":
        demo_full_processing()
    elif choice == "4":
        demo_frame_loader()
    elif choice == "5":
        demo_synthetic_validation()
    elif choice == "6":
        demo_quick_test()
        demo_sampling()
        demo_frame_loader()
        demo_synthetic_validation()
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("Demo Complete")
    print("="*60 + "\n")
