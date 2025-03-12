#!/usr/bin/env python
import os
import glob
from datetime import datetime

def find_latest_images(base_path="results", limit=5):
    """Find the latest reconstructed images and print their paths."""
    image_files = []
    
    # Find all PNG files
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)
                mod_time = os.path.getmtime(full_path)
                image_files.append((full_path, mod_time))
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: x[1], reverse=True)
    
    # Print the newest files
    print(f"Found {len(image_files)} total images. Most recent {limit} images:")
    for i, (path, mod_time) in enumerate(image_files[:limit]):
        time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i+1}. {path} (modified: {time_str})")
        # Print absolute path for easier locating
        abs_path = os.path.abspath(path)
        print(f"   Absolute path: {abs_path}")
        print()

if __name__ == "__main__":
    find_latest_images()
    
    # Also print out specific paths that should always exist after training
    large_comparisons = glob.glob("results/*/reconstructions/large_comparisons/large_comparison_epoch_*.png")
    if large_comparisons:
        print("\nLarge comparison examples (often the most informative):")
        for path in sorted(large_comparisons)[-3:]:
            print(f" - {os.path.abspath(path)}") 