#!/usr/bin/env python3
import os
import glob
import numpy as np
from PIL import Image
import argparse


def _crossfade(img_a, img_b, steps):
    """
    Return a list of PIL images blending img_a → img_b with smoothstep easing.
    The endpoint img_b is NOT included (caller appends the next key-frame).
    """
    a = np.array(img_a, dtype=np.float32)
    b = np.array(img_b, dtype=np.float32)
    blended = []
    for i in range(steps):
        t = i / steps
        # Smoothstep easing: slow start and end, fast middle
        t_smooth = t * t * (3 - 2 * t)
        frame_arr = ((1 - t_smooth) * a + t_smooth * b).clip(0, 255).astype(np.uint8)
        blended.append(Image.fromarray(frame_arr))
    return blended


def create_gif(source_dir, output_path, pattern="large_comparison_epoch_*.png",
               duration=500, loop=0, crossfade_steps=0):
    """
    Create a GIF animation from a series of images.

    Args:
        source_dir: Directory containing image files
        output_path: Path to save the resulting GIF
        pattern: Glob pattern to match image files
        duration: Duration of each key-frame in milliseconds
        loop: Number of loops (0 = infinite)
        crossfade_steps: Number of blended transition frames inserted between
                         each pair of key-frames (0 = no crossfade, plain cut)
    """
    # Find all matching images (sorted by epoch number embedded in filename)
    image_paths = sorted(
        glob.glob(os.path.join(source_dir, pattern)),
        key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0'),
    )

    if not image_paths:
        print(f"No images found matching pattern '{pattern}' in {source_dir}")
        return False

    print(f"Found {len(image_paths)} images. Creating GIF animation...")

    # Load and ensure consistent size
    images = [Image.open(p).convert("RGB") for p in image_paths]
    target_size = images[0].size
    images = [img.resize(target_size, Image.LANCZOS) if img.size != target_size else img
              for img in images]

    frames = []
    frame_durations = []

    if crossfade_steps > 0:
        # Build frames with smooth crossfade transitions
        blend_duration = max(20, duration // max(1, crossfade_steps))
        for idx in range(len(images)):
            # Key-frame itself (held briefly so the eye registers it)
            frames.append(images[idx])
            frame_durations.append(duration // 3)

            # Transition to next key-frame
            next_idx = (idx + 1) % len(images)
            transition = _crossfade(images[idx], images[next_idx], crossfade_steps)
            frames.extend(transition)
            frame_durations.extend([blend_duration] * len(transition))
    else:
        frames = images
        frame_durations = [duration] * len(images)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_durations,
        loop=loop,
        optimize=False,
    )

    print(f"GIF animation saved to {output_path} ({len(frames)} total frames)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create GIF animation from reconstruction images")
    parser.add_argument("--model", type=str, default="cifar10", 
                      help="Model name (cifar10, custom_K10, custom_K15, etc.)")
    parser.add_argument("--type", type=str, default="large_comparisons",
                      choices=["standard", "comparisons", "large_comparisons", "clusters"],
                      help="Type of reconstruction images to use")
    parser.add_argument("--k", type=int, default=10,
                      help="K value for the model")
    parser.add_argument("--output", type=str, default=None,
                      help="Output path (default: animations/model_type.gif)")
    parser.add_argument("--duration", type=int, default=500,
                      help="Duration of each key-frame in milliseconds")
    parser.add_argument("--crossfade", type=int, default=8,
                      help="Number of blended transition frames between key-frames (0 = no crossfade)")

    args = parser.parse_args()

    # Determine output path if not specified
    if args.output is None:
        os.makedirs("animations", exist_ok=True)
        args.output = f"animations/{args.model}_K{args.k}_{args.type}.gif"

    # Determine source directory
    if args.model == "cifar10":
        source_dir = f"results/gmvae_{args.model}_K{args.k}/reconstructions/{args.type}"
    else:
        source_dir = f"results/gmvae_{args.model}/reconstructions/{args.type}"

    # Create pattern based on type
    if args.type == "standard":
        pattern = "reconstruction_epoch_*.png"
    elif args.type == "comparisons":
        pattern = "comparison_epoch_*.png"
    elif args.type == "large_comparisons":
        pattern = "large_comparison_epoch_*.png"
    elif args.type == "clusters":
        pattern = "clusters_epoch_*.png"

    # Create GIF
    create_gif(source_dir, args.output, pattern, args.duration, crossfade_steps=args.crossfade)

if __name__ == "__main__":
    main() 