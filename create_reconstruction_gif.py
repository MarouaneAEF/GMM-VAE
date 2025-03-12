#!/usr/bin/env python3
import os
import glob
from PIL import Image
import argparse

def create_gif(source_dir, output_path, pattern="large_comparison_epoch_*.png", duration=500, loop=0):
    """
    Create a GIF animation from a series of images.
    
    Args:
        source_dir: Directory containing image files
        output_path: Path to save the resulting GIF
        pattern: Glob pattern to match image files
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    # Find all matching images
    image_paths = sorted(glob.glob(os.path.join(source_dir, pattern)))
    
    if not image_paths:
        print(f"No images found matching pattern '{pattern}' in {source_dir}")
        return False
        
    print(f"Found {len(image_paths)} images. Creating GIF animation...")
    
    # Load images
    images = [Image.open(path) for path in image_paths]
    
    # Get the first image
    first_img = images[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GIF
    first_img.save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )
    
    print(f"GIF animation saved to {output_path}")
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
                      help="Duration of each frame in milliseconds")
    
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
    create_gif(source_dir, args.output, pattern, args.duration)

if __name__ == "__main__":
    main() 