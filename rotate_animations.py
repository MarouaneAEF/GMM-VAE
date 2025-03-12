#!/usr/bin/env python3
import os
import glob
from PIL import Image, ImageSequence
import argparse

def rotate_gif(input_path, output_path, rotation=90):
    """
    Rotate each frame in a GIF animation and save as a new GIF.
    
    Args:
        input_path: Path to the input GIF
        output_path: Path to save the rotated GIF
        rotation: Rotation angle in degrees (90, 180, 270)
    """
    print(f"Rotating {input_path} by {rotation} degrees...")
    
    # Open the GIF
    gif = Image.open(input_path)
    
    # Extract frames and rotate each one
    frames = []
    durations = []
    
    for frame in ImageSequence.Iterator(gif):
        # Convert to RGBA to preserve transparency
        frame_rgba = frame.convert('RGBA')
        
        # Rotate the frame
        rotated_frame = frame_rgba.rotate(rotation, expand=True)
        
        # Append to frames list
        frames.append(rotated_frame)
        
        # Store duration of this frame
        try:
            durations.append(frame.info['duration'])
        except:
            durations.append(100)  # Default duration if not specified
    
    # Get the first frame
    first_frame = frames[0]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GIF
    first_frame.save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False
    )
    
    print(f"Rotated GIF saved to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Rotate GIF animations")
    parser.add_argument("--input", type=str, required=True, 
                      help="Input GIF file")
    parser.add_argument("--output", type=str, default=None,
                      help="Output path (default: input_rotated.gif)")
    parser.add_argument("--angle", type=int, default=90, choices=[90, 180, 270],
                      help="Rotation angle in degrees")
    
    args = parser.parse_args()
    
    # Determine output path if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_rotated{ext}"
    
    # Rotate the GIF
    rotate_gif(args.input, args.output, args.angle)

if __name__ == "__main__":
    main() 