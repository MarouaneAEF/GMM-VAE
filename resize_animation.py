#!/usr/bin/env python3
import os
import glob
from PIL import Image, ImageSequence
import argparse

def resize_and_rotate_gif(input_path, output_path, max_width=600, max_height=300, rotation=270):
    """
    Resize and rotate a GIF animation to make it smaller and properly proportioned.
    
    Args:
        input_path: Path to the input GIF
        output_path: Path to save the resized/rotated GIF
        max_width: Maximum width for the output GIF
        max_height: Maximum height for the output GIF
        rotation: Rotation angle in degrees (0, 90, 180, 270)
    """
    print(f"Processing {input_path}...")
    
    # Open the GIF
    gif = Image.open(input_path)
    
    # Extract frames
    frames = []
    durations = []
    
    for frame in ImageSequence.Iterator(gif):
        # Convert to RGBA to preserve transparency
        frame_rgba = frame.convert('RGBA')
        
        # Rotate if needed
        if rotation != 0:
            frame_rgba = frame_rgba.rotate(rotation, expand=True)
        
        # Calculate size while maintaining aspect ratio
        width, height = frame_rgba.size
        
        if width > max_width or height > max_height:
            # Resize to fit within bounds while maintaining aspect ratio
            if width / max_width > height / max_height:
                # Width is the limiting factor
                new_width = max_width
                new_height = int(height * (max_width / width))
            else:
                # Height is the limiting factor
                new_height = max_height
                new_width = int(width * (max_height / height))
            
            # Resize the frame
            frame_rgba = frame_rgba.resize((new_width, new_height), Image.LANCZOS)
        
        # Append to frames list
        frames.append(frame_rgba)
        
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
        optimize=True
    )
    
    print(f"Processed GIF saved to {output_path}")
    print(f"New dimensions: {first_frame.size}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Resize and rotate GIF animations")
    parser.add_argument("--input", type=str, required=True, 
                      help="Input GIF file")
    parser.add_argument("--output", type=str, default=None,
                      help="Output path (default: input_resized.gif)")
    parser.add_argument("--max-width", type=int, default=600,
                      help="Maximum width for the output GIF")
    parser.add_argument("--max-height", type=int, default=300,
                      help="Maximum height for the output GIF")
    parser.add_argument("--rotation", type=int, default=270, choices=[0, 90, 180, 270],
                      help="Rotation angle in degrees")
    
    args = parser.parse_args()
    
    # Determine output path if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_resized{ext}"
    
    # Resize and rotate the GIF
    resize_and_rotate_gif(args.input, args.output, args.max_width, args.max_height, args.rotation)

if __name__ == "__main__":
    main() 