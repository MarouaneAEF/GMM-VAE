#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont
import argparse

def create_progress_bar(output_path, frames=20, width=800, height=100):
    """Create a horizontal progress bar animation."""
    print(f"Creating {frames} frames...")
    
    # Create frames
    frames_list = []
    bar_margin = 20
    bar_height = 30
    bar_y = (height - bar_height) // 2
    bar_width = width - (2 * bar_margin)
    
    for i in range(frames):
        # Create image with white background
        img = Image.new('RGB', (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Calculate progress
        progress = (i + 1) / frames
        
        # Draw background bar
        draw.rectangle(
            [(bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height)],
            fill=(220, 220, 220)
        )
        
        # Draw progress bar
        progress_width = int(bar_width * progress)
        draw.rectangle(
            [(bar_margin, bar_y), (bar_margin + progress_width, bar_y + bar_height)],
            fill=(0, 122, 255)
        )
        
        # Add text
        try:
            font = ImageFont.load_default()
            text = f"Epoch {i+1}/{frames} - {int(progress*100)}%"
            draw.text((bar_margin + 10, bar_y + 5), text, fill=(255, 255, 255), font=font)
        except:
            pass
        
        frames_list.append(img)
    
    # Save as GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames_list[0].save(
        output_path,
        save_all=True,
        append_images=frames_list[1:],
        duration=150,
        loop=0,
        optimize=True
    )
    print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="animations/progress/bar.gif")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    
    create_progress_bar(args.output, args.frames) 