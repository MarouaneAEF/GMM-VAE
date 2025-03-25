#!/bin/bash

# Create directory for GIFs
mkdir -p gifs

# Check if results directory exists
if [ ! -d "results/gmvae_custom_K10" ]; then
    echo "Error: Results directory not found: results/gmvae_custom_K10"
    echo "Please run training first to generate reconstruction images"
    exit 1
fi

# Install required packages if not already installed
echo "Checking for required Python packages..."
pip install imageio matplotlib numpy pillow

# Run the GIF creation script
echo "Creating GIFs from training results..."
python create_reconstruction_gif.py --results-dir results/gmvae_custom_K10 --output-dir gifs --fps 1

echo "GIF creation complete! Check the 'gifs' directory for the animations."
echo "Generated GIFs:"
ls -la gifs/ 