#!/bin/bash

# Create directories if they don't exist
mkdir -p results
mkdir -p models

# Set the correct path to the user's high-resolution photos
CURRENT_DIR=$(pwd)
ABSOLUTE_DATA_DIR="$CURRENT_DIR/../vae/augmented_images"

echo "Using data directory: $ABSOLUTE_DATA_DIR"

# Check if data directory exists
if [ ! -d "$ABSOLUTE_DATA_DIR" ]; then
    echo "Error: Data directory not found: $ABSOLUTE_DATA_DIR"
    echo "Please make sure your high-resolution photos directory exists at $ABSOLUTE_DATA_DIR"
    exit 1
fi

# Count number of images in directory
IMAGE_COUNT=$(find "$ABSOLUTE_DATA_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
echo "Found $IMAGE_COUNT high-resolution images in the dataset"
echo "Training will use a limited subset of 1000 images for faster training"

# Set environment variables for optimal MPS performance on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "Using Apple Silicon GPU (MPS backend) with optimized settings"

# The --parallel-compute flag enables Apple Silicon specific optimizations
python train_gmvae.py \
    --dataset custom \
    --data-dir "$ABSOLUTE_DATA_DIR" \
    --device mps \
    --batch-size 4 \
    --max-images 1000 \
    --epochs 100 \
    --lr 0.0002 \
    --K 10 \
    --x-size 200 \
    --w-size 150 \
    --hidden-size 500 \
    --save-interval 5 \
    --parallel-compute

echo "GPU training complete! Check the 'results' directory for outputs." 