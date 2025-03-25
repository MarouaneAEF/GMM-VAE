#!/bin/bash

# Create directories if they don't exist
mkdir -p results
mkdir -p models

# Get the absolute path to the data directory
ABSOLUTE_DATA_DIR="../vae-cursor/augmented_images"

echo "Data directory: $ABSOLUTE_DATA_DIR"

# Check if data directory exists
if [ ! -d "$ABSOLUTE_DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $ABSOLUTE_DATA_DIR"
    exit 1
fi

# Count images in the directory
IMAGE_COUNT=$(find "$ABSOLUTE_DATA_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
echo "Found $IMAGE_COUNT high-resolution images in the dataset"
echo "Training will use a limited subset of 500 images for faster training"

# Set environment variables for optimal PyTorch performance on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run advanced training on the custom dataset with enhanced architecture
python train_gmvae.py \
    --dataset custom \
    --data-dir "$ABSOLUTE_DATA_DIR" \
    --device mps \
    --batch-size 4 \
    --max-images 500 \
    --epochs 150 \
    --lr 0.00003 \
    --K 15 \
    --x-size 400 \
    --w-size 300 \
    --hidden-size 1000 \
    --save-interval 5 \
    --parallel-compute \
    --random-sample \
    --kl-weight 0.01 \
    --kl-anneal \
    --recon-weight 20.0 \
    --clip-grad 0.5

echo "Advanced training complete! Check the 'results' directory for outputs." 