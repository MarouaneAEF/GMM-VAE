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

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Run distributed training with 4 processes (suitable for multi-GPU systems)
python -m torch.distributed.launch --nproc_per_node=4 distributed_train.py \
    --dataset custom \
    --data-dir "$ABSOLUTE_DATA_DIR" \
    --batch-size 4 \
    --max-images 500 \
    --epochs 100 \
    --lr 0.0001 \
    --K 10 \
    --x-size 200 \
    --w-size 150 \
    --hidden-size 500 \
    --save-interval 5 \
    --parallel-compute \
    --random-sample \
    --kl-weight 0.1 \
    --kl-anneal \
    --recon-weight 10.0 \
    --clip-grad 1.0

echo "Distributed training complete! Check the 'results' directory for outputs." 