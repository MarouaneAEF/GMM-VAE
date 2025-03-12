#!/bin/bash

# Create directories if they don't exist
mkdir -p results
mkdir -p models

# Set environment variables for optimal PyTorch performance on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "Starting GM-VAE training on CIFAR-10 dataset..."

# Run training with parameters optimized for CIFAR-10
python train_gmvae.py \
    --dataset cifar10 \
    --device mps \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.0001 \
    --K 10 \
    --x-size 64 \
    --w-size 32 \
    --hidden-size 512 \
    --target-width 32 \
    --target-height 32 \
    --save-interval 5 \
    --parallel-compute \
    --kl-weight 0.1 \
    --kl-anneal \
    --recon-weight 10.0 \
    --clip-grad 0.5

echo "CIFAR-10 training complete! Check the 'results/gmvae_cifar10_K10' directory for outputs."