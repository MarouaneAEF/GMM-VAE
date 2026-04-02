#!/bin/bash
mkdir -p results models

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "Training GMM-VAE on CIFAR-10..."

python train_gmvae.py \
    --dataset cifar10 \
    --device mps \
    --batch-size 128 \
    --epochs 100 \
    --lr 3e-4 \
    --K 10 \
    --x-size 256 \
    --w-size 128 \
    --hidden-size 512 \
    --save-interval 10 \
    --kl-weight 0.05 \
    --kl-anneal \
    --kl-anneal-epochs 40 \
    --recon-weight 1.0 \
    --clip-grad 1.0 \
    --lr-scheduler \
    --perceptual \
    --perceptual-weight 0.5

echo "Done. Results in results/gmvae_cifar10_K10/"
