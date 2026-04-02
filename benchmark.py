"""
Reconstruction Benchmark for GMM-VAE
=====================================
Computes standard image reconstruction metrics on the test set:
  - MSE    : Mean Squared Error (pixel level)
  - PSNR   : Peak Signal-to-Noise Ratio  (dB)
  - SSIM   : Structural Similarity Index
  - LPIPS  : Learned Perceptual Image Patch Similarity (AlexNet)
  - FID    : Fréchet Inception Distance  (distribution-level)

Usage
-----
# Single model
python benchmark.py --model models/gmvae_cifar10_K10_best.pt --dataset cifar10

# Compare multiple checkpoints
python benchmark.py \
    --model models/gmvae_cifar10_K10_epoch_50.pt \
            models/gmvae_cifar10_K10_epoch_100.pt \
            models/gmvae_cifar10_K10_best.pt \
    --dataset cifar10

# Custom dataset
python benchmark.py --model models/gmvae_custom_K10_best.pt \
    --dataset custom --data-dir ../vae/augmented_images \
    --target-height 64 --target-width 64
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm

import lpips as lpips_lib
import piq
from torchmetrics.image.fid import FrechetInceptionDistance

from GM_VAE import GMVAE
import dataloader as dl


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GMM-VAE Reconstruction Benchmark")
    p.add_argument("--model", nargs="+", required=True,
                   help="Path(s) to model checkpoint(s) to benchmark")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["mnist", "cifar10", "custom"])
    p.add_argument("--data-dir", type=str, default="../vae/augmented_images",
                   help="Image directory (custom dataset only)")
    p.add_argument("--target-width",  type=int, default=None)
    p.add_argument("--target-height", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="mps",
                   choices=["mps", "cuda", "cpu"])
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--x-size", type=int, default=128)
    p.add_argument("--w-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=None,
                   help="Max test images to use (None = full test set)")
    p.add_argument("--save-grid", action="store_true",
                   help="Save a visual comparison grid per model")
    p.add_argument("--out-dir", type=str, default="benchmark_results",
                   help="Directory to save results")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_mse(recon: torch.Tensor, real: torch.Tensor) -> float:
    return F.mse_loss(recon, real).item()

def compute_psnr(recon: torch.Tensor, real: torch.Tensor) -> float:
    # piq.psnr expects [0,1] float tensors, returns dB
    return piq.psnr(recon.clamp(0, 1), real.clamp(0, 1),
                    data_range=1.0, reduction="mean").item()

def compute_ssim(recon: torch.Tensor, real: torch.Tensor) -> float:
    return piq.ssim(recon.clamp(0, 1), real.clamp(0, 1),
                    data_range=1.0).item()


class MetricAccumulator:
    """Streams batches and accumulates per-image metrics."""

    def __init__(self, device: str):
        self.device = device

        # LPIPS — run on same device as model
        self.lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
        self.lpips_fn.eval()

        # FID — deferred: collect uint8 buffers, compute once at the end
        # (Inception on CPU is slow per-batch; one big pass is far faster)
        self._fid_real  = []
        self._fid_recon = []

        self._mse   = []
        self._psnr  = []
        self._ssim  = []
        self._lpips = []

    @torch.no_grad()
    def update(self, recon: torch.Tensor, real: torch.Tensor):
        """recon, real: float32 in [0,1], shape [B,C,H,W]"""
        r = recon.clamp(0, 1)
        t = real.clamp(0, 1)

        # Pixel-level metrics
        self._mse.append(F.mse_loss(r, t).item())
        self._psnr.append(compute_psnr(r, t))
        self._ssim.append(compute_ssim(r, t))

        # LPIPS expects [-1,1]
        lp = self.lpips_fn(r * 2 - 1, t * 2 - 1).mean().item()
        self._lpips.append(lp)

        # Accumulate uint8 tensors for FID (keep on CPU, 3-channel)
        r_u8 = (r * 255).to(torch.uint8).cpu()
        t_u8 = (t * 255).to(torch.uint8).cpu()
        if r_u8.shape[1] == 1:
            r_u8 = r_u8.repeat(1, 3, 1, 1)
            t_u8 = t_u8.repeat(1, 3, 1, 1)
        self._fid_real.append(t_u8)
        self._fid_recon.append(r_u8)

    def compute(self) -> dict:
        # FID: feed all images in one shot on CPU (avoids per-batch Inception overhead)
        fid = FrechetInceptionDistance(feature=2048, normalize=False)
        real_all  = torch.cat(self._fid_real,  dim=0)
        recon_all = torch.cat(self._fid_recon, dim=0)
        fid.update(real_all,  real=True)
        fid.update(recon_all, real=False)
        fid_score = fid.compute().item()

        return {
            "MSE":   float(np.mean(self._mse)),
            "PSNR":  float(np.mean(self._psnr)),
            "SSIM":  float(np.mean(self._ssim)),
            "LPIPS": float(np.mean(self._lpips)),
            "FID":   fid_score,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, args, img_height: int, img_width: int,
               input_channels: int, device: str) -> GMVAE:
    model = GMVAE(
        input_channels=input_channels,
        img_height=img_height,
        img_width=img_width,
        hidden_size=args.hidden_size,
        x_size=args.x_size,
        w_size=args.w_size,
        K=args.K,
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    # Handle DataParallel / torch.compile prefixes
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(model: GMVAE, test_loader, device: str,
                  n_samples: int | None, save_grid_path: str | None) -> dict:
    acc = MetricAccumulator(device)
    total = 0
    first_batch_saved = False

    pbar = tqdm(test_loader, desc="  Evaluating", leave=False)
    for real, _ in pbar:
        real = real.to(device)

        with torch.no_grad():
            _, _, _, _, _, recon, _, _, _ = model(real)

        acc.update(recon, real)
        total += real.shape[0]

        # Save visual grid from first batch
        if save_grid_path and not first_batch_saved:
            n = min(8, real.shape[0])
            pairs = []
            for i in range(n):
                pairs += [real[i].cpu(), recon[i].cpu().clamp(0, 1)]
            grid = vutils.make_grid(pairs, nrow=4, padding=2, normalize=False)
            from torchvision.transforms.functional import to_pil_image
            to_pil_image(grid).save(save_grid_path)
            first_batch_saved = True

        if n_samples and total >= n_samples:
            break

    metrics = acc.compute()
    metrics["n_images"] = total
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Report formatting
# ──────────────────────────────────────────────────────────────────────────────

def print_table(results: dict[str, dict]):
    header = f"{'Model':<45} {'MSE':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FID':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for name, m in results.items():
        short = Path(name).name
        print(f"{short:<45} {m['MSE']:>8.5f} {m['PSNR']:>8.2f} "
              f"{m['SSIM']:>8.4f} {m['LPIPS']:>8.4f} {m['FID']:>10.2f}")
    print(sep + "\n")

    # Best per metric
    print("Best per metric:")
    for metric, low_is_better in [("MSE", True), ("PSNR", False),
                                   ("SSIM", False), ("LPIPS", True), ("FID", True)]:
        key = min if low_is_better else max
        best = key(results, key=lambda n: results[n][metric])
        arrow = "↓" if low_is_better else "↑"
        print(f"  {metric} {arrow}  →  {Path(best).name}  ({results[best][metric]:.4f})")
    print()


def save_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(out_dir, "benchmark.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    csv_path = os.path.join(out_dir, "benchmark.csv")
    metrics = ["MSE", "PSNR", "SSIM", "LPIPS", "FID", "n_images"]
    with open(csv_path, "w") as f:
        f.write("model," + ",".join(metrics) + "\n")
        for name, m in results.items():
            row = [Path(name).name] + [str(m.get(k, "")) for k in metrics]
            f.write(",".join(row) + "\n")

    print(f"Results saved to {out_dir}/benchmark.json and .csv")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve device
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"Device: {args.device}")

    # Load dataset
    if args.dataset == "mnist":
        _, test_loader = dl.mnistloader(args.batch_size)
        img_height = img_width = 28
        input_channels = 1
    elif args.dataset == "cifar10":
        _, test_loader = dl.cifar10loader(args.batch_size)
        img_height = img_width = 32
        input_channels = 3
    else:
        target_size = None
        if args.target_width and args.target_height:
            target_size = (args.target_width, args.target_height)
        _, test_loader, (img_height, img_width) = dl.custom_dataloader(
            args.data_dir, args.batch_size, target_size=target_size)
        input_channels = 3

    print(f"Dataset: {args.dataset}  |  Image size: {img_height}×{img_width}  "
          f"|  Test batches: {len(test_loader)}\n")

    all_results = {}

    for ckpt in args.model:
        print(f"{'─'*60}")
        print(f"Checkpoint: {Path(ckpt).name}")
        if not os.path.exists(ckpt):
            print(f"  ✗  File not found, skipping.")
            continue

        t0 = time.time()
        try:
            model = load_model(ckpt, args, img_height, img_width, input_channels, args.device)
        except RuntimeError as e:
            print(f"  ✗  Architecture mismatch — checkpoint incompatible with current model config.")
            print(f"     Hint: this checkpoint may have been saved with different --hidden-size / --x-size / --w-size.")
            print(f"     {str(e)[:120]}...")
            continue

        grid_path = None
        if args.save_grid:
            os.makedirs(args.out_dir, exist_ok=True)
            stem = Path(ckpt).stem
            grid_path = os.path.join(args.out_dir, f"grid_{stem}.png")

        metrics = run_benchmark(model, test_loader, args.device,
                                args.n_samples, grid_path)
        elapsed = time.time() - t0

        print(f"  MSE={metrics['MSE']:.5f}  PSNR={metrics['PSNR']:.2f}dB  "
              f"SSIM={metrics['SSIM']:.4f}  LPIPS={metrics['LPIPS']:.4f}  "
              f"FID={metrics['FID']:.2f}  "
              f"({metrics['n_images']} images, {elapsed:.1f}s)")
        if grid_path:
            print(f"  Grid saved → {grid_path}")

        all_results[ckpt] = metrics

    if len(all_results) > 1:
        print_table(all_results)

    save_results(all_results, args.out_dir)


if __name__ == "__main__":
    main()
