import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import argparse
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Import our model and data utilities
from GM_VAE import GMVAE, GMVAELoss
import dataloader as dl

def set_device(args):
    """Set and return the appropriate device based on availability and user preferences."""
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
        print(f"Using Apple Silicon GPU (MPS backend)")
    else:
        device = 'cpu'
        print("Using CPU for computation")
    
    return device

def parse_args():
    parser = argparse.ArgumentParser(description='Train GM-VAE')
    parser.add_argument('--dataset', type=str, default='custom', 
                        choices=['mnist', 'cifar10', 'custom'],
                        help='dataset to use (default: custom)')
    parser.add_argument('--data-dir', type=str, default='../vae-cursor/augmented_images',
                        help='directory containing the custom images (default: ../vae-cursor/augmented_images)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='maximum number of images to use for training (default: all images)')
    parser.add_argument('--random-sample', action='store_true',
                        help='randomly sample max-images instead of taking first ones')
    parser.add_argument('--target-width', type=int, default=None,
                        help='target width to resize images (default: None, preserve original size)')
    parser.add_argument('--target-height', type=int, default=None,
                        help='target height to resize images (default: None, preserve original size)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='device to use for training (mps for Apple Silicon, cuda for NVIDIA, cpu for CPU-only)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='how many epochs to wait before saving model')
    parser.add_argument('--K', type=int, default=10,
                        help='number of mixture components (default: 10)')
    parser.add_argument('--hidden-size', type=int, default=500,
                        help='size of hidden layer (default: 500)')
    parser.add_argument('--x-size', type=int, default=200,
                        help='size of latent variable x (default: 200)')
    parser.add_argument('--w-size', type=int, default=150,
                        help='size of latent variable w (default: 150)')
    parser.add_argument('--parallel-compute', action='store_true',
                        help='Enable parallelized computations optimized for Apple Silicon')
    parser.add_argument('--kl-weight', type=float, default=1.0,
                        help='Weight for KL divergence term (default: 1.0)')
    parser.add_argument('--kl-anneal', action='store_true',
                        help='Enable KL annealing for first 10 epochs')
    parser.add_argument('--recon-weight', type=float, default=1.0,
                        help='Weight for reconstruction loss (default: 1.0)')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0)')
    
    return parser.parse_args()

def train(model, train_loader, optimizer, epoch, device, args, writer):
    model.train()
    train_loss = 0
    # Track separate loss components
    component_losses = {}
    
    # Calculate KL weight for this epoch (for KL annealing)
    if args.kl_anneal:
        # Anneal KL weight from 0 to args.kl_weight during first 10 epochs
        kl_weight = min(args.kl_weight * epoch / 10, args.kl_weight) if epoch < 10 else args.kl_weight
    else:
        kl_weight = args.kl_weight
        
    # Reconstruction weight
    recon_weight = args.recon_weight
    
    # Log the current weights
    print(f"Epoch {epoch}: KL weight = {kl_weight:.4f}, Reconstruction weight = {recon_weight:.4f}")
    
    # Wrap the data loader with tqdm for a progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to device
        data = data.to(device)
        target = target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        mu_x, logvar_x, mu_px, logvar_px, qz, recon_x, mu_w, logvar_w, x_sample = model(data)
        
        # Compute loss with weighting
        loss, components = GMVAELoss.compute_loss(
            recon_x, data, mu_w, logvar_w, qz, mu_x, logvar_x, 
            mu_px, logvar_px, x_sample, model.x_size, model.K,
            kl_weight=kl_weight, recon_weight=recon_weight
        )
        
        # Backward pass and optimize
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        
        # Accumulate loss
        train_loss += loss.item()
        
        # Accumulate component losses
        for key, value in components.items():
            if key not in component_losses:
                component_losses[key] = 0.0
            # Check if value is a tensor or already a float
            if isinstance(value, torch.Tensor):
                component_losses[key] += value.item()
            else:
                component_losses[key] += value
            
        # Update progress bar with detailed loss components
        postfix_dict = {"loss": f"{loss.item() / len(data):.4f}"}
        if "kl_w" in components:
            kl_w_value = components["kl_w"].item() if isinstance(components["kl_w"], torch.Tensor) else components["kl_w"]
            postfix_dict["kl_w"] = f"{kl_w_value / len(data):.4f}"
        if "kl_x" in components:
            kl_x_value = components["kl_x"].item() if isinstance(components["kl_x"], torch.Tensor) else components["kl_x"]
            postfix_dict["kl_x"] = f"{kl_x_value / len(data):.4f}"
        if "recon" in components:
            recon_value = components["recon"].item() if isinstance(components["recon"], torch.Tensor) else components["recon"]
            postfix_dict["recon"] = f"{recon_value / len(data):.4f}"
        
        postfix_dict["kl_w"] = f"{kl_weight:.2f}"
        
        pbar.set_postfix(postfix_dict)
        
        # Log progress
        if batch_idx % args.log_interval == 0:
            # Add loss components to tensorboard
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item() / len(data), step)
            for name, value in components.items():
                writer.add_scalar(f'train/{name}', value / len(data), step)
    
    # Calculate average epoch loss
    avg_loss = train_loss / len(train_loader.dataset)
    avg_component_losses = {k: v / len(train_loader.dataset) for k, v in component_losses.items()}
    
    # Print detailed losses
    loss_str = f'Epoch: {epoch} Train loss: {avg_loss:.4f} | '
    for key, value in avg_component_losses.items():
        loss_str += f"{key}: {value:.4f} "
    print(loss_str)
    
    # Return average loss and component losses
    return avg_loss, avg_component_losses

def save_comparison_grid(original_images, reconstructed_images, save_path, num_images=8):
    """
    Save a grid showing original images and their reconstructions side by side.
    
    Args:
        original_images: Tensor of original images [B, C, H, W]
        reconstructed_images: Tensor of reconstructed images [B, C, H, W]
        save_path: Path to save the comparison grid
        num_images: Number of image pairs to include in the grid
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Limit to num_images
    orig = original_images[:num_images]
    recon = reconstructed_images[:num_images]
    
    # Create a side-by-side comparison
    comparison = []
    for i in range(min(num_images, len(orig))):
        # Add original and reconstruction side by side
        comparison.append(orig[i])
        comparison.append(recon[i])
    
    # Create a grid of images
    grid = vutils.make_grid(comparison, nrow=2, normalize=True, pad_value=1)
    
    # Convert to PIL Image and save
    grid_img = transforms.ToPILImage()(grid)
    grid_img.save(save_path)
    
    print(f"Saved comparison grid to {save_path}")

def test(model, test_loader, epoch, device, args, writer, save_dir):
    model.eval()
    test_loss = 0
    # Track separate loss components
    component_losses = {}
    
    # Calculate KL weight for this epoch (for KL annealing)
    if args.kl_anneal:
        # Anneal KL weight from 0 to args.kl_weight during first 10 epochs
        kl_weight = min(args.kl_weight * epoch / 10, args.kl_weight) if epoch < 10 else args.kl_weight
    else:
        kl_weight = args.kl_weight
        
    # Reconstruction weight
    recon_weight = args.recon_weight
    
    # Create dedicated reconstructions directory
    reconstructions_dir = f'{save_dir}/reconstructions'
    os.makedirs(reconstructions_dir, exist_ok=True)
    
    # Create subfolders for different types of reconstructions
    standard_dir = f'{reconstructions_dir}/standard'
    comparison_dir = f'{reconstructions_dir}/comparisons'
    large_dir = f'{reconstructions_dir}/large_comparisons'
    clusters_dir = f'{reconstructions_dir}/clusters'
    
    os.makedirs(standard_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(large_dir, exist_ok=True)
    os.makedirs(clusters_dir, exist_ok=True)
    
    # Lists to store original and reconstructed images for visualization
    original_images = []
    reconstructed_images = []
    
    # Add progress bar for test set
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f"Testing Epoch {epoch}", leave=False)
        for batch_idx, (data, target) in enumerate(test_pbar):
            # Move data to device
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass
            mu_x, logvar_x, mu_px, logvar_px, qz, recon_x, mu_w, logvar_w, x_sample = model(data)
            
            # Compute loss with weighting
            loss, components = GMVAELoss.compute_loss(
                recon_x, data, mu_w, logvar_w, qz, mu_x, logvar_x, 
                mu_px, logvar_px, x_sample, model.x_size, model.K,
                kl_weight=kl_weight, recon_weight=recon_weight
            )
            
            # Accumulate loss
            test_loss += loss.item()
            
            # Accumulate component losses
            for key, value in components.items():
                if key not in component_losses:
                    component_losses[key] = 0.0
                # Check if value is a tensor or already a float
                if isinstance(value, torch.Tensor):
                    component_losses[key] += value.item()
                else:
                    component_losses[key] += value
            
            # Update progress bar
            postfix_dict = {"loss": f"{loss.item() / len(data):.4f}"}
            if "kl_w" in components:
                kl_w_value = components["kl_w"].item() if isinstance(components["kl_w"], torch.Tensor) else components["kl_w"]
                postfix_dict["kl_w"] = f"{kl_w_value / len(data):.4f}"
            if "kl_x" in components:
                kl_x_value = components["kl_x"].item() if isinstance(components["kl_x"], torch.Tensor) else components["kl_x"]
                postfix_dict["kl_x"] = f"{kl_x_value / len(data):.4f}"
            if "recon" in components:
                recon_value = components["recon"].item() if isinstance(components["recon"], torch.Tensor) else components["recon"]
                postfix_dict["recon"] = f"{recon_value / len(data):.4f}"
                
            test_pbar.set_postfix(postfix_dict)
            
            # Store original and reconstructed images for the first batch only
            if batch_idx == 0:
                original_images = data.detach().cpu()
                reconstructed_images = recon_x.detach().cpu()
                
                # Save the traditional comparison grid in the standard folder
                comparison = torch.cat([data[:8], recon_x[:8]])
                save_image(comparison.cpu(),
                          f'{standard_dir}/reconstruction_epoch_{epoch}.png', nrow=8)
                
                # We've collected enough images, no need to process more
                break
    
    # Compute average loss
    test_loss /= len(test_loader.dataset)
    avg_component_losses = {k: v / len(test_loader.dataset) for k, v in component_losses.items()}
    
    # Log metrics
    loss_str = f'====> Test set loss: {test_loss:.4f} | '
    for key, value in avg_component_losses.items():
        loss_str += f"{key}: {value:.4f} "
    print(loss_str)
    
    writer.add_scalar('Loss/test', test_loss, epoch)
    for key, value in avg_component_losses.items():
        writer.add_scalar(f'test/{key}', value, epoch)
    
    # Save comparison of original and reconstructed images
    if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
        comparison_path = f"{comparison_dir}/comparison_epoch_{epoch}.png"
        save_comparison_grid(original_images, reconstructed_images, comparison_path)
        
        # Save a large version with more details for closer inspection
        large_comparison_path = f"{large_dir}/large_comparison_epoch_{epoch}.png"
        save_comparison_grid(original_images, reconstructed_images, large_comparison_path, num_images=16)
        
        # Visualize cluster assignments if model uses clustering
        if hasattr(model, 'K') and model.K > 1:
            cluster_assignments = torch.argmax(qz, dim=1)
            
            # Create visualization of cluster assignments if x has at least 2 dimensions
            if model.x_size >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(mu_x[:, 0].cpu().numpy(), mu_x[:, 1].cpu().numpy(), 
                                    c=cluster_assignments.cpu().numpy(), cmap='tab10', 
                                    alpha=0.6, s=10)
                ax.set_title(f'Latent Space Visualization (Epoch {epoch})')
                ax.set_xlabel('Latent Dimension 1')
                ax.set_ylabel('Latent Dimension 2')
                fig.colorbar(scatter, label='Cluster Assignment')
                plt.tight_layout()
                
                # Save figure to clusters directory
                plt.savefig(f'{clusters_dir}/clusters_epoch_{epoch}.png')
                plt.close(fig)
    
    # Return average loss and component losses
    return test_loss, avg_component_losses

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = set_device(args)
    
    print(f"Using device: {device}")
    
    # Create directories for saving results and models
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=f'runs/gmvae_{args.dataset}_K{args.K}')
    
    # Load dataset
    if args.dataset == 'mnist':
        train_loader, test_loader = dl.mnistloader(args.batch_size)
        input_channels = 1
    elif args.dataset == 'cifar10':
        train_loader, test_loader = dl.cifar10loader(args.batch_size)
        input_channels = 3
    elif args.dataset == 'custom':
        print(f"Loading custom dataset from {args.data_dir}")
        print(f"Maximum images to use: {args.max_images if args.max_images else 'All available'}")
        print(f"Random sampling: {'Enabled' if args.random_sample else 'Disabled'}")
        
        # Check if target size is specified
        target_size = None
        if args.target_width is not None and args.target_height is not None:
            target_size = (args.target_width, args.target_height)
            print(f"Target image size: {args.target_width}×{args.target_height}")
        
        train_loader, test_loader, (img_height, img_width) = dl.custom_dataloader(
            args.data_dir, 
            args.batch_size,
            preserve_size=(target_size is None),
            max_images=args.max_images,
            random_sample=args.random_sample,
            target_size=target_size
        )
        print(f"Dataset image size: {img_height}×{img_width}")
        input_channels = 3  # Assuming RGB images
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Initialize model
    model = GMVAE(
        input_channels=input_channels,
        img_height=None,  # Always use dynamic sizing
        img_width=None,   # Always use dynamic sizing
        hidden_size=args.hidden_size,
        x_size=args.x_size,
        w_size=args.w_size,
        K=args.K
    ).to(device)
    
    print(f"Model initialized with dynamic sizing for reconstruction")
    
    # Apple Silicon MPS optimizations if requested
    if args.parallel_compute and device == 'mps':
        print("Enabling optimized parallel compute for Apple Silicon...")
        # Optimize memory usage
        torch.mps.empty_cache()
        
        # Use parallel processing where possible
        # This doesn't use DDP (which has MPS issues) but leverages
        # Apple's parallelism at the Metal layer
        if hasattr(torch.backends.mps, 'set_benchmark_mode'):
            torch.backends.mps.set_benchmark_mode(True)
            print("MPS benchmark mode enabled")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory for this run
    dataset_name = args.dataset
    save_dir = f'results/gmvae_{dataset_name}_K{args.K}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dedicated directory for reconstruction comparisons
    reconstructions_dir = f'{save_dir}/reconstructions'
    os.makedirs(reconstructions_dir, exist_ok=True)
    
    # Copy README_reconstructions.md to the reconstructions directory if it exists
    readme_path = "README_reconstructions.md"
    if os.path.exists(readme_path):
        import shutil
        shutil.copy(readme_path, os.path.join(reconstructions_dir, "README.md"))
        print(f"Copied README to reconstructions directory")
    
    print(f"Reconstructed images will be saved to: {reconstructions_dir}")
    print(f"  - Standard reconstructions: {reconstructions_dir}/standard")
    print(f"  - Side-by-side comparisons: {reconstructions_dir}/comparisons")
    print(f"  - Large comparison grids: {reconstructions_dir}/large_comparisons")
    print(f"  - Cluster visualizations: {reconstructions_dir}/clusters")
    
    # Training loop
    best_loss = float('inf')
    print(f"Starting training for {args.epochs} epochs...")
    progress_bar = tqdm(range(1, args.epochs + 1), desc="Training Progress")
    
    for epoch in progress_bar:
        train_loss, train_components = train(model, train_loader, optimizer, epoch, device, args, writer)
        test_loss, test_components = test(model, test_loader, epoch, device, args, writer, save_dir)
        
        # Update progress bar with detailed loss information
        progress_bar.set_description(f"Epoch {epoch}/{args.epochs}")
        progress_info = {
            "Train": f"{train_loss:.4f}",
            "Test": f"{test_loss:.4f}",
            "Best": f"{best_loss:.4f}"
        }
        
        # Add KL losses to the progress bar
        if "kl_w" in train_components:
            progress_info["KL_w"] = f"{train_components['kl_w']:.4f}"
        if "kl_x" in train_components:
            progress_info["KL_x"] = f"{train_components['kl_x']:.4f}"
        if "recon" in train_components:
            progress_info["Recon"] = f"{train_components['recon']:.4f}"
            
        progress_bar.set_postfix(progress_info)
        
        # Save model periodically
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), f'models/gmvae_{args.dataset}_K{args.K}_epoch_{epoch}.pt')
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), f'models/gmvae_{args.dataset}_K{args.K}_best.pt')
            
            # Update progress bar with new best notification
            progress_info["Best"] = f"{best_loss:.4f} (New Best!)" 
            progress_bar.set_postfix(progress_info)
    
    # Save final model
    torch.save(model.state_dict(), f'models/gmvae_{args.dataset}_K{args.K}_final.pt')
    
    # Close tensorboard writer
    writer.close()

if __name__ == '__main__':
    main() 