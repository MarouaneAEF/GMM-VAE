import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import os

# Import our model
from GM_VAE import GMVAE

def parse_args():
    parser = argparse.ArgumentParser(description='Sample from trained GM-VAE')
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to trained model checkpoint')
    parser.add_argument('--input-channels', type=int, default=3,
                        help='number of input channels (3 for CIFAR-10, 1 for MNIST)')
    parser.add_argument('--K', type=int, default=10,
                        help='number of mixture components (default: 10)')
    parser.add_argument('--hidden-size', type=int, default=500,
                        help='size of hidden layer (default: 500)')
    parser.add_argument('--x-size', type=int, default=200,
                        help='size of latent variable x (default: 200)')
    parser.add_argument('--w-size', type=int, default=150,
                        help='size of latent variable w (default: 150)')
    parser.add_argument('--samples-per-cluster', type=int, default=10,
                        help='number of samples to generate per cluster (default: 10)')
    parser.add_argument('--output-dir', type=str, default='samples',
                        help='directory to save generated samples (default: samples)')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='size of the grid for interpolation sampling (default: 10)')
    parser.add_argument('--latent-dim', type=int, default=2,
                        help='which 2 dimensions of the latent space to use for 2D grid sampling (default: 0,1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA sampling')
    
    return parser.parse_args()

def load_model(args):
    """Load a trained GM-VAE model."""
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    model = GMVAE(
        input_channels=args.input_channels,
        img_size=32,  # Standard size for both MNIST and CIFAR-10
        hidden_size=args.hidden_size,
        x_size=args.x_size,
        w_size=args.w_size,
        K=args.K
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    return model, device

def sample_from_cluster(model, cluster_idx, num_samples, device):
    """Generate samples from a specific cluster."""
    with torch.no_grad():
        # Create a one-hot encoding for the cluster
        z = torch.zeros(num_samples, model.K, device=device)
        z[:, cluster_idx] = 1.0
        
        # Sample w from standard normal
        w = torch.randn(num_samples, model.w_size, device=device)
        
        # Get the prior distribution parameters for the selected cluster
        h = model.prior_network.prior_stack(w)
        mu_px = model.prior_network.mu_px[cluster_idx](h)
        logvar_px = model.prior_network.logvar_px[cluster_idx](h)
        
        # Sample x from the prior distribution
        std_px = torch.exp(0.5 * logvar_px)
        x = mu_px + std_px * torch.randn_like(std_px)
        
        # Decode x to generate samples
        samples = model.decoder(x)
        
        return samples

def interpolate_latent_space(model, grid_size, latent_dim1=0, latent_dim2=1, device='cuda'):
    """
    Generate samples by interpolating in the latent space along two dimensions.
    """
    # Create a grid in the latent space
    linspace = torch.linspace(-3, 3, grid_size)
    grid_x, grid_y = torch.meshgrid(linspace, linspace)
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
    
    with torch.no_grad():
        # Create a batch of latent vectors with zeros except for the chosen dimensions
        z_samples = torch.zeros(grid_size * grid_size, model.x_size, device=device)
        z_samples[:, latent_dim1] = grid_points[:, 0]
        z_samples[:, latent_dim2] = grid_points[:, 1]
        
        # Decode the latent vectors to generate samples
        samples = model.decoder(z_samples)
        
        return samples.reshape(grid_size, grid_size, model.input_channels, 32, 32)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, device = load_model(args)
    
    # Generate samples from each cluster
    print("Generating samples from each cluster...")
    all_samples = []
    for k in range(model.K):
        samples = sample_from_cluster(model, k, args.samples_per_cluster, device)
        all_samples.append(samples)
        
        # Save samples from this cluster
        save_image(samples, f"{args.output_dir}/cluster_{k}_samples.png", nrow=int(np.sqrt(args.samples_per_cluster)))
    
    # Combine samples from all clusters into a single grid
    all_samples = torch.cat(all_samples, dim=0)
    save_image(all_samples, f"{args.output_dir}/all_clusters_samples.png", nrow=args.samples_per_cluster)
    
    # Generate latent space interpolations
    print("Generating latent space interpolations...")
    interpolated_samples = interpolate_latent_space(model, args.grid_size, device=device)
    
    # Reshape for visualization and save
    grid_img = make_grid(interpolated_samples.view(-1, model.input_channels, 32, 32), nrow=args.grid_size)
    save_image(grid_img, f"{args.output_dir}/latent_space_interpolation.png")
    
    # Create a visualization with the grid
    plt.figure(figsize=(10, 10))
    if model.input_channels == 1:
        plt.imshow(grid_img[0].cpu().numpy(), cmap='gray')
    else:
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/latent_space_grid.png", dpi=300)
    
    print(f"Samples saved to {args.output_dir}")

if __name__ == '__main__':
    main() 