import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from PIL import Image
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
    parser.add_argument('--morph-steps', type=int, default=30,
                        help='number of interpolation steps between cluster pairs (default: 30)')
    parser.add_argument('--morph-fps', type=int, default=15,
                        help='frames per second for morph GIF (default: 15)')
    parser.add_argument('--walk-steps', type=int, default=120,
                        help='number of steps for latent walk GIF (default: 120)')

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
        img_height=32,
        img_width=32,
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

def slerp(t, v0, v1, dot_threshold=0.9995):
    """
    Spherical linear interpolation between two latent vectors.
    Produces more natural transitions than linear interpolation in hyperspherical spaces.

    Args:
        t: interpolation factor in [0, 1]
        v0, v1: start and end vectors (shape: [batch, dim])
        dot_threshold: threshold below which to fall back to linear interpolation
    """
    v0_norm = v0 / (v0.norm(dim=-1, keepdim=True) + 1e-8)
    v1_norm = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)

    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)

    # Fall back to linear interpolation when vectors are nearly parallel
    linear = v0 + t * (v1 - v0)
    omega = torch.acos(dot.abs())
    sin_omega = torch.sin(omega)

    slerp_result = (torch.sin((1 - t) * omega) / (sin_omega + 1e-8)) * v0 + \
                   (torch.sin(t * omega) / (sin_omega + 1e-8)) * v1

    # Use linear where sin_omega is too small
    mask = (sin_omega.squeeze(-1) < 1e-6).unsqueeze(-1)
    return torch.where(mask, linear, slerp_result)


def tensor_to_pil(tensor):
    """Convert a [C, H, W] tensor in [0,1] to a PIL Image."""
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def generate_cluster_morph_gif(model, device, output_path, steps=30, fps=15):
    """
    Generate a smooth morphing GIF that cycles through all cluster pairs.
    Uses SLERP in both w-space (style) and x-space (content) for realistic transitions.

    For each consecutive cluster pair (0→1, 1→2, ..., K-1→0):
      - Sample a representative w from each cluster's prior
      - SLERP between the two w's to get intermediate styles
      - Decode each intermediate point
    """
    model.eval()
    K = model.K
    duration_ms = max(20, 1000 // fps)
    frames = []

    with torch.no_grad():
        # Sample one w per cluster using a fixed seed for reproducibility
        torch.manual_seed(42)
        ws = torch.randn(K, model.w_size, device=device)

        # Get prior means for each cluster (the "canonical" x for each cluster)
        xs = []
        for k in range(K):
            h = model.prior_network.prior_stack(ws[k:k+1])
            mu_x_k = model.prior_network.mu_px[k](h)  # [1, x_size]
            xs.append(mu_x_k)
        xs = torch.cat(xs, dim=0)  # [K, x_size]

        # Cycle through cluster pairs with smooth SLERP interpolation
        for k in range(K):
            k_next = (k + 1) % K
            for i in range(steps):
                t = i / steps

                # Ease in-out with smoothstep so motion decelerates at endpoints
                t_smooth = t * t * (3 - 2 * t)

                # Interpolate x (content latent)
                x_interp = slerp(t_smooth, xs[k:k+1], xs[k_next:k_next+1])

                # Decode
                img_tensor = model.decoder(x_interp)  # [1, C, H, W]
                frames.append(tensor_to_pil(img_tensor[0]))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Cluster morph GIF saved to {output_path} ({len(frames)} frames, {fps} fps)")


def generate_latent_walk_gif(model, device, output_path, steps=120, fps=20):
    """
    Generate a smooth random walk animation through the latent space.
    Uses random waypoints connected by SLERP curves so the walk feels
    continuous and organic rather than jumping between discrete samples.
    """
    model.eval()
    n_waypoints = max(4, steps // 20)
    duration_ms = max(20, 1000 // fps)
    frames = []

    with torch.no_grad():
        torch.manual_seed(0)
        # Sample random waypoints in x-space using the unit hypersphere surface
        waypoints = torch.randn(n_waypoints + 1, model.x_size, device=device)
        waypoints = waypoints / (waypoints.norm(dim=-1, keepdim=True) + 1e-8)
        # Scale to a reasonable latent magnitude (std ≈ 1)
        waypoints = waypoints * (model.x_size ** 0.5)

        steps_per_segment = steps // n_waypoints

        for seg in range(n_waypoints):
            p0 = waypoints[seg:seg+1]
            p1 = waypoints[seg+1:seg+2]

            for i in range(steps_per_segment):
                t = i / steps_per_segment
                # Smoothstep easing
                t_smooth = t * t * (3 - 2 * t)

                x_interp = slerp(t_smooth, p0, p1)
                img_tensor = model.decoder(x_interp)
                frames.append(tensor_to_pil(img_tensor[0]))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Latent walk GIF saved to {output_path} ({len(frames)} frames, {fps} fps)")


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

    # --- Credible animations ---
    print("Generating cluster morph animation (SLERP)...")
    generate_cluster_morph_gif(
        model, device,
        output_path=f"{args.output_dir}/cluster_morph.gif",
        steps=args.morph_steps,
        fps=args.morph_fps,
    )

    print("Generating latent walk animation...")
    generate_latent_walk_gif(
        model, device,
        output_path=f"{args.output_dir}/latent_walk.gif",
        steps=args.walk_steps,
        fps=args.morph_fps,
    )

    print(f"Samples saved to {args.output_dir}")

if __name__ == '__main__':
    main() 