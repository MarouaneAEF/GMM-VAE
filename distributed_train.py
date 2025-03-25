import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.distributed
from datetime import datetime, timedelta
from tqdm import tqdm

# Import our model and data utilities
from GM_VAE import GMVAE, GMVAELoss
import dataloader as dl

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training of GM-VAE')
    parser.add_argument('--dataset', type=str, default='custom', 
                        choices=['mnist', 'cifar10', 'custom'],
                        help='dataset to use (default: custom)')
    parser.add_argument('--data-dir', type=str, default='../augmented_images',
                        help='directory containing the custom images (default: ../augmented_images)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size per worker (default: 32)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='maximum number of images to use for training (default: all images)')
    parser.add_argument('--random-sample', action='store_true',
                        help='randomly sample max-images instead of taking first ones')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='device type to use (mps for Apple Silicon)')
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
    parser.add_argument('--world-size', type=int, default=10,
                        help='number of distributed processes (default: 10 for 10 GPUs)')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:23456',
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', type=str, default='gloo',
                        help='distributed backend')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank for distributed training')
    parser.add_argument('--kl-weight', type=float, default=1.0,
                        help='Weight for KL divergence term (default: 1.0)')
    parser.add_argument('--kl-anneal', action='store_true',
                        help='Enable KL annealing for first 10 epochs')
    parser.add_argument('--recon-weight', type=float, default=1.0,
                        help='Weight for reconstruction loss (default: 1.0)')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0)')
    
    return parser.parse_args()

def train(model, train_loader, optimizer, epoch, device, rank, args, writer=None):
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
    
    # Log the current weights (only for rank 0)
    if rank == 0:
        print(f"Epoch {epoch}: KL weight = {kl_weight:.4f}, Reconstruction weight = {recon_weight:.4f}")
    
    # Only show progress bar for rank 0
    if rank == 0:
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
    
    for batch_idx, (data, target) in enumerate(train_loader):
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
            mu_px, logvar_px, x_sample, model.module.x_size, model.module.K,
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
        
        # Update progress bar on rank 0
        if rank == 0 and isinstance(train_loader, tqdm):
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
                
            train_loader.set_postfix(postfix_dict)
        
        # Log progress
        if batch_idx % args.log_interval == 0 and rank == 0:
            # We no longer need to print here as tqdm shows progress
            
            # Add loss components to tensorboard
            if writer:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item() / len(data), step)
                for name, value in components.items():
                    writer.add_scalar(f'train/{name}', value / len(data), step)
    
    # Reduce losses from all processes
    train_loss = torch.tensor(train_loss).to(device)
    dist.all_reduce(train_loss)
    train_loss = train_loss / dist.get_world_size()
    
    # Reduce component losses from all processes
    for key in component_losses:
        comp_loss = torch.tensor(component_losses[key]).to(device)
        dist.all_reduce(comp_loss)
        component_losses[key] = comp_loss.item() / dist.get_world_size()
    
    # Calculate average epoch loss
    avg_loss = train_loss.item() / len(train_loader.dataset)
    avg_component_losses = {k: v / len(train_loader.dataset) for k, v in component_losses.items()}
    
    # Only print from rank 0
    if rank == 0:
        loss_str = f'Epoch: {epoch} Train loss: {avg_loss:.4f} | '
        for key, value in avg_component_losses.items():
            loss_str += f"{key}: {value:.4f} "
        print(loss_str)
    
    # Return average loss and component losses
    return avg_loss, avg_component_losses

def test(model, test_loader, epoch, device, rank, args, writer=None, save_dir=None):
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
    
    # Create directory for results if it doesn't exist and is main process
    if rank == 0 and save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Only show progress bar for rank 0
    if rank == 0:
        test_loader = tqdm(test_loader, desc=f"Testing Epoch {epoch}", leave=False)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Move data to device
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass
            mu_x, logvar_x, mu_px, logvar_px, qz, recon_x, mu_w, logvar_w, x_sample = model(data)
            
            # Compute loss with weighting
            loss, components = GMVAELoss.compute_loss(
                recon_x, data, mu_w, logvar_w, qz, mu_x, logvar_x, 
                mu_px, logvar_px, x_sample, model.module.x_size, model.module.K,
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
                
            # Update progress bar on rank 0
            if rank == 0 and isinstance(test_loader, tqdm):
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
                    
                test_loader.set_postfix(postfix_dict)
            
            # Save reconstructions for visualization (only for rank 0)
            if batch_idx == 0 and rank == 0 and save_dir:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_x[:n]])
                comparison_cpu = comparison.cpu()
                
                save_image(comparison_cpu,
                          f'{save_dir}/reconstruction_epoch_{epoch}.png', nrow=n)
                
                # Add images to tensorboard
                if writer:
                    writer.add_images('test/reconstruction', comparison_cpu, epoch, dataformats='NCHW')
                
                # Visualize cluster assignments
                if epoch % args.save_interval == 0:
                    # Get cluster assignments
                    cluster_assignments = torch.argmax(qz, dim=1)
                    
                    # Create visualization of cluster assignments
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(mu_x[:, 0].cpu().numpy(), mu_x[:, 1].cpu().numpy(), 
                                       c=cluster_assignments.cpu().numpy(), cmap='tab10', 
                                       alpha=0.6, s=10)
                    ax.set_title(f'Latent Space Visualization (Epoch {epoch})')
                    ax.set_xlabel('Latent Dimension 1')
                    ax.set_ylabel('Latent Dimension 2')
                    fig.colorbar(scatter, label='Cluster Assignment')
                    plt.tight_layout()
                    
                    # Save figure
                    plt.savefig(f'{save_dir}/clusters_epoch_{epoch}.png')
                    plt.close(fig)
                    
                    # Log cluster distribution
                    if writer:
                        for k in range(model.module.K):
                            count = (cluster_assignments == k).sum().item()
                            writer.add_scalar(f'clusters/cluster_{k}', count, epoch)
    
    # Reduce losses from all processes
    test_loss = torch.tensor(test_loss).to(device)
    dist.all_reduce(test_loss)
    test_loss = test_loss / dist.get_world_size()
    
    # Reduce component losses from all processes
    for key in component_losses:
        comp_loss = torch.tensor(component_losses[key]).to(device)
        dist.all_reduce(comp_loss)
        component_losses[key] = comp_loss.item() / dist.get_world_size()
    
    # Calculate average epoch loss and component losses
    test_loss = test_loss.item() / len(test_loader.dataset)
    avg_component_losses = {k: v / len(test_loader.dataset) for k, v in component_losses.items()}
    
    if rank == 0:
        loss_str = f'====> Test set loss: {test_loss:.4f} | '
        for key, value in avg_component_losses.items():
            loss_str += f"{key}: {value:.4f} "
        print(loss_str)
        
        if writer:
            writer.add_scalar('test/loss', test_loss, epoch)
            # Log component losses to tensorboard
            for key, value in avg_component_losses.items():
                writer.add_scalar(f'test/{key}', value, epoch)
    
    # Return average loss and component losses
    return test_loss, avg_component_losses

def setup(rank, world_size, args):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    
    # For Apple Silicon, Gloo backend is the only option that works reliably
    backend = 'gloo'
    
    # Initialize process group with timeout
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
        print(f"Process {rank}: Initialized process group")
    except Exception as e:
        print(f"Process {rank}: Failed to initialize process group: {e}")
        raise

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size, args):
    # Setup the distributed environment
    setup(rank, world_size, args)
    
    print(f"Process {rank}: Initialized process group")
    
    # Set the device
    device = args.device
    if device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{rank}'
        torch.cuda.set_device(rank)
        print(f"Using CUDA GPU {rank}")
    elif device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS backend)")
    else:
        device = 'cpu'
        print(f"Process {rank}: Using CPU")
    
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    if rank == 0:
        print(f"Using device: {device}")
        
        # Create directories for saving results and models
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Initialize tensorboard writer (only for rank 0)
        writer = SummaryWriter(log_dir=f'runs/gmvae_{args.dataset}_K{args.K}_distributed')
    else:
        writer = None
    
    # Load dataset
    if args.dataset == 'mnist':
        dataset_fn = dl.mnistloader
        input_channels = 1
    elif args.dataset == 'cifar10':
        dataset_fn = dl.cifar10loader
        input_channels = 3
    else:  # custom dataset
        if rank == 0:
            print(f"Loading custom dataset from {args.data_dir}")
            print(f"Maximum images to use: {args.max_images if args.max_images else 'All available'}")
            print(f"Random sampling: {'Enabled' if args.random_sample else 'Disabled'}")
            
            # Get image dimensions from the dataloader (just for info)
            _, _, (img_height, img_width) = dl.custom_dataloader(
                args.data_dir, 
                args.batch_size,
                preserve_size=True,
                max_images=args.max_images,
                random_sample=args.random_sample
            )
            print(f"Dataset image size: {img_height}×{img_width}")
        
        # Ensure all processes know the image dimensions
        input_channels = 3  # Assuming RGB images
    
    # Setup distributed sampler for training data
    if args.dataset == 'custom':
        # Create custom dataloader with DistributedSampler
        train_dataset = dl.CustomHighResImageDataset(args.data_dir, transform=transforms.ToTensor(), 
                                                   split='train', max_images=args.max_images, 
                                                   random_sample=args.random_sample)
        test_dataset = dl.CustomHighResImageDataset(args.data_dir, transform=transforms.ToTensor(), 
                                                  split='test', max_images=args.max_images,
                                                  random_sample=args.random_sample)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        
        # For testing, we don't need distributed sampling
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # For standard datasets, we use the existing loaders
        train_loader, test_loader = dataset_fn(args.batch_size)
    
    # Initialize model
    if args.dataset == 'custom':
        model = GMVAE(
            input_channels=input_channels,
            img_height=None,  # Will be set dynamically
            img_width=None,   # Will be set dynamically
            hidden_size=args.hidden_size,
            x_size=args.x_size,
            w_size=args.w_size,
            K=args.K
        ).to(device)
    else:
        model = GMVAE(
            input_channels=input_channels,
            img_height=8,    # Standard size for MNIST/CIFAR
            img_width=8,     # Standard size for MNIST/CIFAR
            hidden_size=args.hidden_size,
            x_size=args.x_size,
            w_size=args.w_size,
            K=args.K
        ).to(device)
    
    # Wrap model with DDP for all device types
    model = DDP(model, device_ids=None if device == 'cpu' or device == 'mps' else [rank])
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory for this run
    save_dir = None
    if rank == 0:
        save_dir = f'results/gmvae_{args.dataset}_K{args.K}_distributed'
        os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    if rank == 0:
        print(f"Starting training for {args.epochs} epochs...")
        epoch_iterator = tqdm(range(1, args.epochs + 1), desc="Training Progress")
    else:
        epoch_iterator = range(1, args.epochs + 1)
        
    for epoch in epoch_iterator:
        # Set epoch for distributed sampler
        if args.dataset == 'custom':
            train_sampler.set_epoch(epoch)
            
        train_loss, train_components = train(model, train_loader, optimizer, epoch, device, rank, args, writer)
        test_loss, test_components = test(model, test_loader, epoch, device, rank, args, writer, save_dir)
        
        # Update progress bar description with loss info (only for rank 0)
        if rank == 0 and isinstance(epoch_iterator, tqdm):
            epoch_iterator.set_description(f"Epoch {epoch}/{args.epochs}")
            
            # Create detailed progress info
            progress_info = {
                "Train": f"{train_loss:.4f}",
                "Test": f"{test_loss:.4f}",
                "Best": f"{best_loss:.4f}"
            }
            
            # Add KL losses to progress bar
            if "kl_w" in train_components:
                progress_info["KL_w"] = f"{train_components['kl_w']:.4f}"
            if "kl_x" in train_components:
                progress_info["KL_x"] = f"{train_components['kl_x']:.4f}"
            if "recon" in train_components:
                progress_info["Recon"] = f"{train_components['recon']:.4f}"
                
            epoch_iterator.set_postfix(progress_info)
        
        # Save model periodically (only rank 0)
        if epoch % args.save_interval == 0 and rank == 0:
            torch.save(model.module.state_dict(), f'models/gmvae_{args.dataset}_K{args.K}_distributed_epoch_{epoch}.pt')
        
        # Save best model (only rank 0)
        if test_loss < best_loss and rank == 0:
            best_loss = test_loss
            torch.save(model.module.state_dict(), f'models/gmvae_{args.dataset}_K{args.K}_distributed_best.pt')
            
            # Update progress bar with new best notification
            if isinstance(epoch_iterator, tqdm):
                progress_info["Best"] = f"{best_loss:.4f} (New Best!)"
                epoch_iterator.set_postfix(progress_info)
    
    # Save final model (only rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), f'models/gmvae_{args.dataset}_K{args.K}_distributed_final.pt')
        
        # Close tensorboard writer
        if writer:
            writer.close()
    
    # Cleanup process group
    cleanup()

def main():
    args = parse_args()
    
    world_size = args.world_size
    
    # MPS backend doesn't support true multi-GPU parallelism in the same way as CUDA
    # For Apple Silicon, we use a process-based approach but with shared memory
    if args.device == 'mps':
        print(f"Using {world_size} Apple Silicon GPUs via process-based data parallelism")
        
        # Set required environment variables for MPS
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # Use torch.multiprocessing to spawn multiple processes
    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main() 