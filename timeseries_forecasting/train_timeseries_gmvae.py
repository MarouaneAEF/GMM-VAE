#!/usr/bin/env python3
"""
Training script for Time Series GM-VAE LSTM model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import json
from datetime import datetime

from time_series_gmvae_lstm import TimeSeriesGMVAELSTM

def create_synthetic_data(n_samples=1000, sequence_length=20, input_dim=1, noise_level=0.1):
    """Create synthetic time series data for testing"""
    np.random.seed(42)
    
    # Generate different types of time series
    data = []
    labels = []
    
    for i in range(n_samples):
        # Randomly choose a pattern type
        pattern_type = np.random.choice([0, 1, 2, 3])  # 4 different patterns
        
        if pattern_type == 0:
            # Sine wave
            t = np.linspace(0, 4*np.pi, sequence_length)
            series = np.sin(t) + noise_level * np.random.randn(sequence_length)
        elif pattern_type == 1:
            # Linear trend
            t = np.linspace(0, 1, sequence_length)
            series = 2*t + 1 + noise_level * np.random.randn(sequence_length)
        elif pattern_type == 2:
            # Exponential decay
            t = np.linspace(0, 3, sequence_length)
            series = np.exp(-t) + noise_level * np.random.randn(sequence_length)
        else:
            # Random walk
            series = np.cumsum(np.random.randn(sequence_length)) + noise_level * np.random.randn(sequence_length)
        
        data.append(series)
        labels.append(pattern_type)
    
    return np.array(data), np.array(labels)

def prepare_sequences(data, sequence_length, forecast_horizon):
    """Prepare sequences for training"""
    X, y = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        # Input sequence - take the first sequence_length points
        x_seq = data[i, :sequence_length]
        # Target sequence - take the next forecast_horizon points
        y_seq = data[i, sequence_length:sequence_length+forecast_horizon]
        
        X.append(x_seq)
        y.append(y_seq)
    
    return np.array(X), np.array(y)

def train_epoch(model, dataloader, optimizer, device, kl_weight=1.0, recon_weight=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    recon_loss = 0
    kl_loss = 0
    gmm_loss = 0
    
    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Training")):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x, return_components=True)
        
        # Compute loss
        loss_dict = model.compute_loss(
            x, outputs['forecast'], outputs['mu'], outputs['logvar'],
            outputs['pi'], outputs['mu_gmm'], outputs['var_gmm'],
            kl_weight=kl_weight, recon_weight=recon_weight
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict['total_loss'].item()
        recon_loss += loss_dict['recon_loss'].item()
        kl_loss += loss_dict['kl_loss'].item()
        gmm_loss += loss_dict['gmm_loss'].item()
    
    return {
        'total_loss': total_loss / len(dataloader),
        'recon_loss': recon_loss / len(dataloader),
        'kl_loss': kl_loss / len(dataloader),
        'gmm_loss': gmm_loss / len(dataloader)
    }

def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x, return_components=True)
            
            # Compute loss
            loss_dict = model.compute_loss(
                x, outputs['forecast'], outputs['mu'], outputs['logvar'],
                outputs['pi'], outputs['mu_gmm'], outputs['var_gmm']
            )
            
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
    
    return {
        'total_loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader)
    }

def plot_results(model, test_data, device, save_dir, epoch):
    """Plot forecasting results"""
    model.eval()
    
    with torch.no_grad():
        # Take a few samples for visualization
        n_samples = min(4, len(test_data))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(n_samples):
            x = test_data[i:i+1].to(device)
            forecast = model.generate_forecast(x)
            
            # Plot original sequence
            x_np = x.cpu().numpy()[0]
            forecast_np = forecast.cpu().numpy()[0]
            
            axes[i].plot(range(len(x_np)), x_np, 'b-', label='Input', linewidth=2)
            axes[i].plot(range(len(x_np), len(x_np) + len(forecast_np)), 
                        forecast_np, 'r--', label='Forecast', linewidth=2)
            axes[i].set_title(f'Sample {i+1}')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/forecast_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Time Series GM-VAE LSTM')
    parser.add_argument('--data-dir', type=str, default='./timeseries_data', 
                       help='directory to save/load data')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, 
                       help='LSTM hidden dimension')
    parser.add_argument('--latent-dim', type=int, default=16, 
                       help='latent space dimension')
    parser.add_argument('--num-clusters', type=int, default=4, 
                       help='number of clusters')
    parser.add_argument('--sequence-length', type=int, default=20, 
                       help='input sequence length')
    parser.add_argument('--forecast-horizon', type=int, default=5, 
                       help='forecast horizon')
    parser.add_argument('--lstm-layers', type=int, default=2, 
                       help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='dropout rate')
    parser.add_argument('--kl-weight', type=float, default=0.1, 
                       help='KL divergence weight')
    parser.add_argument('--recon-weight', type=float, default=1.0, 
                       help='reconstruction loss weight')
    parser.add_argument('--device', type=str, default='auto', 
                       help='device to use (cpu, cuda, mps, auto)')
    parser.add_argument('--save-dir', type=str, default='./timeseries_results', 
                       help='directory to save results')
    parser.add_argument('--save-interval', type=int, default=10, 
                       help='save model every N epochs')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic time series data...")
    data, labels = create_synthetic_data(
        n_samples=2000, 
        sequence_length=args.sequence_length + args.forecast_horizon,
        input_dim=1
    )
    
    # Prepare sequences
    X, y = prepare_sequences(data, args.sequence_length, args.forecast_horizon)
    
    # Split data
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # Add feature dimension
    y_train = torch.FloatTensor(y_train).unsqueeze(-1)
    X_val = torch.FloatTensor(X_val).unsqueeze(-1)
    y_val = torch.FloatTensor(y_val).unsqueeze(-1)
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1)
    
    # Debug: Print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Expected input shape: (batch_size, {args.sequence_length}, 1)")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    model = TimeSeriesGMVAELSTM(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_clusters=args.num_clusters,
        lstm_layers=args.lstm_layers,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        dropout=args.dropout,
        device=device
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            kl_weight=args.kl_weight, recon_weight=args.recon_weight
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Store losses
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['total_loss'])
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Recon: {train_metrics['recon_loss']:.4f}, "
              f"KL: {train_metrics['kl_loss']:.4f}, "
              f"GMM: {train_metrics['gmm_loss']:.4f})")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(model.state_dict(), f'{args.save_dir}/best_model.pt')
            print("New best model saved!")
        
        # Save model periodically
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f'{args.save_dir}/model_epoch_{epoch+1}.pt')
            plot_results(model, X_test, device, args.save_dir, epoch+1)
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_metrics['total_loss']:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{args.save_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save final results
    results = {
        'args': vars(args),
        'final_test_loss': test_metrics['total_loss'],
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    with open(f'{args.save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {args.save_dir}")

if __name__ == '__main__':
    main()
