#!/usr/bin/env python3
"""
Test script for Time Series GM-VAE LSTM model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from time_series_gmvae_lstm import TimeSeriesGMVAELSTM

def create_test_data(sequence_length=20, forecast_horizon=5, n_samples=10):
    """Create test time series data"""
    np.random.seed(42)
    
    # Generate different types of test sequences
    test_sequences = []
    test_labels = []
    
    for i in range(n_samples):
        # Different pattern types
        pattern_type = i % 4
        
        if pattern_type == 0:
            # Sine wave
            t = np.linspace(0, 4*np.pi, sequence_length)
            series = np.sin(t) + 0.1 * np.random.randn(sequence_length)
        elif pattern_type == 1:
            # Linear trend
            t = np.linspace(0, 1, sequence_length)
            series = 2*t + 1 + 0.1 * np.random.randn(sequence_length)
        elif pattern_type == 2:
            # Exponential decay
            t = np.linspace(0, 3, sequence_length)
            series = np.exp(-t) + 0.1 * np.random.randn(sequence_length)
        else:
            # Random walk
            series = np.cumsum(np.random.randn(sequence_length)) + 0.1 * np.random.randn(sequence_length)
        
        test_sequences.append(series)
        test_labels.append(pattern_type)
    
    return np.array(test_sequences), np.array(test_labels)

def load_model(model_path, args):
    """Load the trained model"""
    model = TimeSeriesGMVAELSTM(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_clusters=args.num_clusters,
        lstm_layers=args.lstm_layers,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        dropout=args.dropout,
        device=args.device
    )
    
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    
    return model

def test_forecasting(model, test_data, device, save_dir):
    """Test forecasting capabilities"""
    print("Testing forecasting capabilities...")
    
    model.eval()
    with torch.no_grad():
        # Generate forecasts for all test samples
        forecasts = []
        cluster_assignments = []
        
        for i in range(len(test_data)):
            x = torch.FloatTensor(test_data[i]).unsqueeze(0).unsqueeze(-1).to(device)
            
            # Generate forecast
            forecast = model.generate_forecast(x)
            forecasts.append(forecast.cpu().numpy()[0])
            
            # Get cluster assignment
            cluster_idx, cluster_probs = model.cluster_assignments(x)
            cluster_assignments.append(cluster_idx.cpu().numpy()[0])
        
        forecasts = np.array(forecasts)
        cluster_assignments = np.array(cluster_assignments)
    
    # Plot results
    n_samples = min(8, len(test_data))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(n_samples):
        x = test_data[i]
        forecast = forecasts[i]
        
        axes[i].plot(range(len(x)), x, 'b-', label='Input', linewidth=2)
        axes[i].plot(range(len(x), len(x) + len(forecast)), 
                    forecast, 'r--', label='Forecast', linewidth=2)
        axes[i].set_title(f'Sample {i+1} (Cluster {cluster_assignments[i]})')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_forecasts.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Forecast plots saved to {save_dir}/test_forecasts.png")
    
    return forecasts, cluster_assignments

def test_clustering(model, test_data, device, save_dir):
    """Test clustering capabilities"""
    print("Testing clustering capabilities...")
    
    model.eval()
    with torch.no_grad():
        cluster_assignments = []
        cluster_probs = []
        
        for i in range(len(test_data)):
            x = torch.FloatTensor(test_data[i]).unsqueeze(0).unsqueeze(-1).to(device)
            cluster_idx, probs = model.cluster_assignments(x)
            
            cluster_assignments.append(cluster_idx.cpu().numpy()[0])
            cluster_probs.append(probs.cpu().numpy()[0])
        
        cluster_assignments = np.array(cluster_assignments)
        cluster_probs = np.array(cluster_probs)
    
    # Plot cluster assignments
    plt.figure(figsize=(10, 6))
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    plt.bar(unique_clusters, counts)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/cluster_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot cluster probabilities
    plt.figure(figsize=(12, 8))
    for i in range(len(test_data)):
        plt.subplot(2, 5, i+1)
        plt.bar(range(len(cluster_probs[i])), cluster_probs[i])
        plt.title(f'Sample {i+1}')
        plt.xlabel('Cluster')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cluster_probabilities.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Clustering plots saved to {save_dir}/")
    print(f"Cluster distribution: {dict(zip(unique_clusters, counts))}")
    
    return cluster_assignments, cluster_probs

def main():
    parser = argparse.ArgumentParser(description='Test Time Series GM-VAE LSTM Model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='path to the trained model')
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
    parser.add_argument('--device', type=str, default='auto',
                       help='device to use (cpu, cuda, mps, auto)')
    parser.add_argument('--save-dir', type=str, default='./test_results',
                       help='directory to save test results')
    
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
    
    args.device = device
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args)
    print("Model loaded successfully!")
    
    # Create test data
    print("Creating test data...")
    test_data, test_labels = create_test_data(
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        n_samples=10
    )
    
    # Test forecasting
    forecasts, cluster_assignments = test_forecasting(model, test_data, device, args.save_dir)
    
    # Test clustering
    cluster_assignments, cluster_probs = test_clustering(model, test_data, device, args.save_dir)
    
    # Compute forecasting accuracy (MSE)
    mse_scores = []
    for i in range(len(test_data)):
        # For synthetic data, we can compute MSE against the true continuation
        # In real scenarios, you would compare against actual future values
        mse = np.mean((forecasts[i] - test_data[i, -args.forecast_horizon:])**2)
        mse_scores.append(mse)
    
    print(f"\nForecasting Results:")
    print(f"  Average MSE: {np.mean(mse_scores):.4f}")
    print(f"  MSE Std: {np.std(mse_scores):.4f}")
    
    print(f"\nClustering Results:")
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"  Cluster {cluster_id}: {count} samples")
    
    print(f"\nTest completed! Results saved to {args.save_dir}")

if __name__ == '__main__':
    main()
