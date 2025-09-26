#!/usr/bin/env python3
"""
Demo script for Time Series GM-VAE LSTM model
Shows how to use the model for forecasting and clustering
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from time_series_gmvae_lstm import TimeSeriesGMVAELSTM

def create_demo_data():
    """Create demo time series data"""
    np.random.seed(42)
    
    # Create different types of time series
    patterns = []
    
    # Sine wave
    t = np.linspace(0, 4*np.pi, 20)
    sine_wave = np.sin(t) + 0.1 * np.random.randn(20)
    patterns.append(('Sine Wave', sine_wave))
    
    # Linear trend
    t = np.linspace(0, 1, 20)
    linear = 2*t + 1 + 0.1 * np.random.randn(20)
    patterns.append(('Linear Trend', linear))
    
    # Exponential decay
    t = np.linspace(0, 3, 20)
    exp_decay = np.exp(-t) + 0.1 * np.random.randn(20)
    patterns.append(('Exponential Decay', exp_decay))
    
    # Random walk
    random_walk = np.cumsum(np.random.randn(20)) + 0.1 * np.random.randn(20)
    patterns.append(('Random Walk', random_walk))
    
    return patterns

def demo_forecasting():
    """Demonstrate forecasting capabilities"""
    print("=== Time Series Forecasting Demo ===\n")
    
    # Create model (smaller for demo)
    model = TimeSeriesGMVAELSTM(
        input_dim=1,
        hidden_dim=32,
        latent_dim=8,
        num_clusters=4,
        sequence_length=20,
        forecast_horizon=5,
        device='cpu'
    )
    
    # Create demo data
    patterns = create_demo_data()
    
    print("Generating forecasts for different time series patterns...")
    
    # Generate forecasts
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (name, series) in enumerate(patterns):
        # Prepare input
        x = torch.FloatTensor(series).unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
        
        # Generate forecast
        with torch.no_grad():
            forecast = model.generate_forecast(x)
            forecast_np = forecast.numpy()[0]
        
        # Plot
        axes[i].plot(range(len(series)), series, 'b-', label='Input', linewidth=2)
        axes[i].plot(range(len(series), len(series) + len(forecast_np)), 
                    forecast_np, 'r--', label='Forecast', linewidth=2)
        axes[i].set_title(f'{name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        print(f"  {name}: Generated {len(forecast_np)}-step forecast")
    
    plt.tight_layout()
    plt.savefig('demo_forecasts.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Forecast plots saved to 'demo_forecasts.png'\n")

def demo_clustering():
    """Demonstrate clustering capabilities"""
    print("=== Time Series Clustering Demo ===\n")
    
    # Create model
    model = TimeSeriesGMVAELSTM(
        input_dim=1,
        hidden_dim=32,
        latent_dim=8,
        num_clusters=4,
        sequence_length=20,
        forecast_horizon=5,
        device='cpu'
    )
    
    # Create multiple samples of each pattern
    patterns = []
    labels = []
    
    for pattern_type in range(4):
        for _ in range(5):  # 5 samples per pattern
            if pattern_type == 0:
                # Sine wave
                t = np.linspace(0, 4*np.pi, 20)
                series = np.sin(t) + 0.1 * np.random.randn(20)
            elif pattern_type == 1:
                # Linear trend
                t = np.linspace(0, 1, 20)
                series = 2*t + 1 + 0.1 * np.random.randn(20)
            elif pattern_type == 2:
                # Exponential decay
                t = np.linspace(0, 3, 20)
                series = np.exp(-t) + 0.1 * np.random.randn(20)
            else:
                # Random walk
                series = np.cumsum(np.random.randn(20)) + 0.1 * np.random.randn(20)
            
            patterns.append(series)
            labels.append(pattern_type)
    
    patterns = np.array(patterns)
    labels = np.array(labels)
    
    print("Analyzing clustering of time series patterns...")
    
    # Get cluster assignments
    cluster_assignments = []
    cluster_probs = []
    
    for series in patterns:
        x = torch.FloatTensor(series).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            cluster_idx, probs = model.cluster_assignments(x)
            cluster_assignments.append(cluster_idx.item())
            cluster_probs.append(probs.numpy()[0])
    
    cluster_assignments = np.array(cluster_assignments)
    cluster_probs = np.array(cluster_probs)
    
    # Print results
    print(f"Total samples: {len(patterns)}")
    print(f"Number of clusters: {len(np.unique(cluster_assignments))}")
    
    for cluster_id in np.unique(cluster_assignments):
        cluster_samples = np.where(cluster_assignments == cluster_id)[0]
        cluster_labels = labels[cluster_samples]
        print(f"  Cluster {cluster_id}: {len(cluster_samples)} samples")
        print(f"    Pattern distribution: {np.bincount(cluster_labels)}")
    
    # Plot clustering results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, cluster_id in enumerate(np.unique(cluster_assignments)):
        cluster_samples = np.where(cluster_assignments == cluster_id)[0]
        
        for sample_idx in cluster_samples:
            series = patterns[sample_idx]
            label = labels[sample_idx]
            color = ['blue', 'red', 'green', 'orange'][label]
            axes[i].plot(series, color=color, alpha=0.7, linewidth=1)
        
        axes[i].set_title(f'Cluster {cluster_id} ({len(cluster_samples)} samples)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_clustering.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Clustering plots saved to 'demo_clustering.png'\n")

def main():
    """Run the demo"""
    print("Time Series GM-VAE LSTM Demo")
    print("=" * 40)
    
    try:
        # Demo forecasting
        demo_forecasting()
        
        # Demo clustering
        demo_clustering()
        
        print("Demo completed successfully!")
        print("\nThis demo shows:")
        print("1. How to generate forecasts for different time series patterns")
        print("2. How the model clusters similar time series together")
        print("3. The probabilistic nature of the model's predictions")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Make sure you have matplotlib installed: pip install matplotlib")

if __name__ == '__main__':
    main()
