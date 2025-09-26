# Time Series Forecasting with GM-VAE LSTM

This module extends the GM-VAE architecture to time series forecasting by combining it with LSTM networks. The model can learn temporal patterns, cluster different types of time series, and generate accurate forecasts.

## Architecture

The Time Series GM-VAE LSTM model consists of three main components:

1. **LSTM Encoder**: Captures temporal patterns in the input sequence
2. **GM-VAE Latent Space**: Learns a Gaussian Mixture Model representation for clustering
3. **LSTM Decoder**: Generates forecasts based on the latent representation

## Key Features

- **Temporal Pattern Learning**: LSTM networks capture long-term dependencies
- **Unsupervised Clustering**: Automatically groups similar time series patterns
- **Probabilistic Forecasting**: Generates forecasts with uncertainty estimates
- **Flexible Architecture**: Configurable sequence length, forecast horizon, and model complexity

## Quick Start

### Training

```bash
# Basic training with default parameters
./run_timeseries_training.sh

# Custom training with specific parameters
./run_timeseries_training.sh \
    --epochs 200 \
    --batch-size 64 \
    --hidden-dim 128 \
    --latent-dim 32 \
    --num-clusters 6 \
    --sequence-length 30 \
    --forecast-horizon 10
```

### Testing

```bash
# Test the trained model
python test_timeseries_model.py \
    --model-path ./timeseries_results/best_model.pt \
    --hidden-dim 64 \
    --latent-dim 16 \
    --num-clusters 4
```

## Parameters

### Model Architecture
- `--hidden-dim`: LSTM hidden dimension (default: 64)
- `--latent-dim`: Latent space dimension (default: 16)
- `--num-clusters`: Number of clusters in the Gaussian Mixture (default: 4)
- `--lstm-layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)

### Training Configuration
- `--sequence-length`: Input sequence length (default: 20)
- `--forecast-horizon`: Number of steps to forecast (default: 5)
- `--batch-size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)

### Loss Weights
- `--kl-weight`: KL divergence weight (default: 0.1)
- `--recon-weight`: Reconstruction loss weight (default: 1.0)

## Usage Examples

### Python API

```python
from time_series_gmvae_lstm import TimeSeriesGMVAELSTM
import torch

# Create model
model = TimeSeriesGMVAELSTM(
    input_dim=1,
    hidden_dim=64,
    latent_dim=16,
    num_clusters=4,
    sequence_length=20,
    forecast_horizon=5
)

# Generate forecast
x = torch.randn(1, 20, 1)  # Batch of 1, sequence length 20, 1 feature
forecast = model.generate_forecast(x)

# Get cluster assignments
cluster_idx, cluster_probs = model.cluster_assignments(x)
```

### Custom Data

To use your own time series data, modify the `create_synthetic_data` function in `train_timeseries_gmvae.py`:

```python
def load_your_data():
    # Load your time series data
    # Should return numpy array of shape (n_samples, sequence_length)
    pass
```

## Output Files

The training process generates several output files:

- `best_model.pt`: Best model based on validation loss
- `model_epoch_N.pt`: Model checkpoints every N epochs
- `forecast_epoch_N.png`: Forecast visualizations
- `training_curves.png`: Training and validation loss curves
- `results.json`: Training metrics and configuration

## Applications

This model is suitable for:

- **Financial Forecasting**: Stock prices, exchange rates, market indices
- **Demand Forecasting**: Sales, inventory, resource planning
- **Sensor Data**: IoT sensors, environmental monitoring
- **Energy Forecasting**: Power consumption, renewable energy generation
- **Traffic Prediction**: Vehicle flow, congestion patterns

## Model Performance

The model performance depends on several factors:

- **Data Quality**: Clean, consistent time series data
- **Sequence Length**: Longer sequences may capture more patterns
- **Forecast Horizon**: Shorter horizons are generally more accurate
- **Number of Clusters**: Should match the diversity of patterns in your data
- **Model Complexity**: Larger models may overfit on small datasets

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or sequence length
2. **Poor Forecasting**: Increase model complexity or training epochs
3. **Overfitting**: Add dropout or reduce model size
4. **Convergence Issues**: Adjust learning rate or loss weights

### Performance Tips

- Use appropriate sequence length for your data patterns
- Normalize your time series data
- Experiment with different numbers of clusters
- Monitor training curves to detect overfitting
- Use validation data to tune hyperparameters

## Citation

If you use this time series forecasting extension in your research, please cite the original GM-VAE paper:

```
@article{dilokthanakul2016deep,
  title={Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders},
  author={Dilokthanakul, Nat and Mediano, Pedro A. M. and Garnelo, Marta and Lee, Matthew C. H. and Salimbeni, Hugh and Arulkumaran, Kai and Shanahan, Murray},
  journal={arXiv preprint arXiv:1611.02648},
  year={2016}
}
```
