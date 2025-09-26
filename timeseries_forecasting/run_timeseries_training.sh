#!/bin/bash

# Time Series GM-VAE LSTM Training Script
# This script trains the model for time series forecasting

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Time Series GM-VAE LSTM Training...${NC}"

# Create results directory
mkdir -p timeseries_results

# Set default parameters
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.001
HIDDEN_DIM=64
LATENT_DIM=16
NUM_CLUSTERS=4
SEQUENCE_LENGTH=20
FORECAST_HORIZON=5
LSTM_LAYERS=2
DROPOUT=0.1
KL_WEIGHT=0.1
RECON_WEIGHT=1.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --latent-dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --num-clusters)
            NUM_CLUSTERS="$2"
            shift 2
            ;;
        --sequence-length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --forecast-horizon)
            FORECAST_HORIZON="$2"
            shift 2
            ;;
        --lstm-layers)
            LSTM_LAYERS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --kl-weight)
            KL_WEIGHT="$2"
            shift 2
            ;;
        --recon-weight)
            RECON_WEIGHT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --batch-size BATCH_SIZE        Batch size for training (default: 32)"
            echo "  --epochs EPOCHS                Number of epochs (default: 100)"
            echo "  --lr LEARNING_RATE             Learning rate (default: 0.001)"
            echo "  --hidden-dim HIDDEN_DIM        LSTM hidden dimension (default: 64)"
            echo "  --latent-dim LATENT_DIM        Latent space dimension (default: 16)"
            echo "  --num-clusters NUM_CLUSTERS    Number of clusters (default: 4)"
            echo "  --sequence-length LENGTH       Input sequence length (default: 20)"
            echo "  --forecast-horizon HORIZON     Forecast horizon (default: 5)"
            echo "  --lstm-layers LAYERS           Number of LSTM layers (default: 2)"
            echo "  --dropout DROPOUT              Dropout rate (default: 0.1)"
            echo "  --kl-weight WEIGHT             KL divergence weight (default: 0.1)"
            echo "  --recon-weight WEIGHT          Reconstruction loss weight (default: 1.0)"
            echo "  --device DEVICE                Device to use (cpu, cuda, mps, auto)"
            echo "  -h, --help                     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Display configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Latent Dim: $LATENT_DIM"
echo "  Number of Clusters: $NUM_CLUSTERS"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Forecast Horizon: $FORECAST_HORIZON"
echo "  LSTM Layers: $LSTM_LAYERS"
echo "  Dropout: $DROPOUT"
echo "  KL Weight: $KL_WEIGHT"
echo "  Recon Weight: $RECON_WEIGHT"
echo "  Device: ${DEVICE:-auto}"

# Run training
echo -e "${YELLOW}Starting training...${NC}"

python train_timeseries_gmvae.py \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --hidden-dim $HIDDEN_DIM \
    --latent-dim $LATENT_DIM \
    --num-clusters $NUM_CLUSTERS \
    --sequence-length $SEQUENCE_LENGTH \
    --forecast-horizon $FORECAST_HORIZON \
    --lstm-layers $LSTM_LAYERS \
    --dropout $DROPOUT \
    --kl-weight $KL_WEIGHT \
    --recon-weight $RECON_WEIGHT \
    --device ${DEVICE:-auto} \
    --save-dir ./timeseries_results

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "Results saved in: ${YELLOW}./timeseries_results/${NC}"
    echo -e "You can view the forecast plots and training curves in the results directory."
else
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi
