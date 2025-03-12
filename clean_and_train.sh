#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Cleaning up previous image results...${NC}"

# Create results directory if it doesn't exist
mkdir -p results

# Function to clean a specific result directory
clean_result_dir() {
    local model_dir="$1"
    
    if [ -d "$model_dir" ]; then
        echo -e "Cleaning ${YELLOW}$model_dir${NC}"
        
        # Remove all images from standard reconstructions
        if [ -d "$model_dir/reconstructions/standard" ]; then
            rm -f "$model_dir/reconstructions/standard"/*.png
            echo "  - Cleaned standard reconstructions"
        fi
        
        # Remove all images from comparisons
        if [ -d "$model_dir/reconstructions/comparisons" ]; then
            rm -f "$model_dir/reconstructions/comparisons"/*.png
            echo "  - Cleaned comparisons"
        fi
        
        # Remove all images from large_comparisons
        if [ -d "$model_dir/reconstructions/large_comparisons" ]; then
            rm -f "$model_dir/reconstructions/large_comparisons"/*.png
            echo "  - Cleaned large comparisons"
        fi
        
        # Remove all images from clusters
        if [ -d "$model_dir/reconstructions/clusters" ]; then
            rm -f "$model_dir/reconstructions/clusters"/*.png
            echo "  - Cleaned clusters"
        fi
    else
        # Create reconstruction directories
        mkdir -p "$model_dir/reconstructions/standard"
        mkdir -p "$model_dir/reconstructions/comparisons"
        mkdir -p "$model_dir/reconstructions/large_comparisons"
        mkdir -p "$model_dir/reconstructions/clusters"
        echo "Created new directories for $model_dir"
    fi
}

# Clean all potential result directories
for K_value in 10 12 15; do
    clean_result_dir "results/gmvae_custom_K$K_value"
done

echo -e "${GREEN}All image directories have been cleaned!${NC}"
echo -e "${YELLOW}Starting MPS-optimized training...${NC}"

# Make sure the run script is executable
chmod +x run_mps_optimized.sh

# Run the training script
./run_mps_optimized.sh

echo -e "${GREEN}Training complete!${NC}"
echo -e "You can view the results using: ${YELLOW}open view_results.html${NC}" 