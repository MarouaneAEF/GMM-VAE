# Gaussian Mixture Variational Autoencoder (GM-VAE)

This repository contains a PyTorch implementation of the Gaussian Mixture Variational Autoencoder (GM-VAE) described in the paper:

> "Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders"  
> by Nat Dilokthanakul, Pedro A. M. Mediano, Marta Garnelo, Matthew C. H. Lee, Hugh Salimbeni, Kai Arulkumaran, and Murray Shanahan  
> https://arxiv.org/abs/1611.02648

## Overview

The GM-VAE is a deep generative model that combines Variational Autoencoders (VAEs) with Gaussian Mixture Models (GMMs). The key advantages of GM-VAE include:

1. **Unsupervised Clustering**: It can automatically discover clusters in data without labels
2. **Generative Modeling**: It can generate new data samples from different clusters
3. **Data Reconstruction**: It can reconstruct input data with high fidelity

The model assumes that the observed data is generated from a mixture of K Gaussian distributions, where K is the number of clusters.

## Model Architecture

The GM-VAE consists of several components:

1. **Encoder**: Maps input data to distributions in the latent space
   - Outputs parameters for distributions of latent variables x and w
   - Outputs cluster assignments for each input (qz)

2. **Prior Network**: Generates prior distributions for latent variable x conditioned on w and cluster assignments z
   - For each possible cluster z, computes a conditional prior p(x|z,w)

3. **Decoder**: Maps latent variable x back to the input space
   - Reconstructs the input data from the latent representation

## Key Variables

- **x**: The main latent variable representing the data
- **w**: An auxiliary latent variable that helps with clustering
- **z**: Categorical variable indicating cluster assignment
- **qz**: Posterior distribution over cluster assignments

## Loss Function

The loss function for the GM-VAE consists of several components:

1. **Reconstruction Loss**: Measures how well the model reconstructs the input
2. **KL Divergence for w**: Regularizes the latent variable w
3. **KL Divergence for z**: Regularizes the cluster assignments
4. **Expected KL Divergence**: Measures the discrepancy between the latent distribution and the prior distribution across clusters
5. **Cluster Validation Term**: Helps with clustering by maximizing likelihood under the generative model

## Getting Started

### Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- tensorboard

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gmvae.git
cd gmvae

# Install the requirements
pip install -r requirements.txt
```

### Training

To train the GM-VAE on MNIST or CIFAR-10:

```bash
# Train on MNIST
python train_gmvae.py --dataset mnist --K 10 --epochs 100

# Train on CIFAR-10
python train_gmvae.py --dataset cifar10 --K 10 --epochs 100
```

#### Command-Line Arguments

- `--dataset`: Dataset to use ('mnist' or 'cifar10', default: 'cifar10')
- `--batch-size`: Input batch size for training (default: 64)
- `--epochs`: Number of epochs to train (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--no-cuda`: Disables CUDA training
- `--seed`: Random seed (default: 1)
- `--log-interval`: How many batches to wait before logging training status (default: 10)
- `--save-interval`: How many epochs to wait before saving model (default: 5)
- `--K`: Number of mixture components (default: 10)
- `--hidden-size`: Size of hidden layer (default: 500)
- `--x-size`: Size of latent variable x (default: 200)
- `--w-size`: Size of latent variable w (default: 150)

### Visualization

During training, the model saves:
- Reconstructed images after each epoch
- Latent space visualizations showing cluster assignments
- Training and validation loss curves (viewable in TensorBoard)

To view the results in TensorBoard:

```bash
tensorboard --logdir=runs
```

## Model Components

### GM_VAE.py

This file contains the main model implementation:

- `Encoder`: Maps inputs to latent distributions
- `PriorNetwork`: Generates prior distributions for each cluster
- `Decoder`: Reconstructs inputs from latent representations
- `GMVAE`: Main model class that combines all components
- `GMVAELoss`: Implementation of the loss function

### train_gmvae.py

This script handles training and evaluation:

- Command-line argument parsing
- Training and testing functions
- Model checkpointing
- Visualization of results

## Applications

The GM-VAE is useful for various applications, including:

1. **Unsupervised Clustering**: Automatically discovering structure in data
2. **Data Generation**: Creating new samples from learned clusters
3. **Anomaly Detection**: Identifying outliers that don't fit into any cluster
4. **Data Compression**: Using the latent representation for efficient storage
5. **Semi-Supervised Learning**: Leveraging cluster information to improve supervised tasks

## Differences from Original Implementation

This implementation offers several improvements over the original code:

1. **Modular Design**: Clear separation of encoder, prior network, and decoder
2. **Better Documentation**: Comprehensive comments explaining each component
3. **TensorBoard Integration**: Visualizations of training progress and results
4. **Command-Line Arguments**: Easily configurable parameters
5. **Loss Function Components**: Detailed breakdown of loss components
6. **Cluster Visualization**: Visualizations of discovered clusters

## References

1. Dilokthanakul, N., Mediano, P. A. M., Garnelo, M., Lee, M. C. H., Salimbeni, H., Arulkumaran, K., & Shanahan, M. (2016). Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders. *arXiv preprint arXiv:1611.02648*.

2. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.

3. Graves, A. (2016). Stochastic Backpropagation through Mixture Density Distributions. *arXiv preprint arXiv:1607.05690*. 