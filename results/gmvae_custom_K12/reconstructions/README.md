# Understanding GMM-VAE Reconstructions

This folder contains visualizations of the GMM-VAE model's reconstruction capabilities. The model processes high-resolution custom images and generates reconstructions that attempt to capture the original images' key features while modeling them in a compressed latent space.

## Folder Organization

The reconstructions are organized into four distinct categories:

### 1. Standard Reconstructions (`standard/`)

These are traditional reconstruction grids where:
- The **top row** displays the original input images
- The **bottom row** shows the corresponding reconstructed images

This layout provides a quick overview of the model's performance across multiple images. Files are named `reconstruction_epoch_X.png` where X is the training epoch.

### 2. Side-by-Side Comparisons (`comparisons/`)

These visualizations show direct comparisons of original vs. reconstructed images arranged side by side:
- **Left**: Original image
- **Right**: Reconstructed image

This format makes it easier to directly compare details between individual image pairs. Files are named `comparison_epoch_X.png` where X is the training epoch.

### 3. Large Comparison Grids (`large_comparisons/`)

These are expanded versions of the side-by-side comparisons with more image pairs (typically 16 pairs instead of 8). These grids provide:
- A broader sample of reconstructions 
- More detailed view of reconstruction quality across different image types
- Better insight into how different visual elements are reconstructed

Files are named `large_comparison_epoch_X.png` where X is the training epoch.

### 4. Cluster Visualizations (`clusters/`)

These visualizations show how images are clustered in the latent space:
- Points represent individual images
- Colors indicate cluster assignments
- The 2D projection shows the first two dimensions of the latent space

These visualizations help understand how the model groups similar images together and can reveal patterns in your image collection. Files are named `clusters_epoch_X.png` where X is the training epoch.

## Interpreting Results

When reviewing reconstructions, consider:

1. **Overall Structure**: Are the major shapes and composition preserved?
2. **Color Accuracy**: How well does the model reproduce the color palette?
3. **Fine Details**: What level of detail is maintained or lost?
4. **Progression**: How do reconstructions improve over training epochs?
5. **Clustering**: Do images that appear visually similar end up in the same clusters?

The model balances reconstruction quality with the constraints of the latent representation. Higher values of K (number of clusters) and larger latent dimensions generally allow for more detailed reconstructions but may reduce the model's ability to find meaningful clusters. 