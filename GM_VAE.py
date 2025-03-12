import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """
    Encoder network for GM-VAE that maps inputs to latent distribution parameters
    """
    def __init__(self, input_channels=3, img_size=None, hidden_size=500, x_size=200, w_size=150, K=10):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.img_size = img_size  # Can be None for dynamic sizing
        self.hidden_size = hidden_size
        self.x_size = x_size
        self.w_size = w_size
        self.K = K
        self.flattened_size = None
        
        # Enhanced convolutional feature extraction - designed to work with variable sized inputs
        # First part of the network with convolutional layers
        self.conv_features = nn.Sequential(
            # First block - 3 -> 32 channels
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second block - 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third block - 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth block - 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fifth block - 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # MPS-compatible global pooling operation
        self.global_pool = nn.Sequential(
            # Instead of adaptive pooling, use fixed stride conv to reduce to exactly 4x4
            # This avoids the MPS divisibility requirement 
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)  # Fixed pooling instead of adaptive
        )
        
        # Calculate the flattened size after convolutions
        self.adaptive_fc = None  # Will be created in the forward pass for the first time
    
    def forward(self, x):
        """
        Encode input to distribution parameters
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            qz: Categorical distribution over K clusters
            mu_x, logvar_x: Gaussian parameters for x
            mu_w, logvar_w: Gaussian parameters for w
        """
        batch_size = x.size(0)
        
        # Feature extraction through convolutions
        x = self.conv_features(x)
        
        # Global pooling (MPS-compatible)
        x = self.global_pool(x)
        
        # Flatten
        x_flattened = x.view(batch_size, -1)
        
        # Initialize adaptive fully connected layer if needed
        if self.adaptive_fc is None:
            self.flattened_size = x_flattened.shape[1]
            print(f"Encoder output size after convolutions: {self.flattened_size}")
            
            # Create enhanced fully connected layers with dropout for regularization
            self.adaptive_fc = nn.Sequential(
                nn.Linear(self.flattened_size, self.hidden_size * 2, bias=False),
                nn.BatchNorm1d(self.hidden_size * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False),
                nn.BatchNorm1d(self.hidden_size),
                nn.LeakyReLU(0.2, inplace=True)
            ).to(x.device)
            
            # Create output layers
            self.mu_x = nn.Linear(self.hidden_size, self.x_size).to(x.device)
            self.logvar_x = nn.Linear(self.hidden_size, self.x_size).to(x.device)
            self.mu_w = nn.Linear(self.hidden_size, self.w_size).to(x.device)
            self.logvar_w = nn.Linear(self.hidden_size, self.w_size).to(x.device)
            self.qz = nn.Linear(self.hidden_size, self.K).to(x.device)
        
        # Apply fully connected layers
        x = self.adaptive_fc(x_flattened)
        
        # Distribution parameters
        mu_x = self.mu_x(x)
        logvar_x = self.logvar_x(x)
        
        mu_w = self.mu_w(x)
        logvar_w = self.logvar_w(x)
        
        # Softmax for categorical distribution over clusters
        qz = F.softmax(self.qz(x), dim=1)
        
        return qz, mu_x, logvar_x, mu_w, logvar_w


class PriorNetwork(nn.Module):
    """
    Prior network that maps w to the prior distribution parameters for x given z and w
    """
    def __init__(self, w_size=150, hidden_size=500, x_size=200, K=10):
        super(PriorNetwork, self).__init__()
        self.w_size = w_size
        self.hidden_size = hidden_size
        self.x_size = x_size
        self.K = K
        
        # Enhanced prior generation network with deeper architecture
        self.prior_stack = nn.Sequential(
            nn.Linear(w_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Prior distribution parameters for each cluster with more capacity
        self.mu_px = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_size // 2, x_size)
            ) for _ in range(K)
        ])
        
        self.logvar_px = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_size // 2, x_size)
            ) for _ in range(K)
        ])
    
    def forward(self, w_sample):
        """
        Generate prior distribution parameters for x given w and each possible z
        
        Args:
            w_sample: Sample from q(w|y) of shape [batch_size, w_size]
            
        Returns:
            mu_px, logvar_px: Prior distribution parameters for each cluster
                              Shape: [batch_size, x_size, K]
        """
        batch_size = w_sample.size(0)
        
        # Generate features
        h = self.prior_stack(w_sample)
        
        # Generate distribution parameters for each cluster
        mu_px = torch.empty(batch_size, self.x_size, self.K, device=w_sample.device)
        logvar_px = torch.empty(batch_size, self.x_size, self.K, device=w_sample.device)
        
        for i in range(self.K):
            mu_px[:, :, i] = self.mu_px[i](h)
            logvar_px[:, :, i] = self.logvar_px[i](h)
        
        return mu_px, logvar_px


class Decoder(nn.Module):
    """
    Decoder network that maps x to reconstructed input
    """
    def __init__(self, x_size=200, img_height=None, img_width=None, channels=3):
        super(Decoder, self).__init__()
        self.x_size = x_size
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        
        # Enhanced FC layers with progressive upsampling
        self.fc = nn.Sequential(
            nn.Linear(x_size, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Calculate output padding based on target size
        self.target_set = (img_height is not None and img_width is not None)
        
        # Enhanced transpose convolution stack with residual connections
        self.deconv_stack = nn.Sequential(
            # First deconv block: 512 -> 256 channels (4x4 -> 8x8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second deconv block: 256 -> 128 channels (8x8 -> 16x16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third deconv block: 128 -> 64 channels (16x16 -> 32x32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth deconv block: 64 -> 32 channels (32x32 -> 64x64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fifth deconv block: 32 -> channels (64x64 -> 128x128)
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1)
        )
        
        # This will produce a 128x128 image. For other sizes, we'll use interpolation
    
    def forward(self, x_sample):
        """
        Decode latent sample to reconstructed input
        
        Args:
            x_sample: Sample from q(x|y) of shape [batch_size, x_size]
            
        Returns:
            output: Reconstructed input
        """
        batch_size = x_sample.size(0)
        
        # Fully connected processing
        h = self.fc(x_sample)
        
        # Reshape to match expected convolutional input
        h = h.view(batch_size, 512, 4, 4)
        
        # Apply deconvolution stack
        output = self.deconv_stack(h)  # Results in [batch_size, channels, 128, 128]
        
        # If target size is specified and different from 128x128, resize
        if self.target_set and (self.img_height != 128 or self.img_width != 128):
            output = F.interpolate(
                output, 
                size=(self.img_height, self.img_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply sigmoid to get pixel values in [0, 1]
        output = torch.sigmoid(output)
        
        return output


class GMVAE(nn.Module):
    """
    Gaussian Mixture Variational Autoencoder
    
    Implementation based on:
    "Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders"
    by Nat Dilokthanakul et al.
    """
    def __init__(self, 
                 input_channels=3,
                 img_height=None, 
                 img_width=None,
                 hidden_size=500, 
                 x_size=200, 
                 w_size=150, 
                 K=10):
        """
        Args:
            input_channels: Number of channels in the input image
            img_height: Height of the input image (can be None for dynamic sizing)
            img_width: Width of the input image (can be None for dynamic sizing)
            hidden_size: Size of the hidden layer
            x_size: Dimension of the latent variable x
            w_size: Dimension of the latent variable w
            K: Number of mixture components
        """
        super(GMVAE, self).__init__()
        self.input_channels = input_channels
        self.img_height = img_height
        self.img_width = img_width
        self.hidden_size = hidden_size
        self.x_size = x_size
        self.w_size = w_size
        self.K = K
        
        # Initialize encoder - will adapt to input size
        self.encoder = Encoder(
            input_channels=input_channels,
            img_size=None,  # Let it adapt dynamically
            hidden_size=hidden_size,
            x_size=x_size,
            w_size=w_size,
            K=K
        )
        
        # Initialize prior network
        self.prior_network = PriorNetwork(
            w_size=w_size,
            hidden_size=hidden_size,
            x_size=x_size,
            K=K
        )
        
        # Initialize decoder - will generate images of specified size
        self.decoder = Decoder(
            x_size=x_size,
            img_height=img_height,
            img_width=img_width,
            channels=input_channels
        )
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of the distribution
            logvar: Log variance of the distribution
            
        Returns:
            Sample from the distribution
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x):
        """
        Forward pass through the GM-VAE
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            All parameters needed for computing the loss function
        """
        # Set decoder target size if not yet set
        if self.img_height is None or self.img_width is None:
            self.img_height = x.size(2)
            self.img_width = x.size(3)
            self.decoder.img_height = self.img_height
            self.decoder.img_width = self.img_width
            self.decoder.target_set = True
            print(f"Setting decoder target size to {self.img_height}×{self.img_width}")
        
        # Encode input
        qz, mu_x, logvar_x, mu_w, logvar_w = self.encoder(x)
        
        # Sample from latent distributions
        w_sample = self.reparameterize(mu_w, logvar_w)
        x_sample = self.reparameterize(mu_x, logvar_x)
        
        # Generate prior distribution parameters
        mu_px, logvar_px = self.prior_network(w_sample)
        
        # Decode latent sample
        recon_x = self.decoder(x_sample)
        
        return mu_x, logvar_x, mu_px, logvar_px, qz, recon_x, mu_w, logvar_w, x_sample


class GMVAELoss:
    """
    Loss function for the GM-VAE
    """
    @staticmethod
    def compute_loss(recon_x, x, mu_w, logvar_w, qz, mu_x, logvar_x, mu_px, logvar_px, x_sample, x_size, K, 
                    kl_weight=1.0, recon_weight=1.0):
        """
        Compute the loss for the GM-VAE
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu_w, logvar_w: Parameters for q(w|y)
            qz: Parameters for q(z|y)
            mu_x, logvar_x: Parameters for q(x|y)
            mu_px, logvar_px: Parameters for p(x|z,w)
            x_sample: Sample from q(x|y)
            x_size: Dimension of x
            K: Number of mixture components
            kl_weight: Weight for KL divergence terms (for annealing)
            recon_weight: Weight for reconstruction loss
            
        Returns:
            total_loss: Total loss
            components: Dictionary of loss components
        """
        batch_size = x.size(0)
        
        # 1. Reconstruction loss: -E[log P(y|x)]
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') * recon_weight
        
        # 2. KL divergence between q(w) and p(w): KL(q(w) || p(w))
        kld_w = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp()) * kl_weight
        
        # 3. KL divergence between q(z) and p(z): KL(q(z) || p(z))
        kld_z = torch.sum(qz * torch.log(K * qz + 1e-10)) * kl_weight
        
        # 4. Expected KL divergence: E_z,w[KL(q(x) || p(x|z,w))]
        # Expand dimensions for broadcasting
        mu_x_expanded = mu_x.unsqueeze(-1).expand(-1, x_size, K)
        logvar_x_expanded = logvar_x.unsqueeze(-1).expand(-1, x_size, K)
        
        # Compute KL divergence between q(x) and p(x|z,w) for each cluster
        kld_qx_px = 0.5 * (
            (logvar_px - logvar_x_expanded) + 
            ((logvar_x_expanded.exp() + (mu_x_expanded - mu_px).pow(2)) / logvar_px.exp()) - 
            1
        )
        
        # Weight by cluster probabilities
        qz_expanded = qz.unsqueeze(-1).expand(-1, K, 1)
        e_kld_qx_px = torch.sum(torch.bmm(kld_qx_px, qz_expanded)) * kl_weight
        
        # 5. Clustering validation (CV) term (entropy)
        # Expand x_sample for computation with each cluster
        x_sample_expanded = x_sample.unsqueeze(-1).expand(-1, x_size, K)
        
        # Log-likelihood for each cluster
        temp = 0.5 * x_size * np.log(2 * np.pi)
        log_likelihood = (
            -0.5 * torch.sum(((x_sample_expanded - mu_px).pow(2)) / logvar_px.exp(), dim=1) - 
            0.5 * torch.sum(logvar_px, dim=1) - 
            temp
        )
        
        # Softmax over clusters to get probabilities
        likelihood = F.softmax(log_likelihood, dim=1)
        
        # Entropy (a negative term in the objective)
        cv = torch.sum(torch.mul(torch.log(likelihood + 1e-10), likelihood))
        
        # Total loss
        total_loss = recon_loss + kld_w + kld_z + e_kld_qx_px
        
        # Store unweighted components for reporting
        raw_recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        raw_kld_w = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())
        raw_kld_z = torch.sum(qz * torch.log(K * qz + 1e-10))
        raw_e_kld_qx_px = torch.sum(torch.bmm(kld_qx_px, qz_expanded)) / kl_weight  # Remove weighting
        
        # Return total loss and individual components
        components = {
            'recon': raw_recon_loss,
            'kl_w': raw_kld_w,
            'kl_z': raw_kld_z,
            'kl_x': raw_e_kld_qx_px,
            'cv': cv,
            'recon_weighted': recon_loss,
            'kl_weighted': kld_w + kld_z + e_kld_qx_px
        }
        
        return total_loss, components 