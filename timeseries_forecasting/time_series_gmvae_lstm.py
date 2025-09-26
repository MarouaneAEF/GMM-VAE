import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
import math

class TimeSeriesGMVAELSTM(nn.Module):
    """
    Time Series Forecasting Model combining GM-VAE with LSTM
    Architecture:
    1. LSTM encoder for temporal patterns
    2. GM-VAE for latent representation and clustering
    3. LSTM decoder for forecasting
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_clusters, 
                 lstm_layers=2, sequence_length=10, forecast_horizon=5, 
                 dropout=0.1, device='cpu'):
        super(TimeSeriesGMVAELSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.device = device
        
        # LSTM Encoder for temporal patterns
        self.lstm_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # GM-VAE components
        # Encoder to latent space
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Gaussian Mixture components
        self.pi_net = nn.Linear(latent_dim, num_clusters)  # mixture weights
        self.mu_net = nn.Linear(latent_dim, num_clusters * latent_dim)  # cluster means
        self.var_net = nn.Linear(latent_dim, num_clusters * latent_dim)  # cluster variances
        
        # Decoder from latent space
        self.decoder = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM Decoder for forecasting
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output projection for forecasting
        self.forecast_projection = nn.Linear(hidden_dim, input_dim)
        
        # Initialize cluster parameters
        self.register_buffer('cluster_means', torch.randn(num_clusters, latent_dim))
        self.register_buffer('cluster_vars', torch.ones(num_clusters, latent_dim))
        self.register_buffer('cluster_weights', torch.ones(num_clusters) / num_clusters)
        
    def encode(self, x):
        """Encode time series to latent representation"""
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm_encoder(x)
        
        # Use the last hidden state
        h = hidden[-1]  # Take last layer's hidden state
        
        # Encode to latent space
        mu = self.encoder_mean(h)
        logvar = self.encoder_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_gmm_parameters(self, z):
        """Compute Gaussian Mixture Model parameters"""
        pi = F.softmax(self.pi_net(z), dim=-1)
        mu = self.mu_net(z).view(-1, self.num_clusters, self.latent_dim)
        var = F.softplus(self.var_net(z)).view(-1, self.num_clusters, self.latent_dim) + 1e-8
        
        return pi, mu, var
    
    def sample_from_gmm(self, pi, mu, var):
        """Sample from Gaussian Mixture Model"""
        # Sample cluster assignment
        cluster_dist = Categorical(pi)
        cluster_idx = cluster_dist.sample()
        
        # Sample from selected cluster
        batch_size = mu.size(0)
        mu_selected = mu[torch.arange(batch_size), cluster_idx]
        var_selected = var[torch.arange(batch_size), cluster_idx]
        
        z_sample = mu_selected + torch.randn_like(mu_selected) * torch.sqrt(var_selected)
        return z_sample, cluster_idx
    
    def decode(self, z, forecast_length=None):
        """Decode latent representation to time series forecast"""
        if forecast_length is None:
            forecast_length = self.forecast_horizon
            
        # Decode to hidden representation
        h_decoded = self.decoder(z)
        
        # Prepare input for LSTM decoder
        # Start with the decoded hidden state
        decoder_input = h_decoded.unsqueeze(1).repeat(1, forecast_length, 1)
        
        # LSTM decoding
        lstm_out, _ = self.lstm_decoder(decoder_input)
        
        # Project to output dimension
        forecast = self.forecast_projection(lstm_out)
        
        return forecast
    
    def forward(self, x, return_components=False):
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Compute GMM parameters
        pi, mu_gmm, var_gmm = self.compute_gmm_parameters(z)
        
        # Sample from GMM
        z_sample, cluster_idx = self.sample_from_gmm(pi, mu_gmm, var_gmm)
        
        # Decode to forecast
        forecast = self.decode(z_sample)
        
        if return_components:
            return {
                'forecast': forecast,
                'z': z,
                'z_sample': z_sample,
                'mu': mu,
                'logvar': logvar,
                'pi': pi,
                'mu_gmm': mu_gmm,
                'var_gmm': var_gmm,
                'cluster_idx': cluster_idx
            }
        
        return forecast
    
    def compute_loss(self, x, forecast, mu, logvar, pi, mu_gmm, var_gmm, 
                    kl_weight=1.0, recon_weight=1.0):
        """Compute combined loss for time series forecasting"""
        batch_size = x.size(0)
        
        # Reconstruction loss (MSE for time series)
        recon_loss = F.mse_loss(forecast, x[:, -self.forecast_horizon:])
        
        # KL divergence between q(z|x) and p(z)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()
        
        # GMM loss (negative log likelihood)
        # Compute log probability under each cluster
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        z_expanded = z.unsqueeze(1).expand(-1, self.num_clusters, -1)
        
        # Compute log probability for each cluster
        log_prob_clusters = []
        for k in range(self.num_clusters):
            cluster_mu = mu_gmm[:, k, :]
            cluster_var = var_gmm[:, k, :]
            
            # Normal distribution for cluster k
            normal_dist = Normal(cluster_mu, torch.sqrt(cluster_var))
            log_prob_k = normal_dist.log_prob(z).sum(dim=1)
            log_prob_clusters.append(log_prob_k)
        
        log_prob_clusters = torch.stack(log_prob_clusters, dim=1)
        
        # Weighted log probability
        weighted_log_prob = log_prob_clusters + torch.log(pi + 1e-8)
        gmm_loss = -torch.logsumexp(weighted_log_prob, dim=1).mean()
        
        # Total loss
        total_loss = recon_weight * recon_loss + kl_weight * kl_loss + gmm_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'gmm_loss': gmm_loss
        }
    
    def generate_forecast(self, x, steps_ahead=None):
        """Generate forecast for given number of steps ahead"""
        if steps_ahead is None:
            steps_ahead = self.forecast_horizon
            
        self.eval()
        with torch.no_grad():
            # Encode the input sequence
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            
            # Generate forecast
            forecast = self.decode(z, steps_ahead)
            
        return forecast
    
    def cluster_assignments(self, x):
        """Get cluster assignments for input sequences"""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            pi, mu_gmm, var_gmm = self.compute_gmm_parameters(z)
            
            # Get most likely cluster
            cluster_probs = pi
            cluster_assignments = torch.argmax(cluster_probs, dim=1)
            
        return cluster_assignments, cluster_probs
