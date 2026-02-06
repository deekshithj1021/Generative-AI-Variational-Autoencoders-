"""
Variational Autoencoder (VAE) model for CelebA dataset.
Implements encoder, decoder, and reparameterization trick.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder for 64x64 RGB images.
    
    Architecture:
    - Encoder: Convolutional layers to compress images to latent space
    - Decoder: Transposed convolutional layers to reconstruct images
    - Latent dimension: 128 (can be configured)
    """
    
    def __init__(self, latent_dim=128, image_channels=3):
        """
        Initialize the VAE model.
        
        Args:
            latent_dim (int): Dimension of the latent space
            image_channels (int): Number of input image channels (3 for RGB)
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: maps 64x64x3 image to latent space
        self.encoder = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Latent space layers
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder: maps latent space back to 64x64x3 image
        self.decoder = nn.Sequential(
            # Input: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),  # 3 x 64 x 64
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def encode(self, x):
        """
        Encode input images to latent space parameters.
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 64, 64]
            
        Returns:
            tuple: (mu, logvar) - mean and log variance of latent distribution
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        
        This allows backpropagation through the sampling process.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Sample epsilon from N(0, 1)
        z = mu + eps * std              # Reparameterization trick
        return z
    
    def decode(self, z):
        """
        Decode latent vectors to images.
        
        Args:
            z (torch.Tensor): Latent vectors [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Reconstructed images [batch_size, 3, 64, 64]
        """
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 4, 4)  # Reshape to 4D tensor
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 64, 64]
            
        Returns:
            tuple: (x_recon, mu, logvar)
                - x_recon: Reconstructed images
                - mu: Mean of latent distribution
                - logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device):
        """
        Sample random images from the latent space.
        
        Args:
            num_samples (int): Number of images to sample
            device (torch.device): Device to create tensors on
            
        Returns:
            torch.Tensor: Generated images [num_samples, 3, 64, 64]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
