"""
Loss functions for VAE training.
Implements β-weighted KL divergence loss and reconstruction loss.
"""

import torch
import torch.nn.functional as F


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    Compute the β-weighted VAE loss.
    
    Loss = Reconstruction Loss + β * KL Divergence
    
    The β parameter allows controlling the weight of the KL divergence term:
    - β = 1.0: Standard VAE (β-VAE with β=1)
    - β > 1.0: Stronger regularization, more disentangled representations
    - β < 1.0: Weaker regularization, better reconstructions
    
    Args:
        x_recon (torch.Tensor): Reconstructed images [batch_size, 3, 64, 64]
        x (torch.Tensor): Original images [batch_size, 3, 64, 64]
        mu (torch.Tensor): Mean of latent distribution [batch_size, latent_dim]
        logvar (torch.Tensor): Log variance of latent distribution [batch_size, latent_dim]
        beta (float): Weight for KL divergence term (default: 1.0)
        
    Returns:
        tuple: (total_loss, recon_loss, kl_loss)
            - total_loss: Combined loss value
            - recon_loss: Reconstruction loss component
            - kl_loss: KL divergence component (before β weighting)
    """
    # Reconstruction loss (Binary Cross Entropy)
    # Assumes x and x_recon are in [0, 1] range
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL Divergence: D_KL(q(z|x) || p(z))
    # For q(z|x) = N(mu, sigma^2) and p(z) = N(0, 1):
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with β weighting on KL term
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def vae_loss_per_sample(x_recon, x, mu, logvar, beta=1.0):
    """
    Compute the β-weighted VAE loss averaged per sample.
    
    This version returns the mean loss per sample in the batch,
    which is useful for consistent loss reporting across different batch sizes.
    
    Args:
        x_recon (torch.Tensor): Reconstructed images [batch_size, 3, 64, 64]
        x (torch.Tensor): Original images [batch_size, 3, 64, 64]
        mu (torch.Tensor): Mean of latent distribution [batch_size, latent_dim]
        logvar (torch.Tensor): Log variance of latent distribution [batch_size, latent_dim]
        beta (float): Weight for KL divergence term (default: 1.0)
        
    Returns:
        tuple: (total_loss, recon_loss, kl_loss)
            - total_loss: Combined loss value (mean per sample)
            - recon_loss: Reconstruction loss component (mean per sample)
            - kl_loss: KL divergence component (mean per sample, before β weighting)
    """
    batch_size = x.size(0)
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta)
    
    # Average over batch
    total_loss = total_loss / batch_size
    recon_loss = recon_loss / batch_size
    kl_loss = kl_loss / batch_size
    
    return total_loss, recon_loss, kl_loss
