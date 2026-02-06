"""
Example usage script demonstrating all VAE features.
This script shows how to use the trained model for various tasks.
"""

import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model import VAE


def example_1_basic_usage():
    """Example 1: Basic VAE usage - encode and decode"""
    print("=" * 60)
    print("Example 1: Basic VAE Usage")
    print("=" * 60)
    
    # Create model
    model = VAE(latent_dim=128)
    model.eval()
    
    # Create a random image (batch_size=1, channels=3, height=64, width=64)
    x = torch.rand(1, 3, 64, 64)
    
    # Forward pass
    x_recon, mu, logvar = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent log variance shape: {logvar.shape}")
    print()


def example_2_reparameterization_trick():
    """Example 2: Understanding the reparameterization trick"""
    print("=" * 60)
    print("Example 2: Reparameterization Trick")
    print("=" * 60)
    
    model = VAE(latent_dim=128)
    
    # Create sample mu and logvar
    mu = torch.randn(1, 128)
    logvar = torch.randn(1, 128)
    
    # Sample multiple times using reparameterization trick
    samples = []
    for _ in range(5):
        z = model.reparameterize(mu, logvar)
        samples.append(z)
    
    # All samples are different (stochastic)
    print(f"Mean of latent distribution: {mu[0, :5]}")
    print(f"Log variance: {logvar[0, :5]}")
    print()
    print("Five different samples from the same distribution:")
    for i, z in enumerate(samples, 1):
        print(f"  Sample {i}: {z[0, :5]}")
    print()


def example_3_beta_vae_loss():
    """Example 3: β-VAE loss with different β values"""
    print("=" * 60)
    print("Example 3: β-VAE Loss")
    print("=" * 60)
    
    from loss import vae_loss_per_sample
    
    # Create dummy data
    x = torch.rand(4, 3, 64, 64)
    x_recon = torch.rand(4, 3, 64, 64)
    mu = torch.randn(4, 128)
    logvar = torch.randn(4, 128)
    
    # Compute loss with different β values
    for beta in [0.5, 1.0, 2.0, 4.0]:
        total_loss, recon_loss, kl_loss = vae_loss_per_sample(
            x_recon, x, mu, logvar, beta=beta
        )
        print(f"β = {beta:.1f}:")
        print(f"  Total Loss: {total_loss.item():.4f}")
        print(f"  Recon Loss: {recon_loss.item():.4f}")
        print(f"  KL Loss: {kl_loss.item():.4f}")
        print(f"  β × KL: {(beta * kl_loss).item():.4f}")
        print()


def example_4_sampling():
    """Example 4: Sampling random images from the latent space"""
    print("=" * 60)
    print("Example 4: Sampling Random Images")
    print("=" * 60)
    
    model = VAE(latent_dim=128)
    model.eval()
    
    # Sample 16 random images
    num_samples = 16
    device = torch.device('cpu')
    samples = model.sample(num_samples, device)
    
    print(f"Generated {num_samples} random images")
    print(f"Sample shape: {samples.shape}")
    print("These images are sampled from N(0, I) in latent space")
    print()


def example_5_interpolation():
    """Example 5: Latent space interpolation"""
    print("=" * 60)
    print("Example 5: Latent Space Interpolation")
    print("=" * 60)
    
    model = VAE(latent_dim=128)
    model.eval()
    
    # Create two random images
    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.rand(1, 3, 64, 64)
    
    # Encode to latent space
    with torch.no_grad():
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
    
    # Linear interpolation
    print("Linear Interpolation:")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        img_interp = model.decode(z_interp)
        print(f"  α = {alpha:.2f}, interpolated image shape: {img_interp.shape}")
    print()


def example_6_attribute_vector():
    """Example 6: Using semantic attribute vectors"""
    print("=" * 60)
    print("Example 6: Semantic Attribute Vectors")
    print("=" * 60)
    
    model = VAE(latent_dim=128)
    model.eval()
    
    # Simulate an attribute vector (in practice, computed from dataset)
    # This represents the direction in latent space for an attribute
    attribute_vector = torch.randn(128)
    
    # Original image
    img = torch.rand(1, 3, 64, 64)
    
    # Encode
    with torch.no_grad():
        mu, _ = model.encode(img)
    
    # Apply attribute vector with different strengths
    print("Applying 'Smiling' attribute vector:")
    for alpha in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        z_modified = mu + alpha * attribute_vector.unsqueeze(0)
        img_modified = model.decode(z_modified)
        
        if alpha < 0:
            effect = "Remove attribute"
        elif alpha == 0:
            effect = "Original"
        else:
            effect = "Add attribute"
        
        print(f"  α = {alpha:+.1f} ({effect}): {img_modified.shape}")
    print()


def example_7_model_architecture():
    """Example 7: Inspecting model architecture"""
    print("=" * 60)
    print("Example 7: Model Architecture")
    print("=" * 60)
    
    model = VAE(latent_dim=128)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    print("Encoder architecture:")
    print(model.encoder)
    print()
    
    print("Decoder architecture:")
    print(model.decoder)
    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "VAE Examples and Usage Demonstrations" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    examples = [
        example_1_basic_usage,
        example_2_reparameterization_trick,
        example_3_beta_vae_loss,
        example_4_sampling,
        example_5_interpolation,
        example_6_attribute_vector,
        example_7_model_architecture,
    ]
    
    for example in examples:
        example()
        input("Press Enter to continue to next example...")
        print("\n")
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
