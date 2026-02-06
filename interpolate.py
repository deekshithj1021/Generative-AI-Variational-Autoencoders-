"""
Latent space interpolation script for VAE.
Interpolates between two images in the latent space and generates intermediate images.
"""

import os
import argparse
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from model import VAE


def load_model(checkpoint_path, latent_dim=128, device='cuda'):
    """
    Load a trained VAE model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        latent_dim (int): Dimension of latent space
        device (str): Device to load model on
        
    Returns:
        VAE: Loaded model
    """
    model = VAE(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def slerp(z1, z2, alpha):
    """
    Spherical linear interpolation (SLERP) between two latent vectors.
    
    SLERP is often preferred over linear interpolation in latent spaces
    because it maintains constant "speed" and works better for normalized vectors.
    
    Args:
        z1 (torch.Tensor): First latent vector
        z2 (torch.Tensor): Second latent vector
        alpha (float): Interpolation factor in [0, 1]
        
    Returns:
        torch.Tensor: Interpolated latent vector
    """
    # Normalize vectors
    z1_norm = z1 / torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = z2 / torch.norm(z2, dim=-1, keepdim=True)
    
    # Compute angle between vectors
    dot = torch.sum(z1_norm * z2_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    omega = torch.acos(dot)
    
    # Compute SLERP
    sin_omega = torch.sin(omega)
    s1 = torch.sin((1.0 - alpha) * omega) / sin_omega
    s2 = torch.sin(alpha * omega) / sin_omega
    
    return s1 * z1 + s2 * z2


def linear_interpolate(z1, z2, alpha):
    """
    Linear interpolation between two latent vectors.
    
    Args:
        z1 (torch.Tensor): First latent vector
        z2 (torch.Tensor): Second latent vector
        alpha (float): Interpolation factor in [0, 1]
        
    Returns:
        torch.Tensor: Interpolated latent vector
    """
    return (1 - alpha) * z1 + alpha * z2


def interpolate_images(model, img1, img2, num_steps=10, method='linear', device='cuda'):
    """
    Interpolate between two images in latent space.
    
    Args:
        model (VAE): Trained VAE model
        img1 (torch.Tensor): First image [1, 3, 64, 64]
        img2 (torch.Tensor): Second image [1, 3, 64, 64]
        num_steps (int): Number of interpolation steps
        method (str): Interpolation method ('linear' or 'slerp')
        device (str): Device to run on
        
    Returns:
        torch.Tensor: Interpolated images [num_steps, 3, 64, 64]
    """
    model.eval()
    with torch.no_grad():
        # Encode images to latent space
        mu1, logvar1 = model.encode(img1.to(device))
        mu2, logvar2 = model.encode(img2.to(device))
        
        # Use mean of distribution for interpolation (not sampling)
        z1 = mu1
        z2 = mu2
        
        # Generate interpolation steps
        interpolated_images = []
        alphas = np.linspace(0, 1, num_steps)
        
        for alpha in alphas:
            if method == 'slerp':
                z_interp = slerp(z1, z2, alpha)
            else:  # linear
                z_interp = linear_interpolate(z1, z2, alpha)
            
            # Decode interpolated latent vector
            img_interp = model.decode(z_interp)
            interpolated_images.append(img_interp)
        
        # Stack all interpolated images
        interpolated_images = torch.cat(interpolated_images, dim=0)
        
    return interpolated_images


def interpolate_from_dataset(args):
    """
    Load two random images from dataset and interpolate between them.
    
    Args:
        args: Command line arguments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, latent_dim=args.latent_dim, device=device)
    print('Model loaded successfully')
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CelebA(
        root=args.data_dir,
        split='test',
        transform=transform,
        download=False
    )
    
    # Select two random images
    idx1 = args.img1_idx if args.img1_idx is not None else np.random.randint(len(dataset))
    idx2 = args.img2_idx if args.img2_idx is not None else np.random.randint(len(dataset))
    
    img1, _ = dataset[idx1]
    img2, _ = dataset[idx2]
    
    img1 = img1.unsqueeze(0)  # Add batch dimension
    img2 = img2.unsqueeze(0)
    
    print(f'Interpolating between images {idx1} and {idx2}...')
    
    # Perform interpolation
    interpolated = interpolate_images(
        model, img1, img2, 
        num_steps=args.num_steps, 
        method=args.method,
        device=device
    )
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save as grid
    save_path = os.path.join(args.save_dir, f'interpolation_{idx1}_{idx2}.png')
    save_image(interpolated, save_path, nrow=args.num_steps, padding=2)
    print(f'Interpolation saved to {save_path}')
    
    # Also create a detailed plot
    fig, axes = plt.subplots(1, args.num_steps, figsize=(2*args.num_steps, 2))
    
    for i in range(args.num_steps):
        img = interpolated[i].cpu().permute(1, 2, 0).numpy()
        if args.num_steps == 1:
            axes.imshow(img)
            axes.axis('off')
        else:
            axes[i].imshow(img)
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('Start', fontsize=8)
            elif i == args.num_steps - 1:
                axes[i].set_title('End', fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, f'interpolation_plot_{idx1}_{idx2}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {plot_path}')


def interpolate_random_latent(args):
    """
    Generate interpolation between two random latent vectors.
    
    Args:
        args: Command line arguments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, latent_dim=args.latent_dim, device=device)
    print('Model loaded successfully')
    
    # Sample two random latent vectors
    z1 = torch.randn(1, args.latent_dim).to(device)
    z2 = torch.randn(1, args.latent_dim).to(device)
    
    print(f'Interpolating between random latent vectors...')
    
    # Generate interpolation steps
    interpolated_images = []
    alphas = np.linspace(0, 1, args.num_steps)
    
    model.eval()
    with torch.no_grad():
        for alpha in alphas:
            if args.method == 'slerp':
                z_interp = slerp(z1, z2, alpha)
            else:  # linear
                z_interp = linear_interpolate(z1, z2, alpha)
            
            # Decode interpolated latent vector
            img_interp = model.decode(z_interp)
            interpolated_images.append(img_interp)
    
    interpolated = torch.cat(interpolated_images, dim=0)
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    save_path = os.path.join(args.save_dir, 'interpolation_random.png')
    save_image(interpolated, save_path, nrow=args.num_steps, padding=2)
    print(f'Interpolation saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Interpolate in VAE latent space')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Dimension of latent space (default: 128)')
    
    # Interpolation parameters
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of interpolation steps (default: 10)')
    parser.add_argument('--method', type=str, default='linear',
                        choices=['linear', 'slerp'],
                        help='Interpolation method (default: linear)')
    
    # Image source
    parser.add_argument('--mode', type=str, default='dataset',
                        choices=['dataset', 'random'],
                        help='Source of images to interpolate (default: dataset)')
    parser.add_argument('--img1-idx', type=int, default=None,
                        help='Index of first image (random if not specified)')
    parser.add_argument('--img2-idx', type=int, default=None,
                        help='Index of second image (random if not specified)')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for CelebA dataset (default: ./data)')
    
    # Output parameters
    parser.add_argument('--save-dir', type=str, default='./interpolations',
                        help='Directory to save results (default: ./interpolations)')
    
    args = parser.parse_args()
    
    if args.mode == 'dataset':
        interpolate_from_dataset(args)
    else:
        interpolate_random_latent(args)


if __name__ == '__main__':
    main()
