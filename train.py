"""
Training script for VAE on CelebA dataset.
Trains the model with β-weighted KL loss and saves checkpoints.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import VAE
from loss import vae_loss_per_sample


def get_celeba_dataloader(data_dir, batch_size=128, num_workers=4, image_size=64):
    """
    Create a DataLoader for the CelebA dataset.
    
    Args:
        data_dir (str): Root directory where CelebA dataset will be downloaded/stored
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        image_size (int): Size to resize images to (default: 64 for 64x64)
        
    Returns:
        DataLoader: PyTorch DataLoader for CelebA dataset
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CelebA(
        root=data_dir,
        split='train',
        transform=transform,
        download=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def save_reconstructions(model, dataloader, device, epoch, save_dir='results'):
    """
    Save sample reconstructions for visualization.
    
    Args:
        model (VAE): Trained VAE model
        dataloader (DataLoader): Data loader
        device (torch.device): Device to run on
        epoch (int): Current epoch number
        save_dir (str): Directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        images, _ = next(iter(dataloader))
        images = images[:8].to(device)  # Take first 8 images
        
        # Reconstruct
        x_recon, _, _ = model(images)
        
        # Plot original and reconstructed
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        
        for i in range(8):
            # Original
            img = images[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstructed
            img_recon = x_recon[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(img_recon)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png'))
        plt.close()
    
    model.train()


def train_vae(args):
    """
    Train the VAE model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = VAE(latent_dim=args.latent_dim).to(device)
    print(f'Model created with latent dimension: {args.latent_dim}')
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create data loader
    print('Loading CelebA dataset...')
    dataloader = get_celeba_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=64
    )
    print(f'Dataset loaded with {len(dataloader.dataset)} images')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print(f'Starting training for {args.epochs} epochs...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_sum = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(images)
            
            # Compute loss with β weighting
            total_loss, recon_loss, kl_loss = vae_loss_per_sample(
                x_recon, images, mu, logvar, beta=args.beta
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.2f}',
                    'recon': f'{recon_loss.item():.2f}',
                    'kl': f'{kl_loss.item():.2f}'
                })
        
        # Average losses over epoch
        avg_total_loss = total_loss_sum / len(dataloader)
        avg_recon_loss = recon_loss_sum / len(dataloader)
        avg_kl_loss = kl_loss_sum / len(dataloader)
        
        print(f'Epoch {epoch} - Loss: {avg_total_loss:.2f}, '
              f'Recon: {avg_recon_loss:.2f}, KL: {avg_kl_loss:.2f}')
        
        # Save reconstructions
        if epoch % args.save_interval == 0:
            save_reconstructions(model, dataloader, device, epoch, 
                               os.path.join(args.save_dir, 'reconstructions'))
            
            # Save model checkpoint
            checkpoint_path = os.path.join(args.save_dir, f'vae_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'vae_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_total_loss,
    }, final_path)
    print(f'Training complete! Final model saved to {final_path}')


def main():
    parser = argparse.ArgumentParser(description='Train VAE on CelebA dataset')
    
    # Model parameters
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Dimension of latent space (default: 128)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta weight for KL divergence (default: 1.0)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for CelebA dataset (default: ./data)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Saving parameters
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    train_vae(args)


if __name__ == '__main__':
    main()
