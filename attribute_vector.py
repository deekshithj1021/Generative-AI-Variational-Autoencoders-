"""
Semantic attribute vector calculation for CelebA dataset.
Computes attribute vectors (e.g., for 'Smiling') by finding the difference
between mean latent vectors of images with and without the attribute.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

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


def get_celeba_attributes():
    """
    Get list of all CelebA attribute names.
    
    Returns:
        list: List of 40 attribute names
    """
    attributes = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        'Wearing_Necktie', 'Young'
    ]
    return attributes


def compute_attribute_vector(model, dataset, attribute_name, max_samples=None, 
                             batch_size=128, device='cuda'):
    """
    Compute the semantic attribute vector for a given attribute.
    
    The attribute vector is computed as:
        v_attribute = mean(z | attribute=1) - mean(z | attribute=0)
    
    This vector can then be added to/subtracted from latent codes to add/remove
    the attribute from generated images.
    
    Args:
        model (VAE): Trained VAE model
        dataset: CelebA dataset with attributes
        attribute_name (str): Name of the attribute (e.g., 'Smiling')
        max_samples (int): Maximum number of samples to use (None for all)
        batch_size (int): Batch size for processing
        device (str): Device to run on
        
    Returns:
        tuple: (attribute_vector, pos_mean, neg_mean, pos_count, neg_count)
            - attribute_vector: The computed attribute vector
            - pos_mean: Mean latent vector for positive samples
            - neg_mean: Mean latent vector for negative samples
            - pos_count: Number of positive samples
            - neg_count: Number of negative samples
    """
    # Get attribute index
    attributes = get_celeba_attributes()
    if attribute_name not in attributes:
        raise ValueError(f"Unknown attribute: {attribute_name}. "
                        f"Available attributes: {', '.join(attributes)}")
    
    attr_idx = attributes.index(attribute_name)
    print(f"Computing attribute vector for '{attribute_name}' (index {attr_idx})...")
    
    # Separate dataset into positive and negative samples
    positive_indices = []
    negative_indices = []
    
    for idx in range(len(dataset)):
        _, attrs = dataset[idx]
        # CelebA attributes are in {-1, 1}, convert to {0, 1}
        attr_value = (attrs[attr_idx].item() + 1) // 2
        
        if attr_value == 1:
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)
        
        if max_samples and len(positive_indices) >= max_samples and len(negative_indices) >= max_samples:
            break
    
    # Limit to max_samples if specified
    if max_samples:
        positive_indices = positive_indices[:max_samples]
        negative_indices = negative_indices[:max_samples]
    
    print(f"Found {len(positive_indices)} positive and {len(negative_indices)} negative samples")
    
    # Create subsets
    positive_dataset = Subset(dataset, positive_indices)
    negative_dataset = Subset(dataset, negative_indices)
    
    # Create data loaders
    positive_loader = DataLoader(positive_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=4)
    negative_loader = DataLoader(negative_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4)
    
    # Encode all positive samples
    print("Encoding positive samples...")
    positive_latents = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(positive_loader):
            images = images.to(device)
            mu, _ = model.encode(images)
            positive_latents.append(mu.cpu())
    
    positive_latents = torch.cat(positive_latents, dim=0)
    pos_mean = positive_latents.mean(dim=0)
    
    # Encode all negative samples
    print("Encoding negative samples...")
    negative_latents = []
    with torch.no_grad():
        for images, _ in tqdm(negative_loader):
            images = images.to(device)
            mu, _ = model.encode(images)
            negative_latents.append(mu.cpu())
    
    negative_latents = torch.cat(negative_latents, dim=0)
    neg_mean = negative_latents.mean(dim=0)
    
    # Compute attribute vector
    attribute_vector = pos_mean - neg_mean
    
    print(f"Attribute vector computed:")
    print(f"  Positive samples: {len(positive_latents)}")
    print(f"  Negative samples: {len(negative_latents)}")
    print(f"  Vector norm: {torch.norm(attribute_vector).item():.4f}")
    
    return (attribute_vector, pos_mean, neg_mean, 
            len(positive_latents), len(negative_latents))


def apply_attribute_vector(model, image, attribute_vector, alpha=1.0, device='cuda'):
    """
    Apply an attribute vector to an image.
    
    Args:
        model (VAE): Trained VAE model
        image (torch.Tensor): Input image [1, 3, 64, 64]
        attribute_vector (torch.Tensor): Attribute vector to apply
        alpha (float): Strength of application (positive to add, negative to remove)
        device (str): Device to run on
        
    Returns:
        torch.Tensor: Modified image
    """
    model.eval()
    with torch.no_grad():
        # Encode image
        mu, _ = model.encode(image.to(device))
        
        # Apply attribute vector
        z_modified = mu + alpha * attribute_vector.to(device)
        
        # Decode modified latent
        image_modified = model.decode(z_modified)
    
    return image_modified


def main():
    parser = argparse.ArgumentParser(
        description='Compute semantic attribute vectors for CelebA'
    )
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Dimension of latent space (default: 128)')
    
    # Attribute parameters
    parser.add_argument('--attribute', type=str, default='Smiling',
                        help='Attribute name (default: Smiling)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per class (default: all)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for processing (default: 128)')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for CelebA dataset (default: ./data)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'valid', 'test'],
                        help='Dataset split to use (default: train)')
    
    # Output parameters
    parser.add_argument('--save-dir', type=str, default='./attribute_vectors',
                        help='Directory to save vectors (default: ./attribute_vectors)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, latent_dim=args.latent_dim, device=device)
    print('Model loaded successfully')
    
    # Load dataset
    print(f'Loading CelebA {args.split} dataset...')
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CelebA(
        root=args.data_dir,
        split=args.split,
        transform=transform,
        download=False
    )
    print(f'Dataset loaded with {len(dataset)} images')
    
    # Compute attribute vector
    attribute_vector, pos_mean, neg_mean, pos_count, neg_count = compute_attribute_vector(
        model, dataset, args.attribute, 
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=device
    )
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    save_path = os.path.join(args.save_dir, f'{args.attribute}_vector.pt')
    torch.save({
        'attribute': args.attribute,
        'attribute_vector': attribute_vector,
        'positive_mean': pos_mean,
        'negative_mean': neg_mean,
        'positive_count': pos_count,
        'negative_count': neg_count,
        'latent_dim': args.latent_dim,
    }, save_path)
    
    print(f'\nAttribute vector saved to {save_path}')
    print(f'\nTo use this vector:')
    print(f'  1. Load it: data = torch.load("{save_path}")')
    print(f'  2. Get vector: attr_vec = data["attribute_vector"]')
    print(f'  3. Apply to latent code: z_modified = z + alpha * attr_vec')
    print(f'     where alpha > 0 adds the attribute, alpha < 0 removes it')
    
    # Print available attributes
    print(f'\nAll available CelebA attributes:')
    attributes = get_celeba_attributes()
    for i, attr in enumerate(attributes, 1):
        print(f'  {i:2d}. {attr}')


if __name__ == '__main__':
    main()
