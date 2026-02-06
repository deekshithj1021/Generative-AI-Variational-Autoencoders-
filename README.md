# Generative AI: Variational Autoencoders for CelebA

Image Synthesis: Trained generative models for facial reconstruction and latent space semantic analysis using PyTorch.

## Overview

This repository implements a **Variational Autoencoder (VAE)** for the CelebA dataset using PyTorch. The implementation includes:

- ✅ VAE model with encoder-decoder architecture for 64×64 RGB images
- ✅ **Reparameterization trick** for backpropagation through stochastic sampling
- ✅ **β-weighted KL divergence loss** for controlling the regularization strength
- ✅ **Latent space interpolation** (linear and spherical)
- ✅ **Semantic attribute vector calculation** for manipulating facial attributes (e.g., 'Smiling')

## Features

### 1. VAE Architecture
- **Encoder**: Convolutional neural network that maps 64×64×3 images to a 128-dimensional latent space
- **Decoder**: Transposed convolutional network that reconstructs images from latent codes
- **Reparameterization Trick**: Enables gradient flow through the stochastic sampling process

### 2. β-VAE Loss Function
The loss function balances reconstruction quality and latent space regularization:

```
Loss = Reconstruction Loss + β × KL Divergence
```

- β = 1.0: Standard VAE
- β > 1.0: Stronger regularization, more disentangled representations
- β < 1.0: Better reconstructions, less regularization

### 3. Latent Space Interpolation
Two interpolation methods are supported:
- **Linear interpolation**: z = (1-α)z₁ + αz₂
- **Spherical interpolation (SLERP)**: Better for normalized latent spaces

### 4. Semantic Attribute Vectors
Compute attribute vectors for CelebA's 40 facial attributes by finding the difference between mean latent codes of images with and without each attribute.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/deekshithj1021/Generative-AI-Variational-Autoencoders-.git
cd Generative-AI-Variational-Autoencoders-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the VAE

Train the VAE on CelebA dataset with default settings:

```bash
python train.py --data-dir ./data --save-dir ./checkpoints
```

**Key arguments:**
- `--latent-dim`: Dimension of latent space (default: 128)
- `--beta`: β weight for KL divergence (default: 1.0)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-3)

**Example with β-VAE:**
```bash
python train.py --beta 4.0 --epochs 30 --batch-size 64
```

The CelebA dataset will be automatically downloaded on first run.

### Latent Space Interpolation

Interpolate between two images from the dataset:

```bash
python interpolate.py --checkpoint ./checkpoints/vae_final.pth --num-steps 10
```

**Key arguments:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--num-steps`: Number of interpolation steps (default: 10)
- `--method`: Interpolation method - 'linear' or 'slerp' (default: linear)
- `--mode`: Source mode - 'dataset' or 'random' (default: dataset)
- `--img1-idx`, `--img2-idx`: Specific image indices to interpolate

**Example with SLERP interpolation:**
```bash
python interpolate.py --checkpoint ./checkpoints/vae_final.pth \
    --method slerp --num-steps 15 --img1-idx 100 --img2-idx 200
```

**Random latent interpolation:**
```bash
python interpolate.py --checkpoint ./checkpoints/vae_final.pth \
    --mode random --num-steps 20
```

### Computing Semantic Attribute Vectors

Calculate the semantic attribute vector for 'Smiling':

```bash
python attribute_vector.py --checkpoint ./checkpoints/vae_final.pth \
    --attribute Smiling --data-dir ./data
```

**Key arguments:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--attribute`: Attribute name (default: Smiling)
- `--max-samples`: Maximum samples per class (default: all)
- `--split`: Dataset split - 'train', 'valid', or 'test' (default: train)

**Available CelebA Attributes:**
The dataset includes 40 facial attributes:
- `5_o_Clock_Shadow`, `Arched_Eyebrows`, `Attractive`, `Bags_Under_Eyes`
- `Bald`, `Bangs`, `Big_Lips`, `Big_Nose`, `Black_Hair`, `Blond_Hair`
- `Blurry`, `Brown_Hair`, `Bushy_Eyebrows`, `Chubby`, `Double_Chin`
- `Eyeglasses`, `Goatee`, `Gray_Hair`, `Heavy_Makeup`, `High_Cheekbones`
- `Male`, `Mouth_Slightly_Open`, `Mustache`, `Narrow_Eyes`, `No_Beard`
- `Oval_Face`, `Pale_Skin`, `Pointy_Nose`, `Receding_Hairline`, `Rosy_Cheeks`
- `Sideburns`, `Smiling`, `Straight_Hair`, `Wavy_Hair`, `Wearing_Earrings`
- `Wearing_Hat`, `Wearing_Lipstick`, `Wearing_Necklace`, `Wearing_Necktie`, `Young`

**Using Attribute Vectors:**
```python
import torch
from model import VAE

# Load model and attribute vector
model = VAE(latent_dim=128)
checkpoint = torch.load('checkpoints/vae_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

attr_data = torch.load('attribute_vectors/Smiling_vector.pt')
smiling_vector = attr_data['attribute_vector']

# Apply to an image
with torch.no_grad():
    mu, _ = model.encode(image)
    
    # Add smiling attribute (alpha > 0)
    z_smiling = mu + 2.0 * smiling_vector
    img_smiling = model.decode(z_smiling)
    
    # Remove smiling attribute (alpha < 0)
    z_serious = mu - 2.0 * smiling_vector
    img_serious = model.decode(z_serious)
```

## Model Architecture Details

### Encoder
```
Input: 3 × 64 × 64 RGB image
├─ Conv2d(3 → 32, k=4, s=2, p=1) + BatchNorm + ReLU  → 32 × 32 × 32
├─ Conv2d(32 → 64, k=4, s=2, p=1) + BatchNorm + ReLU  → 64 × 16 × 16
├─ Conv2d(64 → 128, k=4, s=2, p=1) + BatchNorm + ReLU → 128 × 8 × 8
├─ Conv2d(128 → 256, k=4, s=2, p=1) + BatchNorm + ReLU → 256 × 4 × 4
├─ Flatten → 4096
├─ Linear(4096 → 128) → μ (mean)
└─ Linear(4096 → 128) → log σ² (log variance)
```

### Reparameterization Trick
```python
z = μ + σ ⊙ ε, where ε ~ N(0, I)
```

### Decoder
```
Input: 128-dimensional latent vector z
├─ Linear(128 → 4096) → Reshape to 256 × 4 × 4
├─ ConvTranspose2d(256 → 128, k=4, s=2, p=1) + BatchNorm + ReLU → 128 × 8 × 8
├─ ConvTranspose2d(128 → 64, k=4, s=2, p=1) + BatchNorm + ReLU  → 64 × 16 × 16
├─ ConvTranspose2d(64 → 32, k=4, s=2, p=1) + BatchNorm + ReLU  → 32 × 32 × 32
└─ ConvTranspose2d(32 → 3, k=4, s=2, p=1) + Sigmoid           → 3 × 64 × 64
```

## Loss Function

The β-weighted VAE loss consists of two components:

1. **Reconstruction Loss** (Binary Cross-Entropy):
   ```
   L_recon = -Σ[x log(x̂) + (1-x) log(1-x̂)]
   ```

2. **KL Divergence** (regularization):
   ```
   L_KL = -0.5 × Σ[1 + log(σ²) - μ² - σ²]
   ```

3. **Total Loss**:
   ```
   L_total = L_recon + β × L_KL
   ```

## File Structure

```
├── model.py              # VAE model architecture
├── loss.py               # β-weighted loss functions
├── train.py              # Training script
├── interpolate.py        # Latent space interpolation
├── attribute_vector.py   # Semantic attribute vector computation
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision 0.15+
- numpy 1.24+
- matplotlib 3.5+
- tqdm 4.65+
- Pillow 9.0+

## References

1. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114
2. Higgins, I., et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.
3. Liu, Z., et al. (2015). Deep Learning Face Attributes in the Wild. ICCV.

## License

MIT License

## Author

Deekshith J
