import torchvision.models as models
from torchvision.models import VisionTransformer
import torch
import torch.nn as nn


# Common output dimension for all encoders (for fair comparison with raw embeddings)
DEFAULT_EMBED_DIM = 512


class MLPEncoder(nn.Module):
    """
    Multi-Layer Perceptron encoder for CIFAR-10.
    Flattens input images and passes them through fully connected layers.

    Low inductive bias - treats image as flat vector with no spatial structure.

    Args:
        hidden_dim: Dimension of hidden layers (default: 2048)
        num_hidden_layers: Number of hidden layers (default: 3)
        output_dim: Dimension of output embeddings (default: 512)
        dropout: Dropout probability (default: 0.1)
        input_dim: Input dimension after flattening (default: 3072 for CIFAR-10)
    """
    def __init__(self, hidden_dim=2048, num_hidden_layers=3, output_dim=DEFAULT_EMBED_DIM, dropout=0.1, input_dim=3072):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Build layers
        layers = []

        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # Output layer: hidden_dim -> output_dim (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images of shape (B, C, H, W)

        Returns:
            Embeddings of shape (B, output_dim)
        """
        # Flatten: (B, C, H, W) -> (B, C*H*W)
        x = x.view(x.size(0), -1)

        # Pass through network
        x = self.network(x)

        return x


def get_resnet18_encoder(output_dim=DEFAULT_EMBED_DIM):
    """
    ResNet-18 encoder for CIFAR-10.

    High inductive bias - convolutional architecture with strong spatial priors.

    Args:
        output_dim: Dimension of output embeddings (default: 512)

    Returns:
        ResNet-18 model with classification head replaced by projection.
        Output shape: (B, output_dim)
        Parameters: ~11.2M
    """
    model = models.resnet18(weights=None)

    # ResNet18 has 512-dim features before fc layer
    resnet_feat_dim = 512

    if output_dim == resnet_feat_dim:
        # No projection needed - just remove classification head
        model.fc = nn.Identity()
    else:
        # Project to desired output dimension
        model.fc = nn.Linear(resnet_feat_dim, output_dim)

    return model


def get_vit_encoder(img_size=32, patch_size=4, embed_dim=384, num_layers=6, num_heads=6, mlp_dim=1536, output_dim=DEFAULT_EMBED_DIM):
    """
    Vision Transformer encoder for CIFAR-10 using torchvision.

    Medium inductive bias - attention-based with patch structure but no convolutions.

    4x4 patches on 32x32 images = 64 patches (8x8 grid)
    ~10.7M parameters to match ResNet18 (~11.2M)

    Args:
        img_size: Input image size (32 for CIFAR-10)
        patch_size: Patch size (4 recommended for CIFAR-10)
        embed_dim: Transformer hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_dim: MLP hidden dimension in transformer blocks
        output_dim: Dimension of output embeddings (default: 512)

    Returns:
        ViT model outputting (B, output_dim) embeddings
        Parameters: ~10.7M
    """
    model = VisionTransformer(
        image_size=img_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=embed_dim,
        mlp_dim=mlp_dim,
        num_classes=output_dim,  # This sets the head output dimension
    )

    # Fix: torchvision ViT initializes heads to zeros for custom num_classes.
    # Reinitialize with standard truncated normal (std=0.02) for proper training.
    for module in model.heads.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return model


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test all encoders
    print(f"Testing encoders on CIFAR-10 sized input (3, 32, 32)...")
    print(f"Common output dimension: {DEFAULT_EMBED_DIM}")
    print("=" * 60)

    x = torch.randn(2, 3, 32, 32)

    # Test ResNet18
    resnet = get_resnet18_encoder()
    out_resnet = resnet(x)
    print(f"ResNet18:")
    print(f"  Output shape: {out_resnet.shape}")
    print(f"  Parameters: {count_parameters(resnet) / 1e6:.1f}M")

    # Test ViT
    vit = get_vit_encoder()
    out_vit = vit(x)
    print(f"\nViT (4x4 patches):")
    print(f"  Output shape: {out_vit.shape}")
    print(f"  Parameters: {count_parameters(vit) / 1e6:.1f}M")

    # Test MLP
    mlp = MLPEncoder()
    out_mlp = mlp(x)
    print(f"\nMLPEncoder:")
    print(f"  Output shape: {out_mlp.shape}")
    print(f"  Parameters: {count_parameters(mlp) / 1e6:.1f}M")

    print("\n" + "=" * 60)
    print("All encoders output same dimension - ready for fair comparison!")
