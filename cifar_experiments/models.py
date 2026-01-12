import torchvision.models as models
from torchvision.models import VisionTransformer
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


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


class ViTWithCheckpointing(nn.Module):
    """
    ViT wrapper that supports gradient checkpointing for memory efficiency.

    Gradient checkpointing trades compute for memory by not storing intermediate
    activations during forward pass, recomputing them during backward pass.

    This wrapper maintains state_dict compatibility with the unwrapped ViT model
    by mapping keys appropriately in state_dict() and load_state_dict().
    """
    def __init__(self, vit_model, use_checkpointing=True):
        super().__init__()
        self.vit = vit_model
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        # Patch embedding and position encoding (from ViT)
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Apply encoder layers with or without checkpointing
        if self.use_checkpointing and self.training:
            # Checkpoint each encoder layer
            for layer in self.vit.encoder.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            x = self.vit.encoder.ln(x)
        else:
            x = self.vit.encoder(x)

        # Get class token and apply head
        x = x[:, 0]
        x = self.vit.heads(x)
        return x

    def state_dict(self, *args, **kwargs):
        # Return state dict without 'vit.' prefix for compatibility
        state = self.vit.state_dict(*args, **kwargs)
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Load state dict into the wrapped vit model
        return self.vit.load_state_dict(state_dict, *args, **kwargs)


def get_vit_encoder(img_size=32, patch_size=4, embed_dim=384, num_layers=6, num_heads=6, mlp_dim=1536, output_dim=DEFAULT_EMBED_DIM, use_checkpointing=False):
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
        use_checkpointing: Enable gradient checkpointing for memory efficiency

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
        num_classes=embed_dim,  # Temporary, we replace heads below
    )

    # Remove classification head and add our own projection layer
    model.heads = nn.Linear(embed_dim, output_dim)

    if use_checkpointing:
        return ViTWithCheckpointing(model, use_checkpointing=True)

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
