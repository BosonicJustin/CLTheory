import torchvision.models as models
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class VisionTransformerLayer(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # MultiheadAttention handles Q, K, V projections internally
        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)

        self.mlp = MLP(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # Pre-norm architecture (as in ViT paper)
        # Self-attention block with residual
        x_normed = self.norm1(x)
        attn_output, _ = self.self_attention(x_normed, x_normed, x_normed, need_weights=False)
        x = x + attn_output

        # MLP block with residual
        x = x + self.mlp(self.norm2(x))

        return x


class ViT1x1(nn.Module):
    def __init__(self, img_size=32, embed_dim=256, hidden_dim=512, msa_heads=8, num_layers=6):
        super().__init__()
        self.img_size = img_size
        self.num_patches = img_size * img_size  # Number of 1x1 patches (H*W)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.msa_heads = msa_heads
        self.patch_size_one = 1

        # Positional embeddings for N+1 tokens (patches + CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.projection = nn.Linear(in_features=3, out_features=self.embed_dim)  # Project from C=3 channels to embed_dim

        self.layers = nn.ModuleList(
            [VisionTransformerLayer(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, num_heads=self.msa_heads) for _ in range(self.num_layers)]
        )

    
    def project_patches(self, x):
        # (B, C, H, W) -> (B, N, C) where N = H*W (number of 1x1 patches)
        batch_size = x.shape[0]

        # Reshape: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x_patches = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = 1024 for 32x32 image

        # Project each patch from C channels to embed_dim
        embedded_patches = self.projection(x_patches)  # (B, N, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)

        # Concatenate CLS token with patch embeddings
        embedded_patches = torch.cat((cls_tokens, embedded_patches), dim=1)  # (B, N+1, embed_dim)

        # Add positional embeddings
        positional_embedded_patches = embedded_patches + self.pos_embed

        return positional_embedded_patches


    def forward(self, x):
        positional_embedded_patches = self.project_patches(x)

        for layer in self.layers:
            positional_embedded_patches = layer(positional_embedded_patches)

        return positional_embedded_patches[:, 0, :]


def get_resnet50_model():
    """
    Download (instantiate) a ResNet-50 model without pre-trained weights.

    Returns:
        torch.nn.Module: An untrained ResNet-50 model.
    """
    model = models.resnet50(weights=None)
    return model
