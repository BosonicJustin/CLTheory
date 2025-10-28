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

        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

        # TODO: Look into output dimensions here
        self.q_proj = nn.Linear(in_features=self.embed_dim, out_features=self.hidden_dim)
        self.k_proj = nn.Linear(in_features=self.embed_dim, out_features=self.hidden_dim)
        self.v_proj = nn.Linear(in_features=self.embed_dim, out_features=self.hidden_dim)

        self.mlp = MLP(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x_normed = self.norm1(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        attention_with_residual = self.self_attention(q, k, v, need_weights=False) + x
        
        return self.mlp(self.norm2(attention_with_residual)) + attention_with_residual


class ViT1x1(nn.Module):
    def __init__(self, img_size=32, embed_dim=256, hidden_dim=512, msa_heads=8, num_layers=6):
        super().__init__()
        self.img_size = img_size
        self.max_len = img_size * img_size * 3 # 1x1 patches with 3 channels
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.patch_size_one = 1

        self.pos_embed = nn.Parameter(torch.zeros(self.patch_size_one, self.max_len, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.projection = nn.Linear(in_features=self.patch_size_one, out_features=self.embed_dim)

        self.layers = nn.ModuleList(
            [VisionTransformerLayer(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, num_heads=self.msa_heads) for _ in range(self.num_layers)]
        )

    
    def project_patches(self, x):
        x_flat = x.flatten(start_dim=1)
        embedded_patches = self.projection(x_flat)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded_patches = embedded_patches.unsqueeze(1)
        positional_embedded_patches = torch.cat((cls_tokens, embedded_patches), dim=1) + self.pos_embed

        return positional_embedded_patches


    def forward(self, x):
        positional_embedded_patches = self.project_patches(x)

        for layer in self.layers:
            positional_embedded_patches = layer(positional_embedded_patches)

        return positional_embedded_patches


def get_resnet50_model():
    """
    Download (instantiate) a ResNet-50 model without pre-trained weights.

    Returns:
        torch.nn.Module: An untrained ResNet-50 model.
    """
    model = models.resnet50(weights=None)
    return model




