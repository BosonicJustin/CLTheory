from torch import nn
import torch
import torch.nn.functional as F


class FeedForwardEncoder(nn.Module):
    def __init__(self, input_dim=550, hidden_dims=[1024, 512, 256], output_dim=128):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return F.normalize(self.encoder(x), dim=-1)
