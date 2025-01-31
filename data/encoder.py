import torch
import torch.nn as nn
import torch.nn.functional as F

class SphereEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SphereEncoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 256),  # First layer
        #     nn.ReLU(),                 # Non-linear activation
        #     nn.Linear(256, 128),       # Second layer
        #     nn.Sigmoid(),
        #     nn.Linear(128, latent_dim) # Final layer to map to latent_dim
        # )

        self.encoder = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)

        return F.normalize(z, dim=1, p=2)