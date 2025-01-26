import torch
import torch.nn as nn

class SphereDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(SphereDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)
