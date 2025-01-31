import torch
import torch.nn as nn


class SphereDecoderLinear(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(SphereDecoderLinear, self).__init__()
        self.decoder = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.decoder(z)


class SphereDecoderIdentity(nn.Module):
    def __init__(self):
        super(SphereDecoderIdentity, self).__init__()

    def forward(self, z):
        return z


class RotationDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(RotationDecoder, self).__init__()