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


class InjectiveLinearDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(InjectiveLinearDecoder, self).__init__()

        assert latent_dim <= output_dim

        self.decoder = nn.Linear(latent_dim, output_dim)

        matrix = torch.diag(torch.rand(latent_dim))

        if latent_dim < output_dim:
            lower_matrix = torch.randn((output_dim - latent_dim, latent_dim))
            matrix = torch.vstack([matrix, lower_matrix])

        assert matrix.shape == (output_dim, latent_dim)

        with torch.no_grad():
            self.decoder.weight.copy_(matrix)

    def forward(self, z):
        return self.decoder(z)
