import torch
import torch.nn as nn
import numpy as np

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


class SpiralRotation(nn.Module):
    def __init__(self, period_n):
        super(SpiralRotation, self).__init__()

        self.period_n = period_n

    def forward(self, z):
        first_dims = z[..., :2]
        second_dims = z[..., 2:]

        r = self.period_n * torch.pi * second_dims
        x, y = first_dims.chunk(2, dim=-1)

        rotated = torch.cat(
            (
                torch.cos(r) * x - torch.sin(r) * y,
                torch.sin(r) * x + torch.cos(r) * y,
            ),
            dim=1
        )

        return torch.cat([rotated, second_dims], dim=-1)

# TODO: cleanup this
# Source: Prof. Gatidis code
def rotate_2d(x, factor):
    x, y = torch.chunk(x, 2, dim=-1)
    return torch.cat((torch.cos(factor) * x - torch.sin(factor) * y,
                      torch.sin(factor) * x + torch.cos(factor) * y), -1)

def rotate_3d(x, roll=0., pitch=0., yaw=0.):
    RX = torch.tensor([
                [1., 0, 0],
                [0., np.cos(roll), -np.sin(roll)],
                [0., np.sin(roll), np.cos(roll)]
            ], dtype=x.dtype)
    RY = torch.tensor([
                    [np.cos(pitch), 0., np.sin(pitch)],
                    [0., 1., 0.],
                    [-np.sin(pitch), 0., np.cos(pitch)]
                ], dtype=x.dtype)
    RZ = torch.tensor([
                    [np.cos(yaw), -np.sin(yaw), 0.],
                    [np.sin(yaw), np.cos(yaw), 0.],
                    [0., 0., 1]
                ], dtype=x.dtype)
    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    x = torch.matmul(x, R.T)

    return x


def piecewise_rotation(x, slice_number, dim=2):
    boundaries = torch.linspace(0, 1., slice_number + 1)
    bucket = torch.bucketize(abs(x[..., dim]), boundaries, right=False) - 1
    rot = (torch.pi / (slice_number - bucket)).unsqueeze(-1)

    if dim == 0:
        x = torch.cat((x[..., :1], rotate_2d(x[..., 1:], rot)), -11)
    elif dim == 1:
        x_temp = rotate_2d(torch.stack((x[..., 0], x[..., 2]), 1), rot)
        x = torch.stack((x_temp[..., 0], x[..., 1], x_temp[..., 1]), -1)
    elif dim == 2:
        x = torch.cat((rotate_2d(x[..., :2], rot), x[..., 2:]), -1)

    return x


class Patches(torch.nn.Module):

    def __init__(self, slice_number=4):
        super().__init__()

        self.slice_number = slice_number

    def forward(self, z):
        x = piecewise_rotation(z, self.slice_number, dim=2)
        x = rotate_3d(x, 0, torch.pi / 2, 0.)

        return piecewise_rotation(x, self.slice_number, dim=2)
