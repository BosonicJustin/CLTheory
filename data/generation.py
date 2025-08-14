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
    dtype = x.dtype
    device = x.device

    roll = torch.tensor(roll, dtype=dtype, device=device)
    pitch = torch.tensor(pitch, dtype=dtype, device=device)
    yaw = torch.tensor(yaw, dtype=dtype, device=device)

    RX = torch.tensor([
        [1., 0, 0],
        [0., torch.cos(roll), -torch.sin(roll)],
        [0., torch.sin(roll), torch.cos(roll)]
    ], dtype=dtype, device=device)

    RY = torch.tensor([
        [torch.cos(pitch), 0., torch.sin(pitch)],
        [0., 1., 0.],
        [-torch.sin(pitch), 0., torch.cos(pitch)]
    ], dtype=dtype, device=device)

    RZ = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0.],
        [torch.sin(yaw), torch.cos(yaw), 0.],
        [0., 0., 1.]
    ], dtype=dtype, device=device)

    R = torch.mm(RZ, torch.mm(RY, RX))  # Compose rotations
    return torch.matmul(x, R.T)  # Apply rotation


def piecewise_rotation(x, slice_number, dim=2):
    device = x.device
    dtype = x.dtype

    # Ensure boundaries are on the same device/dtype as x
    boundaries = torch.linspace(0, 1., slice_number + 1, device=device, dtype=dtype)
    
    # Bucketize input
    bucket = torch.bucketize(torch.abs(x[..., dim]), boundaries, right=False) - 1
    
    # Compute rotation factor — safe from divide-by-zero since slice_number >= bucket
    denom = torch.clamp(slice_number - bucket, min=1)  # avoid division by 0
    rot = (torch.tensor(torch.pi, device=device, dtype=dtype) / denom).unsqueeze(-1)

    # Apply piecewise rotation along specified dimension
    if dim == 0:
        x = torch.cat((x[..., :1], rotate_2d(x[..., 1:], rot)), -1)
    elif dim == 1:
        x_temp = rotate_2d(torch.stack((x[..., 0], x[..., 2]), dim=-1), rot)
        x = torch.stack((x_temp[..., 0], x[..., 1], x_temp[..., 1]), dim=-1)
    elif dim == 2:
        x = torch.cat((rotate_2d(x[..., :2], rot), x[..., 2:]), -1)

    return x


class Patches(torch.nn.Module):
    def __init__(self, slice_number=4, device='cpu'):
        super().__init__()
        self.slice_number = slice_number
        self.device = device

    def forward(self, z):
        z = z.to(self.device)
        x = piecewise_rotation(z, self.slice_number, dim=2)
        x = rotate_3d(x, 0, torch.pi / 2, 0.)
        
        result = piecewise_rotation(x, self.slice_number, dim=2)
        
        # Clamp to handle floating-point precision errors
        return torch.clamp(result, min=-1.0, max=1.0).to(self.device)


class MonomialEmbedding(nn.Module):
    """
    Fast batch-processing monomial embedding: (x1,...,xd) -> (x1, x1^2, ..., x1^max_degree, x2, x2^2, ..., x2^max_degree, ...)
    
    Maps from d-dimensional sphere to d*max_degree dimensional space.
    Example: 10D -> 70D with max_degree=7
    """
    def __init__(self, latent_dim, max_degree=7):
        super(MonomialEmbedding, self).__init__()
        
        self.latent_dim = latent_dim
        self.max_degree = max_degree
        self.output_dim = latent_dim * max_degree
        
        # Pre-compute degree powers for efficiency (1, 2, 3, ..., max_degree)
        self.register_buffer('degrees', torch.arange(1, max_degree + 1, dtype=torch.float32))
    
    def forward(self, z):
        """
        Args:
            z: (batch_size, latent_dim) tensor from sphere
        
        Returns:
            (batch_size, latent_dim * max_degree) tensor with monomial features
        """
        batch_size = z.shape[0]
        
        # Vectorized computation: z.unsqueeze(-1) ** degrees.unsqueeze(0).unsqueeze(0)
        # z: (batch_size, latent_dim) -> (batch_size, latent_dim, 1)
        # degrees: (max_degree,) -> (1, 1, max_degree)
        z_expanded = z.unsqueeze(-1)  # (batch_size, latent_dim, 1)
        degrees_expanded = self.degrees.unsqueeze(0).unsqueeze(0)  # (1, 1, max_degree)
        
        # Compute all powers at once: (batch_size, latent_dim, max_degree)
        powers = z_expanded ** degrees_expanded
        
        # Reshape to interleave coordinates: (batch_size, latent_dim * max_degree)
        # Result: [x1^1, x1^2, ..., x1^max_degree, x2^1, x2^2, ..., x2^max_degree, ...]
        result = powers.reshape(batch_size, -1)
        
        return result
    
    def get_feature_names(self):
        """Returns list of feature names for interpretability"""
        names = []
        for i in range(self.latent_dim):
            for degree in range(1, self.max_degree + 1):
                names.append(f'x{i+1}^{degree}')
        return names


class MonomialEmbedding10D70D(MonomialEmbedding):
    """Convenience class for 10D -> 70D monomial embedding (degree 7)"""
    def __init__(self):
        super(MonomialEmbedding10D70D, self).__init__(latent_dim=10, max_degree=7)


class MonomialEmbedding3D21D(MonomialEmbedding):
    """Convenience class for 3D -> 21D monomial embedding (degree 7)"""
    def __init__(self):
        super(MonomialEmbedding3D21D, self).__init__(latent_dim=3, max_degree=7)
