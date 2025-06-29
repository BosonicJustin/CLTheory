from torch import nn
import layers as ls
from typing import List, Union
from typing_extensions import Literal

import torch.nn.functional as F

import torch

__all__ = ["get_mlp"]


def get_mlp(
    n_in: int,
    n_out: int,
    layers: List[int],
    layer_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
    output_normalization: Union[
        None,
        Literal["fixed_sphere"],
        Literal["learnable_sphere"],
        Literal["fixed_box"],
        Literal["learnable_box"],
    ] = None,
    output_normalization_kwargs=None,
):
    """
    Creates an MLP.

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        layers: Number of neurons for each hidden layer
        layer_normalization: Normalization for each hidden layer.
            Possible values: bn (batch norm), gn (group norm), None
        output_normalization: (Optional) Normalization applied to output of network.
        output_normalization_kwargs: Arguments passed to the output normalization, e.g., the radius for the sphere.
    """
    modules: List[nn.Module] = []

    def add_module(n_layer_in: int, n_layer_out: int, last_layer: bool = False):
        modules.append(nn.Linear(n_layer_in, n_layer_out))
        # perform normalization & activation not in last layer
        if not last_layer:
            if layer_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))
            elif layer_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))
            modules.append(nn.LeakyReLU())

        return n_layer_out

    if len(layers) > 0:
        n_out_last_layer = n_in
    else:
        assert n_in == n_out, "Network with no layers must have matching n_in and n_out"
        modules.append(layers.Lambda(lambda x: x))

    layers.append(n_out)

    for i, l in enumerate(layers):
        n_out_last_layer = add_module(n_out_last_layer, l, i == len(layers) - 1)

    if output_normalization_kwargs is None:
        output_normalization_kwargs = {}

    if output_normalization == "fixed_sphere":
        modules.append(ls.RescaleLayer(fixed_r=True, **output_normalization_kwargs))
    elif output_normalization == "learnable_sphere":
        modules.append(ls.RescaleLayer(init_r=1.0, fixed_r=False))
    elif output_normalization == "fixed_box":
        modules.append(
            ls.SoftclipLayer(
                n=n_out, fixed_abs_bound=True, **output_normalization_kwargs
            )
        )
    elif output_normalization == "learnable_box":
        modules.append(
            ls.SoftclipLayer(
                n=n_out, fixed_abs_bound=False, **output_normalization_kwargs
            )
        )
    elif output_normalization is None:
        pass
    else:
        raise ValueError("output_normalization")

    return nn.Sequential(*modules)


def get_flow(
    n_in: int,
    n_out: int,
    init_identity: bool = False,
    coupling_block: Union[Literal["gin", "glow"]] = "gin",
    num_nodes: int = 8,
    node_size_factor: int = 1,
):
    """
    Creates an flow-based network.

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        init_identity: Initialize weights to identity network.
        coupling_block: Coupling method to use to combine nodes.
        num_nodes: Depth of the flow network.
        node_size_factor: Multiplier for the hidden units per node.
    """

    # do lazy imports here such that the package is only
    # required if one wants to use the flow mixing
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm

    def _invertible_subnet_fc(c_in, c_out, init_identity):
        subnet = nn.Sequential(
            nn.Linear(c_in, c_in * node_size),
            nn.ReLU(),
            nn.Linear(c_in * node_size, c_in * node_size),
            nn.ReLU(),
            nn.Linear(c_in * node_size, c_out),
        )
        if init_identity:
            subnet[-1].weight.data.fill_(0.0)
            subnet[-1].bias.data.fill_(0.0)
        return subnet

    assert n_in == n_out

    if coupling_block == "gin":
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == "glow"
        block = Fm.GLOWCouplingBlock

    nodes = [Ff.InputNode(n_in, name="input")]

    for k in range(num_nodes):
        nodes.append(
            Ff.Node(
                nodes[-1],
                block,
                {
                    "subnet_constructor": lambda c_in, c_out: _invertible_subnet_fc(
                        c_in, c_out, init_identity
                    ),
                    "clamp": 2.0,
                },
                name=f"coupling_{k}",
            )
        )

    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)


def construct_mlp_encoder(latent_dim, data_dim, device="cpu"):
    return get_mlp(
        n_in=latent_dim,
        n_out=data_dim,
        layers=[
            latent_dim * 10,
            latent_dim * 50,
            latent_dim * 50,
            latent_dim * 50,
            latent_dim * 50,
            latent_dim * 10,
        ],
        output_normalization="fixed_sphere",
    ).to(device)


class SphericalEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[128, 256, 128], output_dim=3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)  # Helps with spherical training
            ])
            prev_dim = hidden_dim
            
        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        h = self.backbone(x)
        z = self.projection(h)
        return F.normalize(z, p=2, dim=-1)

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        z = self.linear(x)
        return F.normalize(z, p=2, dim=-1)


class InverseSpiralEncoder(nn.Module):
    """
    Direct parametric inverse of SpiralRotation.
    Learns to predict the rotation factor and applies inverse rotation.
    """
    def __init__(self, input_dim, latent_dim, period_n):
        super(InverseSpiralEncoder, self).__init__()
        self.period_n = period_n
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Network to predict the rotation control parameter
        self.rotation_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predicts the control parameter for rotation
        )
        
        # Optional: additional encoding layers
        if input_dim != latent_dim:
            self.final_encoder = nn.Linear(input_dim, latent_dim)
        else:
            self.final_encoder = nn.Identity()

    def forward(self, x):
        # Extract first two dims and control dims
        first_dims = x[..., :2]  # [batch, 2]
        control_dims = x[..., 2:]  # [batch, remaining_dims]
        
        # Predict rotation factor from the full input
        predicted_control = self.rotation_predictor(x)  # [batch, 1]
        
        # Apply inverse rotation
        r = self.period_n * torch.pi * predicted_control
        x_rot, y_rot = first_dims.chunk(2, dim=-1)
        
        # Inverse rotation (negative angle)
        x_original = torch.cos(-r) * x_rot - torch.sin(-r) * y_rot
        y_original = torch.sin(-r) * x_rot + torch.cos(-r) * y_rot
        
        # Reconstruct original latent
        reconstructed = torch.cat([x_original, y_original, predicted_control], dim=-1)
        
        z = self.final_encoder(reconstructed)
        return F.normalize(z, p=2, dim=-1)


class InversePatchesEncoder(nn.Module):
    """
    Direct parametric inverse of the Patches transformation.
    Attempts to undo: piecewise_rotation -> rotate_3d -> piecewise_rotation
    """
    def __init__(self, input_dim, latent_dim, slice_number=4):
        super(InversePatchesEncoder, self).__init__()
        self.slice_number = slice_number
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Network to predict which bucket/slice the original point was in
        self.bucket_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, slice_number),  # Softmax over buckets
            nn.Softmax(dim=-1)
        )
        
        # Network to predict the original z-coordinate (controls piecewise rotation)
        self.z_predictor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # Assuming z is in [-1, 1]
        )
        
        # Final encoding layer
        if input_dim != latent_dim:
            self.final_encoder = nn.Linear(input_dim, latent_dim)
        else:
            self.final_encoder = nn.Identity()

    def inverse_piecewise_rotation(self, x, z_pred, bucket_weights):
        """Apply inverse piecewise rotation using predicted z and bucket weights"""
        device = x.device
        dtype = x.dtype
        
        # Weighted inverse rotation based on predicted buckets
        reconstructed = torch.zeros_like(x)
        
        for bucket_idx in range(self.slice_number):
            # Compute inverse rotation for this bucket
            denom = max(1, self.slice_number - bucket_idx)
            rot_angle = -torch.pi / denom  # Negative for inverse
            
            # Convert to tensor with correct device and dtype
            rot_angle = torch.tensor(rot_angle, dtype=dtype, device=device)
            
            # Apply inverse rotation
            xy_coords = x[..., :2]
            cos_rot = torch.cos(rot_angle)
            sin_rot = torch.sin(rot_angle)
            
            x_inv = cos_rot * xy_coords[..., 0] - sin_rot * xy_coords[..., 1]
            y_inv = sin_rot * xy_coords[..., 0] + cos_rot * xy_coords[..., 1]
            
            xy_inv = torch.stack([x_inv, y_inv], dim=-1)
            x_bucket = torch.cat([xy_inv, x[..., 2:]], dim=-1)
            
            # Weight by bucket probability - fix broadcasting issue
            # bucket_weights shape: [batch_size, slice_number]
            # weight should be [batch_size, 1] to broadcast with x_bucket [batch_size, 3]
            weight = bucket_weights[..., bucket_idx].unsqueeze(-1)  # [batch_size, 1]
            reconstructed += weight * x_bucket
            
        return reconstructed

    def inverse_3d_rotation(self, x):
        """Apply inverse of the 3D rotation (inverse pitch of π/2)"""
        dtype = x.dtype
        device = x.device
        
        # Inverse rotation matrix for pitch = -π/2
        pitch = torch.tensor(-torch.pi / 2, dtype=dtype, device=device)
        cos_pitch = torch.cos(pitch)
        sin_pitch = torch.sin(pitch)
        
        RY_inv = torch.tensor([
            [cos_pitch, 0., sin_pitch],
            [0., 1., 0.],
            [-sin_pitch, 0., cos_pitch]
        ], dtype=dtype, device=device)
        
        return torch.matmul(x, RY_inv.T)

    def forward(self, x):
        # Predict original z-coordinate and bucket probabilities
        z_pred = self.z_predictor(x)
        bucket_weights = self.bucket_predictor(x)
        
        # Step 1: Inverse of second piecewise rotation
        x_step1 = self.inverse_piecewise_rotation(x, z_pred, bucket_weights)
        
        # Step 2: Inverse 3D rotation
        x_step2 = self.inverse_3d_rotation(x_step1)
        
        # Step 3: Inverse of first piecewise rotation
        x_reconstructed = self.inverse_piecewise_rotation(x_step2, z_pred, bucket_weights)
        
        # Replace z-coordinate with prediction
        if x_reconstructed.shape[-1] >= 3:
            x_reconstructed = torch.cat([
                x_reconstructed[..., :2], 
                z_pred, 
                x_reconstructed[..., 3:]
            ], dim=-1)
        
        z = self.final_encoder(x_reconstructed)
        return F.normalize(z, p=2, dim=-1)