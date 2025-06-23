import torch

def calculate_angle_preservation_error(z_original, z_reconstructed):
    # N x 3 @ 3 x N - N x N
    dot1 = z_original @ z_original.T
    dot2 = z_reconstructed @ z_reconstructed.T

    return ((dot1 - dot2).abs().mean() / 2).item()

