from data.generation import InjectiveLinearDecoder
from encoders import construct_mlp_encoder
import torch
import numpy as np
from sklearn import linear_model

from simclr.simclr import SimCLR


def linear_unrotation(z, z_enc):
    """
    Compute the linear unrotation of encoded representations back to the original latent space.

    Fits a linear regression from z_enc -> z to learn the inverse of the orthogonal
    transformation induced by contrastive learning, then applies it to z_enc.

    Args:
        z: Original latent samples (torch.Tensor)
        z_enc: Encoded representations (torch.Tensor)

    Returns:
        torch.Tensor: Unrotated representations projected back onto the unit sphere
    """
    z_np = z.detach().cpu().numpy()
    z_enc_np = z_enc.detach().cpu().numpy()

    model = linear_model.LinearRegression()
    model.fit(z_enc_np, z_np)

    unrotated = model.predict(z_enc_np)
    unrotated = unrotated / np.linalg.norm(unrotated, axis=1, keepdims=True)

    return torch.from_numpy(unrotated).float()


def perform_linear_experiment(data_dimension, iterations, batch, latent_dim, sample_pair_fixed, sample_uniform_fixed, tau, device=torch.device('cpu'), f=None):
    g = InjectiveLinearDecoder(latent_dim, data_dimension)

    if f is None:
        f = construct_mlp_encoder(data_dimension, latent_dim)

    simclr = SimCLR(f, g, sample_pair_fixed, sample_uniform_fixed, tau, device)

    f, scores = simclr.train(batch, iterations)

    return lambda z: f(g(z)), scores