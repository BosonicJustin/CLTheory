from data.generation import InjectiveLinearDecoder
from encoders import construct_mlp_encoder
import torch

from simclr.simclr import SimCLR


def perform_linear_experiment(data_dimension, iterations, batch, latent_dim, sample_pair_fixed, sample_uniform_fixed, tau, device=torch.device('cpu'), f=None):
    g = InjectiveLinearDecoder(latent_dim, data_dimension)

    if f is None:
        f = construct_mlp_encoder(data_dimension, latent_dim)

    simclr = SimCLR(f, g, sample_pair_fixed, sample_uniform_fixed, tau, device)

    f, scores = simclr.train(batch, iterations)

    h_w = f.linear.weight @ g.decoder.weight
    A = h_w.T @ h_w

    print('Orthogonality of h:', A)
    print('Orthogonality of h:', A / torch.diag(A).max())
    print('Orthogonality score of h:', A.abs().sum() - A.abs().trace())

    return lambda z: f(g(z)), scores