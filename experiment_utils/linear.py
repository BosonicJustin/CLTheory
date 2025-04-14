from data.generation import InjectiveLinearDecoder
from encoders import construct_mlp_encoder

from simclr.simclr import SimCLR


def perform_linear_experiment(data_dimension, iterations, batch, latent_dim, sample_pair_fixed, sample_uniform_fixed, tau):
    g = InjectiveLinearDecoder(latent_dim, data_dimension)
    f = construct_mlp_encoder(data_dimension, latent_dim)

    simclr = SimCLR(f, g, sample_pair_fixed, sample_uniform_fixed, tau)

    f = simclr.train(batch, iterations)

    return lambda z: f(g(z))