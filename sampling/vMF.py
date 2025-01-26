import torch

# Note: Took it from example code - inspect and understand


def sample_vMF_sequential(mu, kappa):
    """Generate num_samples N-dimensional samples from von Mises Fisher
    distribution around center mu \in R^N with concentration kappa.
    """

    assert mu.dim() == 1 and len(mu) >= 2
    n_features = len(mu)
    w = _sample_weight_sequential(kappa, n_features)
    # sample a point v on the unit sphere that's orthogonal to mu
    v = _sample_orthonormal_to_sequential(mu)
    # compute new point
    sample = v * torch.sqrt(1.0 - w ** 2) + w * mu

    return sample


def _sample_weight_sequential(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (torch.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + dim * torch.log(1 - x ** 2)

    while True:
        z = torch.distributions.beta.Beta(dim / 2.0, dim / 2.0).sample()
        # z = torch.random.beta(dim / 2.0, dim / 2.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = torch.rand(1)
        if kappa * w + dim * torch.log(1.0 - x * w) - c >= torch.log(u):
            return w


def _sample_orthonormal_to_sequential(mu):
    """Sample point on sphere orthogonal to mu."""
    v = torch.randn(len(mu), device=mu.device)
    proj_mu_v = mu * torch.dot(mu, v) / torch.norm(mu)
    orthto = v - proj_mu_v
    return orthto / torch.norm(orthto)