import torch
import torch.nn.functional as F

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


def sample_on_sphere_uniform(num_samples, dimension, radius=1):
    assert radius > 0, "Radius must be greater than 0"
    assert dimension > 0, "Dimension must be greater than 0"
    assert num_samples > 0, "Number of samples must be greater than 0"

    normal_samples = torch.randn((num_samples, dimension)) # (N, d)

    return radius * F.normalize(normal_samples, p=2, dim=1)


def sample_conditional(z, kappa, d_fixed):
    assert z.dim() == 2, "Element must be provided in a (batch, latent_dim) form"
    assert z.size(1) >= 2, "Latent dimension must be at least 2"
    assert d_fixed >= 0, "Fixed dimension must be at least 0"
    assert kappa > 0, "Kappa must be greater than 0"
    assert z.size(1) > d_fixed, "Latent vector must have at least one variable dimension"

    # Decompose the vector into variable dimensions
    u = z[:, :d_fixed]
    v = z[:, d_fixed:]


    # Saving the norms of v to enforce the sample to be norm 1
    v_norm = torch.norm(v, p=2, dim=1, keepdim=True)

    # Normalizing the variable part to be on the unit sphere
    v_normed = F.normalize(v, p=2, dim=1)

    batch_size = z.size(0)

    # Q: Why is kappa adjusted here?
    s = [sample_vMF_sequential(v_normed[b], kappa * v_norm[b] ** 2) for b in range(batch_size)]
    s = torch.stack(s, 0)

    return torch.cat((u, s * v_norm), 1)
