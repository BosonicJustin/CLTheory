import torch
import torch.nn.functional as F

import vMF

def sample_on_sphere_uniform(num_samples, dimension, radius=1):
    assert radius > 0, "Radius must be greater than 0"
    assert dimension > 0, "Dimension must be greater than 0"
    assert num_samples > 0, "Number of samples must be greater than 0"

    normal_samples = torch.randn((num_samples, dimension)) # (N, d)

    return radius * F.normalize(normal_samples, p=2, dim=1)


def sample_conditional(z, kappa, d_fixed):
    assert z.dim() == 2, "Element must be provided in a (batch, latent_dim) form"
    assert z.size(1) >= 2, "Latent dimension must be at least 2"
    assert d_fixed > 0, "Fixed dimensions must be greater than 0"
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
    s = [vMF.sample_vMF_sequential(v_normed[b], kappa * v_norm[b] ** 2) for b in range(batch_size)]
    s = torch.stack(s, 0)

    return torch.cat((u, s * v_norm), 1)
