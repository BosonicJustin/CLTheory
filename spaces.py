"""Definition of topological/mathematical spaces with probability densities defined on."""

"""This was taken from Zimmerman's codebase and extended"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from sampling import sphere_sampling

class Space(ABC):
    """Base class."""

    @abstractmethod
    def uniform(self, size, device):
        pass

    @abstractmethod
    def normal(self, mean, std, size, device):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass


class NSphereSpace(Space):
    """N-dimensional hypersphere, i.e. {x | |x| = r and x € R^N}."""

    def __init__(self, n, r=1):
        self.n = n
        self._n_sub = n - 1
        self.r = r

    @property
    def dim(self):
        return self.n

    def uniform(self, size, device="cpu"):
        x = torch.randn((size, self.n), device=device)
        x /= torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))

        return x

    def normal(self, mean, std, size, device="cpu"):
        """Sample from a Normal distribution in R^N and then project back on the sphere.

        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """

        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        mean = mean.to(device)
        if not torch.is_tensor(std):
            std = torch.ones(self.n) * std
        std = std.to(device)

        assert mean.shape[1] == self.n
        assert torch.allclose(
            torch.sqrt((mean ** 2).sum(-1)), torch.Tensor([self.r]).to(device)
        )

        result = torch.randn((size, self.n), device=device) * std + mean
        # project back on sphere
        result /= torch.sqrt(torch.sum(result ** 2, dim=-1, keepdim=True))

        return result

    def von_mises_fisher(self, mean, kappa, size, device="cpu"):
        """Sample from a von Mises-Fisher distribution (=Normal distribution on a hypersphere).

        Args:
            mean: Value(s) to sample around.
            kappa: Concentration parameter of the distribution.
            size: Number of samples to draw.
            device: torch device identifier
        """
        assert len(mean.shape) == 1 or (len(mean.shape) == 2 and len(mean) == size)
        assert mean.shape[-1] == self.n

        mean = mean.cpu().detach().numpy()

        if len(mean.shape) == 1:
            mean = np.repeat(np.expand_dims(mean, 0), size, axis=0)

        assert mean.shape[1] == self.n
        assert np.allclose(np.sqrt((mean ** 2).sum(-1)), self.r)

        samples_np = sphere_sampling.sample_vMF(mean, kappa, size)
        samples = torch.Tensor(samples_np).to(device)

        return samples

    def sample_pair_vmf(self, num_samples,  kappa, device="cpu"):
        z = self.uniform(num_samples)
        z_aug = self.von_mises_fisher(z, kappa, num_samples)

        return z, z_aug

    def sample_pair_normal(self, num_samples, concentration, device="cpu"):
        z = self.uniform(num_samples)
        z_aug = self.normal(z, concentration, num_samples)

        return z, z_aug
