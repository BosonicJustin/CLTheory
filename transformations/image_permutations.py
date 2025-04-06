import torch


class ImagePermutationTransform(torch.nn.Module):
    def __init__(self, H, W, channels=3, seed=None, device=None):
        super().__init__()

        self.H, self.W = H, W
        self.channels = channels
        self.seed = seed
        self.device = device

        self.perm = (self._generate_fixed_image_permutation
                     (H, W, channels=channels, seed=seed, device=device))
        self.inverse_perm = [torch.argsort(p) for p in self.perm]

    def forward(self, x):
        return self._apply_permutation(x, self.perm)

    def inverse(self, x_permuted):
        return self._apply_permutation(x_permuted, self.inverse_perm)

    def _generate_fixed_image_permutation(self, H, W, channels=3, seed=None, device=None):
        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
        else:
            g = None

        return [torch.randperm(H * W, generator=g, device=device) for _ in range(channels)]

    def _apply_permutation(self, x, perms):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).clone()
        x_perm_flat = torch.zeros_like(x_flat)

        for c in range(C):
            x_perm_flat[:, c] = x_flat[:, c][:, perms[c]]

        return x_perm_flat.view(B, C, H, W)
