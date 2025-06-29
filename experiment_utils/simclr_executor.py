from simclr.simclr import SimCLR
from visualization_utils.scoring import plot_scores
from visualization_utils.spheres import scatter3d_sphere, visualize_spheres_side_by_side
import torch

class SimCLRExecutor:
    def __init__(self, f, g, sample_pair_fixed, sample_uniform_fixed, tau, device):
        self.f = f
        self.g = g
        self.sample_pair_fixed = sample_pair_fixed
        self.sample_uniform_fixed = sample_uniform_fixed
        self.tau = tau
        self.device = device

    def compute_orthogonal_transformation_loss(self, sample_pair, batch_size, latent_dim=3):
        z, z_aug = sample_pair(batch_size)

        z_neg = torch.nn.functional.normalize(
            torch.randn((batch_size, batch_size, latent_dim), device=z.device), p=2, dim=-1
        )

        pos = - torch.sum(z * z_aug, dim=-1).mean() / self.tau
        neg = torch.log(torch.exp((z.unsqueeze(1) * z_neg).sum(dim=-1) / self.tau).sum(-1)).mean()

        return (pos + neg).item()
        

    def execute(self, batch_size, iterations, plt):
        f, scores = SimCLR(
            self.f, self.g, self.sample_pair_fixed, self.sample_uniform_fixed, self.tau, self.device
        ).train(batch_size, iterations)

        plot_scores(plt, scores)

        h = lambda z: f(self.g(z))

        z = self.sample_uniform_fixed(1000).to(self.device)
        z_enc = h(z).to(self.device)

        visualize_spheres_side_by_side(plt, z.cpu(), z_enc.cpu())

        z = self.sample_uniform_fixed(100000).to(self.device)
        z_enc = h(z).to(self.device)

        fig = scatter3d_sphere(plt, z.cpu(), z.cpu(), s=10, a=.8)
        fig = scatter3d_sphere(plt, z.cpu(), h(z).cpu(), s=10, a=.8)

        print("Orthogonal transformation loss: ", self.compute_orthogonal_transformation_loss(self.sample_pair_fixed, batch_size))
        print("Training loss: ", scores['eval_losses'][-1])

        return f, scores