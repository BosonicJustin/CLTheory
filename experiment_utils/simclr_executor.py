from simclr.simclr import SimCLR
from visualization_utils.scoring import plot_scores
from visualization_utils.spheres import scatter3d_sphere, visualize_spheres_side_by_side
from experiment_utils.linear import linear_unrotation
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


    def execute(self, batch_size, iterations, plt, save_dir=None, exp_name=None):
        f, scores = SimCLR(
            self.f, self.g, self.sample_pair_fixed, self.sample_uniform_fixed, self.tau, self.device
        ).train(batch_size, iterations)

        fig_scores = plot_scores(plt, scores)
        if save_dir and exp_name:
            fig_scores.savefig(save_dir / f'{exp_name}_training_scores.png', dpi=150, bbox_inches='tight')
        plt.show()

        h = lambda z: f(self.g(z))

        z = self.sample_uniform_fixed(4000).to(self.device)
        z_enc = h(z).to(self.device)

        fig_sidebyside = visualize_spheres_side_by_side(plt, z.cpu(), z_enc.cpu())
        if save_dir and exp_name:
            fig_sidebyside.savefig(save_dir / f'{exp_name}_original_vs_encoded.png', dpi=150, bbox_inches='tight')
        plt.show()

        z = self.sample_uniform_fixed(100000).to(self.device)
        z_enc = h(z).to(self.device)

        fig_orig = scatter3d_sphere(plt, z.cpu(), z.cpu(), s=10, a=.8)
        if save_dir and exp_name:
            fig_orig.savefig(save_dir / f'{exp_name}_sphere_original.png', dpi=150, bbox_inches='tight')
        plt.show()

        z_enc = h(z)
        fig_enc = scatter3d_sphere(plt, z.cpu(), z_enc.cpu(), s=10, a=.8)
        if save_dir and exp_name:
            fig_enc.savefig(save_dir / f'{exp_name}_sphere_encoded.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Unrotation
        z_unrotated = linear_unrotation(z, z_enc)
        fig_unrot = scatter3d_sphere(plt, z.cpu(), z_unrotated, s=10, a=.8)
        if save_dir and exp_name:
            fig_unrot.savefig(save_dir / f'{exp_name}_sphere_unrotated.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Orthogonal transformation loss: ", self.compute_orthogonal_transformation_loss(self.sample_pair_fixed, batch_size))
        print("Training loss: ", scores['eval_losses'][-1])

        if save_dir and exp_name:
            print(f"Figures saved to {save_dir}/ with prefix '{exp_name}_'")

        return f, scores