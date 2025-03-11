import torch

class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_recovered, z_sim_recovered, z_neg_recovered):
        # Compute the dot products between each z and each "negative" sample
        neg = torch.einsum("ij,kj -> ik", z_recovered, z_neg_recovered)

        # Compute the dot product between each z and recovered (positive sample)
        pos = torch.einsum("ij,ij -> i", z_recovered, z_sim_recovered)

        neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

        loss_pos = -pos / self.temperature
        loss_neg = torch.logsumexp(neg_and_pos / self.temperature, dim=1)

        return (loss_pos + loss_neg).mean()

