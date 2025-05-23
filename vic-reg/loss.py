import torch


class VarianceLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, eps=1e-6):
        super(VarianceLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.relu = torch.nn.ReLU()

    def s(self, z):
        return torch.sqrt(z.var(dim=0) + self.eps)

    # Expected latent intputs are (N, d)
    def forward(self, latents):
        deviations = self.s(latents)

        return self.relu(self.gamma - deviations).mean()


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        super(CovarianceLoss, self).__init__()


    def forward(self, latents):
        d = latents.shape[1]
        cov = torch.cov(latents.T)
        diag_mask = ~torch.eye(cov.size(0), dtype=torch.bool, device=cov.device)
        masked = (cov * diag_mask)

        return (masked ** 2).sum() / d


class AlignmentLoss(torch.nn.Module):
    def __init__(self):
        super(AlignmentLoss, self).__init__()

    def forward(self, latents, similar_latents):
        return ((similar_latents - latents) ** 2).sum(dim=1).mean()


class VicRegLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, eps=1e-6, lambda_c=25, mu_c=25, nu_c=1):
        super(VicRegLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.variance_loss = VarianceLoss(gamma, eps)
        self.covariance_loss = CovarianceLoss()
        self.alignment_loss = AlignmentLoss()

        self.lambda_c = lambda_c
        self.mu_c = mu_c
        self.nu_c = nu_c

    def forward(self, latents, similar_latents):
        l1 = self.mu_c * (self.variance_loss(latents) + self.variance_loss(similar_latents))
        l2 = self.nu_c * (self.covariance_loss(latents) + self.covariance_loss(similar_latents))
        l3 = self.lambda_c * self.alignment_loss(latents, similar_latents)

        return l1 + l2 + l3
