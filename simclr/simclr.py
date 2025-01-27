import torch


# Here SimCLR will be without the projection head
# We will pass the samples after the augmentation
class SimCLR(torch.nn.Module):
    def __init__(self, encoder, samples, similar_samples):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.samples = samples
        self.augmentation_module = similar_samples

    #
    def fit(self):
        raise Exception("Not implemented")


class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature

    # Here I'm expecting an input of shape (2n, d_latent)
    def forward(self, z):
        s_matrix = z @ z.t()  / self.temperature
        batch_size = z.size(0)

        # Here we will be summing over the horizontal direction of the matrix (in the exp), hence masking the diagonal
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        s_matrix = s_matrix.masked_fill(mask, float('-inf'))

        # Here the logic is that we stacked the original and augmented pairs
        # So pair i (from the first half) will be similar to the pair N/2 + i from the second half
        labels = torch.arange(batch_size, device=z.device)
        labels[:batch_size // 2] += batch_size // 2 # For all elements in the first half - add N
        labels[batch_size // 2:] -= batch_size // 2 # For all elements in the second half - subtract N


        return self.criterion(s_matrix, labels)
