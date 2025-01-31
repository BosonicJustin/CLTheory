import torch


class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature

    # Here I'm expecting an input of shape (2n, d_latent)
    def forward(self, z):
        s_matrix = z @ z.t()  / self.temperature
        batch_size = z.size(0)

        # print(s_matrix.shape)

        # Here we will be summing over the horizontal direction of the matrix (in the exp), hence masking the diagonal
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        s_matrix = s_matrix.masked_fill(mask, -1e11)

        # Here the logic is that we stacked the original and augmented pairs
        # So pair i (from the first half) will be similar to the pair N/2 + i from the second half
        labels = torch.arange(batch_size, device=z.device)
        labels[:batch_size // 2] += batch_size // 2 # For all elements in the first half - add N
        labels[batch_size // 2:] -= batch_size // 2 # For all elements in the second half - subtract N

        # print('labels', labels)
        # print('look', s_matrix[0])
        # print('S_MATRIX SHAPE', s_matrix)

        return self.criterion(s_matrix, labels)
