import torch

class InfoNceLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(InfoNceLoss, self).__init__()
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


class SimCLR(torch.nn.Module):
    def __init__(self, encoder, decoder, sample_pair, sample_uniform, temperature):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sample_pair = sample_pair
        self.sample_uniform = sample_uniform

        self.loss = InfoNceLoss(temperature)

    def training_step(self, z_enc, z_enc_sim, z_enc_neg, optimizer):
        optimizer.zero_grad()
        loss_result = self.loss(z_enc, z_enc_sim, z_enc_neg)
        loss_result.backward()
        optimizer.step()

        return loss_result.item()


    def train(self, batch_size, iterations):
        for p in self.decoder.parameters():
            p.requires_grad = False

        adam_normal = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.encoder.train()

        h = lambda latent : self.encoder(self.decoder(latent))

        for i in range(iterations):
            z, z_sim = self.sample_pair(batch_size)
            z_neg =  self.sample_uniform(batch_size)

            z_enc = h(z)
            z_enc_sim = h(z_sim)
            z_enc_neg = h(z_neg)

            loss_result = self.training_step(z_enc, z_enc_sim, z_enc_neg, adam_normal)

            if i % 250 == 1:
                print('Loss:', loss_result, 'Samples processed:', i)

        self.encoder.eval()

        return self.encoder