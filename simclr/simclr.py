import torch
import kornia.augmentation as K
from tqdm import tqdm
from torch import nn
import os

from evals.disentanglement import linear_disentanglement, permutation_disentanglement
from evals.knn_eval import run_knn_eval

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

        adam = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.encoder.train()

        h = lambda latent : self.encoder(self.decoder(latent))

        control_latent = self.sample_uniform(batch_size)
        linear = linear_disentanglement(control_latent, control_latent)

        print("Linear control score:", linear[0][0])

        perm = permutation_disentanglement(control_latent, control_latent)

        print("Permutation control score:", perm[0][0])

        for i in range(iterations):
            z, z_sim = self.sample_pair(batch_size)
            z_neg = self.sample_uniform(batch_size)

            z_enc = h(z)
            z_enc_sim = h(z_sim)
            z_enc_neg = h(z_neg)

            loss_result = self.training_step(z_enc, z_enc_sim, z_enc_neg, adam)

            if i % 250 == 1:
                lin_dis, _ = linear_disentanglement(z, z_enc)
                lin_score, _ = lin_dis

                perm_dis, _ = permutation_disentanglement(z, z_enc)
                perm_score, _ = perm_dis

                print('Loss:', loss_result, 'Samples processed:', i, "linear disentanglement:", lin_score, 'permutation disentanglement:', perm_score)

        self.encoder.eval()

        return self.encoder

class SimCLRImages(nn.Module):
    def __init__(self, encoder, training_dataset, image_h, image_w,
                 isomorphism=None, epochs=10, temperature=0.5,
                 checkpoint_dir='checkpoints',
                 resume_from=None,
                 save_every=10,
                 eval_every=10,
                 val_dataset=None,
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.training_dataset = training_dataset
        self.isomorphism = isomorphism if isomorphism is not None else nn.Identity()
        self.loss_fn = torch.nn.functional.cross_entropy
        self.normalize = lambda z: torch.nn.functional.normalize(z, p=2.0, dim=-1, eps=1e-12)
        self.optimizer = torch.optim.SGD(self.encoder.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
        self.T1 = K.RandomRotation(degrees=30, p=1.0).to(self.device)
        self.T2 = K.RandomResizedCrop((image_h, image_w), scale=(0.2, 0.8), p=1.0).to(self.device)
        self.epochs = epochs
        self.temperature = temperature
        self.checkpoint_dir = checkpoint_dir
        self.resume_from = resume_from
        self.save_every = save_every
        self.eval_every = eval_every
        self.val_dataset = val_dataset

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.start_epoch = 0
        self.losses = []
        self.epoch_losses = []

        print(f"Using device: {self.device}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.resume_from:
            self._load_checkpoint(self.resume_from)

    def _save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"encoder_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'encoder_state': self.encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'losses': self.losses,
            'epoch_losses': self.epoch_losses
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def _load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.losses = checkpoint.get('losses', [])
        self.epoch_losses = checkpoint.get('epoch_losses', [])

    def loss(self, embeddings):
        N = embeddings.shape[0] // 2
        z_norm = self.normalize(embeddings)
        sim_matrix = z_norm @ z_norm.T / self.temperature

        mask = torch.eye(embeddings.shape[0], device=embeddings.device).bool()
        sim_matrix.masked_fill_(mask, -10e15)

        gt = torch.cat((torch.arange(N, embeddings.shape[0]), torch.arange(0, N)), dim=0).to(embeddings.device)

        return self.loss_fn(sim_matrix, gt)

    def training_step(self, batch):
        self.optimizer.zero_grad()
        x = self.T1(batch)
        x_sim = self.T2(batch)
        x_all = torch.cat((x, x_sim), dim=0)
        z_all = self.encoder(x_all)
        l = self.loss(z_all)
        l.backward()
        self.optimizer.step()

        return l.item()

    def evaluate_knn(self, epoch):
        self.encoder.eval()

        acc = run_knn_eval(self.encoder, self.training_dataset, self.val_loader, self.device)

        print(f"🔍 [k-NN Eval] Epoch {epoch + 1}, Accuracy: {acc:.2f}%")
        self.encoder.train()

    def train(self):
        self.encoder.train()
        print("🚀 Starting SimCLR training")

        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss = 0.0
            batch_count = 0

            pbar = tqdm(self.training_dataset, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                images, _ = batch
                images = images.to(self.device)
                loss = self.training_step(self.isomorphism(images))
                self.losses.append(loss)
                epoch_loss += loss
                batch_count += 1
                pbar.set_postfix({'loss': f"{loss:.4f}"})

            avg_epoch_loss = epoch_loss / batch_count
            self.epoch_losses.append(avg_epoch_loss)
            self.scheduler.step()
            print(f"📉 Epoch {epoch + 1}/{self.epochs}, Avg Loss: {avg_epoch_loss:.4f}")

            if (epoch + 1) % self.save_every == 0 or (epoch + 1) == self.epochs:
                self._save_checkpoint(epoch + 1)

            if self.val_loader and (epoch + 1) % self.eval_every == 0:
                self.evaluate_knn(epoch)

        self.encoder.eval()
        return self.encoder, {'batch_losses': self.losses, 'epoch_losses': self.epoch_losses}