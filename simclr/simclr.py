import torch
import kornia.augmentation as K
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import os

from evals.disentanglement import linear_disentanglement, permutation_disentanglement
from evals.knn_eval import run_knn_eval
from evals.distance_preservation import calculate_angle_preservation_error

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
        
        total_loss = (loss_pos + loss_neg).mean()
        
        # Split for monitoring
        pos_component = loss_pos.mean()
        neg_component = loss_neg.mean()

        return pos_component.detach().item(), neg_component.detach().item(), total_loss


class SimCLR(torch.nn.Module):
    def __init__(self, encoder, decoder, sample_pair, sample_uniform, temperature, device=torch.device('cpu')):
        super(SimCLR, self).__init__()

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.sample_pair = sample_pair
        self.sample_uniform = sample_uniform
        self.loss = InfoNceLoss(temperature)
        self.device = device

    def training_step(self, z_enc, z_enc_sim, z_enc_neg, optimizer):
        optimizer.zero_grad()
        pos_loss, neg_loss, total_loss = self.loss(z_enc, z_enc_sim, z_enc_neg)
        total_loss.backward()
        optimizer.step()

        return pos_loss, neg_loss, total_loss.item()

    def train(self, batch_size, iterations):
        # Freeze decoder
        for p in self.decoder.parameters():
            p.requires_grad = False

        adam = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.encoder.train()

        # Compose h = encoder ∘ decoder
        h = lambda latent: self.encoder(self.decoder(latent.to(self.device)))

        # --- Evaluation on latent space before training
        control_latent = self.sample_uniform(batch_size).to(self.device)

        linear = linear_disentanglement(control_latent, control_latent)
        print("Linear control score:", linear[0][0])

        perm = permutation_disentanglement(control_latent, control_latent, mode="pearson", solver="munkres")
        print("Permutation control score:", perm[0][0])

        linear_scores = []
        perm_scores = []
        distance_preservation_errors = []
        eval_losses = []
        eval_pos_losses = []
        eval_neg_losses = []

        for i in range(iterations):
            z, z_sim = self.sample_pair(batch_size)
            z_neg = self.sample_uniform(batch_size)

            z = z.to(self.device)
            z_sim = z_sim.to(self.device)
            z_neg = z_neg.to(self.device)

            z_enc = h(z)
            z_enc_sim = h(z_sim)
            z_enc_neg = h(z_neg)

            pos_loss, neg_loss, loss_result = self.training_step(z_enc, z_enc_sim, z_enc_neg, adam)

            if i % 20 == 1:
                lin_dis, _ = linear_disentanglement(z.cpu(), z_enc.cpu())
                lin_score, _ = lin_dis

                perm_dis, _ = permutation_disentanglement(z.cpu(), z_enc.cpu(), mode="pearson", solver="munkres")
                perm_score, _ = perm_dis

                distance_preservation_error = calculate_angle_preservation_error(z.cpu(), z_enc.cpu())

                linear_scores.append(lin_score)
                perm_scores.append(perm_score)
                distance_preservation_errors.append(distance_preservation_error)
                eval_losses.append(loss_result)
                eval_pos_losses.append(pos_loss)
                eval_neg_losses.append(neg_loss)

                print('Loss:', loss_result, 'Pos Loss:', pos_loss, 'Neg Loss:', neg_loss, 'Samples processed:', i,
                      "linear disentanglement:", lin_score,
                      'permutation disentanglement:', perm_score, 'angle_preservation_error:', distance_preservation_error)

        self.encoder.eval()
        return self.encoder, {
            'linear_scores': linear_scores, 
            'perm_scores': perm_scores, 
            'angle_preservation_errors': distance_preservation_errors, 
            'eval_losses': eval_losses,
            'eval_pos_losses': eval_pos_losses,
            'eval_neg_losses': eval_neg_losses
        }

class SimCLRImages(nn.Module):
    def __init__(self, encoder, training_dataset, image_h, image_w,
                 isomorphism=None, epochs=10, temperature=0.5,
                 checkpoint_dir='checkpoints',
                 resume_from=None,
                 save_every=10,
                 eval_every=10,
                 val_dataset=None,
                 permutation_transform=None,
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.training_dataset = training_dataset
        self.isomorphism = isomorphism if isomorphism is not None else nn.Identity()
        self.permutation_transform = permutation_transform
        if self.permutation_transform is not None:
            self.permutation_transform = self.permutation_transform.to(self.device)
        self.loss_fn = torch.nn.functional.cross_entropy
        self.normalize = lambda z: torch.nn.functional.normalize(z, p=2.0, dim=-1, eps=1e-12)
        self.optimizer = torch.optim.SGD(self.encoder.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
        self.T1 = K.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=1).to(self.device) # TODO: the problem might be here - the
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
        if self.permutation_transform is not None:
            print("🔄 Permutation will be applied after augmentations")
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
        
        # Apply permutation after augmentations if specified
        if self.permutation_transform is not None:
            x = self.permutation_transform(x)
            x_sim = self.permutation_transform(x_sim)
        
        x_all = torch.cat((x, x_sim), dim=0)
        z_all = self.encoder(x_all)
        l = self.loss(z_all)
        l.backward()
        self.optimizer.step()

        return l.item()

    def evaluate_knn(self, epoch):
        self.encoder.eval()

        acc = run_knn_eval(self.encoder, self.training_dataset, self.val_dataset, self.device)

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

            if self.val_dataset and (epoch + 1) % self.eval_every == 0:
                self.evaluate_knn(epoch)

        self.encoder.eval()
        return self.encoder, {'batch_losses': self.losses, 'epoch_losses': self.epoch_losses}


class InfoNceLossAdjusted(nn.Module):
    def __init__(self, temperature=0.1):
        """
        InfoNCE Loss where each sample has dedicated negative samples.
        
        Args:
            temperature (float): Temperature parameter for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchors, positives, negatives):
        """
        Compute InfoNCE loss with dedicated negatives for each sample.
        
        Args:
            anchors: Tensor of shape (N, d) - anchor embeddings
            positives: Tensor of shape (N, d) - positive embeddings  
            negatives: Tensor of shape (N, M, d) - negative embeddings
                      M negatives for each of the N samples
        
        Returns:
            tuple: (pos_component, neg_component, total_loss)
        """
        N, d = anchors.shape
        N_pos, d_pos = positives.shape
        N_neg, _, d_neg = negatives.shape
        
        # Validate input shapes
        assert N == N_pos == N_neg, f"Batch sizes must match: {N}, {N_pos}, {N_neg}"
        assert d == d_pos == d_neg, f"Embedding dimensions must match: {d}, {d_pos}, {d_neg}"
        
        # Compute similarities
        # Positive similarities: (N,)
        pos_similarities = torch.sum(anchors * positives, dim=1) / self.temperature
        
        # Negative similarities: (N, M)
        # For each anchor, compute similarity with its M dedicated negatives
        neg_similarities = torch.bmm(
            anchors.unsqueeze(1),  # (N, 1, d)
            negatives.transpose(1, 2)  # (N, d, M)
        ).squeeze(1) / self.temperature  # (N, M)
        
        # Concatenate positive and negative similarities for each sample
        # Shape: (N, 1 + M)
        all_similarities = torch.cat([
            pos_similarities.unsqueeze(1),  # (N, 1)
            neg_similarities  # (N, M)
        ], dim=1)
        
        # Compute InfoNCE loss
        targets = torch.zeros(N, dtype=torch.long, device=anchors.device)
        total_loss = F.cross_entropy(all_similarities, targets)
        
        # Split for monitoring - similar to original InfoNceLoss
        pos_component = (-pos_similarities).mean()  # Negative of positive similarities
        neg_component = torch.logsumexp(all_similarities, dim=1).mean()  # Log sum exp of all similarities
        
        return pos_component.detach().item(), neg_component.detach().item(), total_loss


class SimCLRAdjusted(nn.Module):
    def __init__(self, neg_samples, decoder, encoder, sample_pair, sample_negative, tau=0.1, device=None):
        super(SimCLRAdjusted, self).__init__()

        self.neg_samples = neg_samples
        self.tau = tau
        self.decoder = decoder
        self.encoder = encoder
        self.sample_pair = sample_pair
        self.sample_negative = sample_negative

        self.loss_fn = InfoNceLossAdjusted(tau)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.decoder.to(self.device)
        self.encoder.to(self.device)

    def train(self, batch_size, iterations):
        for p in self.decoder.parameters():
            p.requires_grad = False

        adam = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.encoder.train()
        h = lambda latent: self.encoder(self.decoder(latent.to(self.device)))

        control_latent, _ = self.sample_pair(batch_size)
        control_latent = control_latent.to(self.device)

        linear = linear_disentanglement(control_latent, control_latent)
        print("Linear control score:", linear[0][0])

        perm = permutation_disentanglement(control_latent, control_latent, mode="pearson", solver="munkres")
        print("Permutation control score:", perm[0][0])

        linear_scores = []
        perm_scores = []
        distance_preservation_errors = []
        eval_losses = []
        eval_pos_losses = []
        eval_neg_losses = []

        for i in range(iterations):
            adam.zero_grad()
            
            z, z_sim = self.sample_pair(batch_size)
            z = z.to(self.device)
            z_sim = z_sim.to(self.device)
            z_neg = self.sample_negative(z, self.neg_samples).to(self.device)
            
            z_enc = h(z)
            z_enc_sim = h(z_sim)

            z_neg_reshaped = z_neg.view(-1, z_neg.size(-1)) # (N * M, d)
            z_enc_neg = h(z_neg_reshaped) 
            z_enc_neg = z_enc_neg.view(z_neg.size(0), z_neg.size(1), -1) # (N, M, d)

            pos_loss, neg_loss, total_loss = self.loss_fn(z_enc, z_enc_sim, z_enc_neg)

            total_loss.backward()
            adam.step()
            
            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

            if i % 20 == 1:
                with torch.no_grad():  # Ensure no gradients during evaluation
                    lin_dis, _ = linear_disentanglement(z.cpu(), z_enc.cpu())
                    lin_score, _ = lin_dis

                    perm_dis, _ = permutation_disentanglement(z.cpu(), z_enc.cpu(), mode="pearson", solver="munkres")
                    perm_score, _ = perm_dis

                    distance_preservation_error = calculate_angle_preservation_error(z.cpu(), z_enc.cpu())

                linear_scores.append(lin_score)
                perm_scores.append(perm_score)
                distance_preservation_errors.append(distance_preservation_error)
                eval_losses.append(total_loss.item())
                eval_pos_losses.append(pos_loss)
                eval_neg_losses.append(neg_loss)

                print('Loss:', total_loss.item(), 'Pos Loss:', pos_loss, 'Neg Loss:', neg_loss, 
                    'Samples processed:', i,
                    "linear disentanglement:", lin_score,
                    'permutation disentanglement:', perm_score, 
                    'angle_preservation_error:', distance_preservation_error)

        self.encoder.eval()
        return self.encoder, {
            'linear_scores': linear_scores, 
            'perm_scores': perm_scores, 
            'angle_preservation_errors': distance_preservation_errors, 
            'eval_losses': eval_losses,
            'eval_pos_losses': eval_pos_losses,
            'eval_neg_losses': eval_neg_losses
        }