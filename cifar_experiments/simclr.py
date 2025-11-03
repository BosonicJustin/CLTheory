import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
from tqdm import tqdm


class SimCLRTrainer:
    """
    Trainer class for SimCLR (without projection head)

    Args:
        model: Encoder model (e.g., ResNet, ViT)
        temperature: Temperature parameter for NT-Xent loss
        dataloader: DataLoader that returns (img1, img2) - two augmented views
        save_dir: Directory to save checkpoints and logs
        device: Device to train on
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        config: Additional configuration dictionary
    """
    def __init__(
        self,
        model,
        temperature,
        dataloader,
        save_dir,
        device='cuda',
        learning_rate=3e-4,
        weight_decay=1e-4,
        config=None
    ):
        self.model = model.to(device)
        self.temperature = temperature
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.device = device
        self.config = config or {}

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Tensorboard writer
        log_dir = os.path.join(save_dir, 'logs')
        self.writer = SummaryWriter(log_dir)

        # Save config
        self.save_config()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def nt_xent_loss(self, z_i, z_j):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
        Using BB^T method for similarity computation

        Args:
            z_i: Embeddings of first augmented views (B, embedding_dim)
            z_j: Embeddings of second augmented views (B, embedding_dim)

        Returns:
            loss: NT-Xent loss value
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate representations: (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix: BB^T -> (2B, 2B)
        similarity_matrix = torch.mm(z, z.t()) / self.temperature

        # Create mask to remove self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # For each sample i in [0, B), its positive is at i+B
        # For each sample i in [B, 2B), its positive is at i-B
        positive_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=self.device),
            torch.arange(0, batch_size, device=self.device)
        ])

        # Extract positive similarities
        positives = similarity_matrix[torch.arange(2 * batch_size, device=self.device), positive_indices]

        # Compute loss: -log(exp(pos) / sum(exp(all)))
        loss = -torch.mean(positives - torch.logsumexp(similarity_matrix, dim=1))

        return loss

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (img1, img2) in enumerate(pbar):
            # Move to device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            # Forward pass - get embeddings directly from encoder
            z_i = self.model(img1)  # (B, embedding_dim)
            z_j = self.model(img2)  # (B, embedding_dim)

            # Compute loss
            loss = self.nt_xent_loss(z_i, z_j)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1

            self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            self.global_step += 1

        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

        return avg_loss

    def train(self, num_epochs, save_freq=10):
        """
        Train the model for multiple epochs

        Args:
            num_epochs: Number of epochs to train
            save_freq: Save checkpoint every save_freq epochs
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Save directory: {self.save_dir}")
        print(f"Temperature: {self.temperature}")
        print(f"Device: {self.device}")
        print("-" * 80)

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train one epoch
            avg_loss = self.train_epoch(epoch)

            print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

            # Save checkpoint
            if epoch % save_freq == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, avg_loss)

        print("-" * 80)
        print("Training completed!")
        self.writer.close()

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.save_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'global_step': self.global_step,
            'config': self.config
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

        # Also save as 'latest.pt'
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'global_step': self.global_step,
            'config': self.config
        }, latest_path)

    def save_config(self):
        """Save training configuration"""
        config_dict = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': self.temperature,
            'device': str(self.device),
            'optimizer': {
                'type': 'Adam',
                'lr': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0]['weight_decay']
            },
            **self.config
        }

        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

        print(f"Config saved: {config_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")

        return checkpoint['epoch'], checkpoint['loss']
