"""
Adjusted SimCLR Trainer

Trainer class for SimCLR with Adjusted InfoNCE loss, where negatives come
from different augmentations of the SAME source image rather than other
images in the batch.

Key differences from standard SimCLR:
- Uses single-view dataloader (images are unaugmented)
- Applies augmentation on-the-fly to generate M+1 views (1 positive + M negatives)
- Anchor remains unaugmented
- Uses AdjustedInfoNCELoss instead of NT-Xent

Optimizations:
- Uses kornia for GPU-parallel augmentations
- Batches all M+1 views through encoder in chunks for efficiency
- Supports mixed precision (AMP) training
"""

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import os
import json
import sys
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path to import from evals
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals.knn_eval import run_knn_eval

from adjusted_infonce import AdjustedInfoNCELoss


class AdjustedSimCLRTrainer:
    """
    Trainer class for SimCLR with Adjusted InfoNCE loss.

    Args:
        model: Encoder model (e.g., ResNet, ViT)
        augmentation_fn: Function that applies stochastic augmentation to tensor batches
        num_negatives: Number of negative augmentations M per image
        temperature: Temperature parameter for loss
        dataloader: DataLoader that returns (images, labels) - single view
        save_dir: Directory to save checkpoints and logs
        device: Device to train on
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        config: Additional configuration dictionary
        val_dataloader_train: DataLoader for training set validation
        val_dataloader_test: DataLoader for test set validation
        val_freq: Frequency of validation (every N epochs)
    """

    def __init__(
        self,
        model,
        augmentation_fn,
        num_negatives,
        temperature,
        dataloader,
        save_dir,
        device='cuda',
        learning_rate=3e-4,
        weight_decay=1e-4,
        config=None,
        val_dataloader_train=None,
        val_dataloader_test=None,
        val_freq=10,
        use_amp=True,
        encoder_chunk_size=2048
    ):
        self.config = config or {}
        self.model = model.to(device)

        # Use channels-last memory format for ResNet (faster convolutions)
        # Not beneficial for MLP (no convs) or ViT (attention-based)
        if self.config.get('model_type') == 'resnet':
            self.model = self.model.to(memory_format=torch.channels_last)

        self.augmentation_fn = augmentation_fn
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.device = device
        self.encoder_chunk_size = encoder_chunk_size

        # Validation parameters
        self.val_dataloader_train = val_dataloader_train
        self.val_dataloader_test = val_dataloader_test
        self.val_freq = val_freq

        # Mixed precision training
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Initialize loss function
        self.criterion = AdjustedInfoNCELoss(temperature=temperature)

        # Optimizer - use Adam with foreach for speed
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            foreach=True
        )

        # Tensorboard writer
        log_dir = os.path.join(save_dir, 'logs')
        self.writer = SummaryWriter(log_dir)

        # Save config
        self.save_config()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def _encode_chunked(self, images):
        """
        Encode images through the model in chunks to manage memory.

        Args:
            images: Tensor of shape (N, C, H, W)

        Returns:
            Embeddings of shape (N, D)
        """
        N = images.shape[0]
        chunk_size = self.encoder_chunk_size

        if N <= chunk_size:
            with autocast('cuda', enabled=self.use_amp):
                return self.model(images)

        # Process in chunks
        embeddings = []
        for i in range(0, N, chunk_size):
            chunk = images[i:i + chunk_size]
            with autocast('cuda', enabled=self.use_amp):
                emb = self.model(chunk)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)

    def train_epoch(self, epoch):
        """Train for one epoch using Adjusted InfoNCE loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            # Move to device - images are unaugmented (just ToTensor)
            images = images.to(self.device)
            B = images.shape[0]
            M = self.num_negatives

            # Anchor: encode unaugmented images
            with autocast('cuda', enabled=self.use_amp):
                anchor = self.model(images)  # (B, D)

            # Generate all M+1 augmented views at once using kornia
            # Stack original images M+1 times and augment in one batch
            images_expanded = images.unsqueeze(0).expand(M + 1, -1, -1, -1, -1)
            images_flat = images_expanded.reshape((M + 1) * B, *images.shape[1:])

            # Apply augmentation to all (M+1)*B images at once on GPU
            with torch.no_grad():
                augmented_flat = self.augmentation_fn(images_flat)

            # Encode all augmented views in chunks
            augmented_embeddings = self._encode_chunked(augmented_flat)  # ((M+1)*B, D)

            # Reshape to (M+1, B, D)
            D = augmented_embeddings.shape[-1]
            augmented_embeddings = augmented_embeddings.reshape(M + 1, B, D)

            # Split into positive and negatives
            positive = augmented_embeddings[0]  # (B, D)
            negatives = augmented_embeddings[1:]  # (M, B, D)

            # Compute loss
            with autocast('cuda', enabled=self.use_amp):
                loss = self.criterion(anchor, positive, negatives)

            # Backward pass with AMP
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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

    def validate(self, epoch):
        """
        Run KNN validation on the current model.

        Args:
            epoch: Current epoch number for logging

        Returns:
            float: KNN accuracy (0-100)
        """
        if self.val_dataloader_train is None or self.val_dataloader_test is None:
            print("Warning: Validation dataloaders not provided, skipping validation")
            return None

        self.model.eval()

        print(f"Running KNN validation...")
        acc = run_knn_eval(
            self.model,
            self.val_dataloader_train,
            self.val_dataloader_test,
            self.device
        )

        # Log to TensorBoard
        self.writer.add_scalar('Validation/knn_accuracy', acc, epoch)

        print(f"[KNN Eval] Epoch {epoch}, Accuracy: {acc:.2f}%")

        self.model.train()
        return acc

    def train(self, num_epochs, save_freq=10, start_epoch=1):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_freq: Save checkpoint every save_freq epochs
            start_epoch: Epoch to start from (for resuming training)
        """
        print(f"Starting Adjusted InfoNCE training for {num_epochs} epochs")
        if start_epoch > 1:
            print(f"Resuming from epoch {start_epoch}")
        print(f"Save directory: {self.save_dir}")
        print(f"Temperature: {self.temperature}")
        print(f"Num negatives (M): {self.num_negatives}")
        print(f"Device: {self.device}")
        print("-" * 80)

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch

            # Train one epoch
            avg_loss = self.train_epoch(epoch)

            print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

            # Run validation
            if self.val_dataloader_test is not None and (epoch % self.val_freq == 0 or epoch == num_epochs):
                self.validate(epoch)

            # Save checkpoint
            if epoch % save_freq == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, avg_loss)

        print("-" * 80)
        print("Training completed!")
        self.writer.close()

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint."""
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
        """Save training configuration."""
        config_dict = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': self.temperature,
            'num_negatives': self.num_negatives,
            'loss_type': 'adjusted_infonce',
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
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")

        return checkpoint['epoch'], checkpoint['loss']
