"""
Training script for contrastive learning on brain atlas data.

Usage:
    python train_contrastive.py --data atlas_brain_638850_CCF.h5ad --pairs pairs_output --output checkpoints
"""

import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import dataset from separate file
from dataset import BrainAtlasDataset, collate_as_list


# Configure logging
def setup_logging(output_dir):
    """Setup logging to both file and console."""
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """InfoNCE (NT-Xent) loss for contrastive learning."""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: embeddings of first view [N, D]
            z_j: embeddings of second view [N, D]
        Returns:
            loss: scalar
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)  # [2N, D]
        similarity_matrix = torch.matmul(representations, representations.T)  # [2N, 2N]
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Compute logits
        logits = similarity_matrix / self.temperature
        
        # Create labels for cross entropy
        # For each sample i in first N samples, positive is at position i + N
        # For each sample i in last N samples, positive is at position i - N
        positive_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_i.device),
            torch.arange(0, batch_size, device=z_i.device)
        ], dim=0)
        
        # Compute loss using cross entropy
        loss = F.cross_entropy(logits, positive_indices)
        
        return loss


def load_pairs(pairs_dir):
    """Load preprocessed pairs and configuration."""
    pairs_dir = Path(pairs_dir)
    
    with open(pairs_dir / "pairs.pkl", 'rb') as f:
        pairs = pickle.load(f)
    
    with open(pairs_dir / "config.pkl", 'rb') as f:
        config = pickle.load(f)
    
    with open(pairs_dir / "batch_metadata.pkl", 'rb') as f:
        batch_metadata = pickle.load(f)
    
    return pairs, config, batch_metadata


def organize_pairs_by_batch(pairs, batch_metadata):
    """Organize pairs by batch index for efficient lookup."""
    pairs_per_batch = []
    for meta in batch_metadata:
        start_idx = meta['pair_start_idx']
        end_idx = meta['pair_end_idx']
        batch_pairs = pairs[start_idx:end_idx]
        pairs_per_batch.append(batch_pairs)
    return pairs_per_batch


def train_epoch(model, loader, pairs_per_batch, criterion, optimizer, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Extract features from batch
        features = torch.stack([b['features'] for b in batch]).to(device)
        indices = [b['idx'] for b in batch]
        
        # Get corresponding pairs for this batch
        batch_pairs = pairs_per_batch[batch_idx]
        
        # Create index mapping from global idx to batch position
        idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}
        
        # Extract paired samples
        pairs_i = []
        pairs_j = []
        for idx_i, idx_j in batch_pairs:
            if idx_i in idx_to_pos and idx_j in idx_to_pos:
                pairs_i.append(idx_to_pos[idx_i])
                pairs_j.append(idx_to_pos[idx_j])
            else:
                raise ValueError(f"Pair {idx_i}, {idx_j} not found in batch {batch_idx}")
        
        # Get features for paired samples
        features_i = features[pairs_i]  # [N, D]
        features_j = features[pairs_j]  # [N, D]
        

        print("features", features_i.shape)
        # Forward pass
        embeddings_i = model(features_i)
        embeddings_j = model(features_j)
        
        # Compute loss
        loss = criterion(embeddings_i, embeddings_j)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, output_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save latest checkpoint
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
    
    return checkpoint_path


def train(args):
    """Main training function."""
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    writer = SummaryWriter(output_dir / "tensorboard")
    
    # Log arguments
    logger.info("="*80)
    logger.info("Training Configuration")
    logger.info("="*80)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load pairs and configuration
    logger.info("Loading preprocessed pairs...")
    pairs, preprocess_config, batch_metadata = load_pairs(args.pairs)
    logger.info(f"Loaded {len(pairs)} pairs from {len(batch_metadata)} batches")
    
    # Organize pairs by batch
    pairs_per_batch = organize_pairs_by_batch(pairs, batch_metadata)
    
    # Verify batch size matches
    if args.batch_size != preprocess_config['batch_size']:
        raise ValueError(
            f"Training batch size ({args.batch_size}) must match "
            f"preprocessing batch size ({preprocess_config['batch_size']})"
        )
    
    # Load dataset (must match preprocessing exactly)
    logger.info("Loading dataset...")
    dataset = BrainAtlasDataset(args.data)
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # Get input dimension from first sample
    sample = dataset[0]
    input_dim = sample['features'].shape[0]
    logger.info(f"Input dimension: {input_dim}")
    
    # Create dataloader (MUST match preprocessing)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # CRITICAL: must be False to match preprocessing
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,  # CRITICAL: must be true to match preprocessing
        collate_fn=collate_as_list
    )
    
    logger.info(f"Number of batches: {len(loader)}")
    
    # Import and create model
    logger.info("Creating FeedForwardEncoder model...")
    from encoders import FeedForwardEncoder
    
    model = FeedForwardEncoder(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.embedding_dim
    ).to(device)
    
    logger.info(f"Model architecture:")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Hidden dims: {args.hidden_dims}")
    logger.info(f"  Output dim: {args.embedding_dim}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs
    )
    
    # Training loop
    best_loss = float('inf')
    
    logger.info("="*80)
    logger.info("Starting training")
    logger.info("="*80)
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        avg_loss = train_epoch(
            model, loader, pairs_per_batch, criterion, optimizer, device, epoch, logger
        )
        
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            logger.info(f"New best loss: {best_loss:.4f}")
        
        if epoch % args.save_every == 0 or is_best:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, avg_loss, output_dir, is_best=is_best
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("="*80)
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info("="*80)
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train contrastive learning model on brain atlas data"
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--pairs', type=str, required=True,
                        help='Path to preprocessed pairs directory')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', 
                        default=[1024, 512, 256],
                        help='Hidden layer dimensions')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=20000,
                        help='Batch size (must match preprocessing)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for InfoNCE loss')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()