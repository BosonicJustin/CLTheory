"""
Adjusted InfoNCE Training Script for CIFAR-10

This script trains encoders using Adjusted InfoNCE loss, where negatives
come from different augmentations of the SAME source image rather than
other images in the batch.

Usage:
    python experiments_adjusted.py --model resnet --aug-mode all --num-negatives 512

This is the parallel equivalent of experiments.py for the Adjusted InfoNCE variant.
"""

import argparse
import torch
from datetime import datetime
import os

from models import get_resnet18_encoder, get_vit_encoder, MLPEncoder, DEFAULT_EMBED_DIM
from data import get_cifar10_single_dataloader, get_cifar10_eval_dataloaders
from adjusted_simclr import AdjustedSimCLRTrainer
from tensor_augmentations import get_tensor_augmentation_fn


def get_model(model_type):
    """
    Get the encoder model based on type.

    All encoders output DEFAULT_EMBED_DIM (512) dimensional embeddings for fair comparison.

    Args:
        model_type: 'resnet' for ResNet18, 'vit' for ViT with 4x4 patches, or 'mlp' for MLP

    Returns:
        encoder model
    """
    if model_type == 'resnet':
        print(f"Initializing ResNet18 encoder (output_dim={DEFAULT_EMBED_DIM})...")
        model = get_resnet18_encoder(output_dim=DEFAULT_EMBED_DIM)
        return model

    elif model_type == 'vit':
        print(f"Initializing ViT encoder with 4x4 patches (output_dim={DEFAULT_EMBED_DIM})...")
        model = get_vit_encoder(output_dim=DEFAULT_EMBED_DIM)
        return model

    elif model_type == 'mlp':
        print(f"Initializing MLP encoder (output_dim={DEFAULT_EMBED_DIM})...")
        model = MLPEncoder(output_dim=DEFAULT_EMBED_DIM)
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'resnet', 'vit', or 'mlp'")


def main():
    parser = argparse.ArgumentParser(description='Adjusted InfoNCE Training on CIFAR-10')
    parser.add_argument('--model', type=str, choices=['resnet', 'vit', 'mlp'], required=True,
                       help='Model architecture: resnet (ResNet18), vit (ViT with 4x4 patches), or mlp (MLP)')
    parser.add_argument('--aug-mode', type=str, choices=['all', 'crop', 'all-no-crop'], default='all',
                       help='Augmentation mode: all (all transforms), crop (only crop+cutout), or all-no-crop (default: all)')
    parser.add_argument('--num-negatives', type=int, default=512,
                       help='Number of negative augmentations M per image (default: 512)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature parameter for loss (default: 0.5)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Save checkpoint every N epochs (default: 50)')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for CIFAR-10 data (default: ./data)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save checkpoints and logs (default: auto-generated)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training (default: cuda)')
    parser.add_argument('--val-freq', type=int, default=10,
                       help='Run KNN validation every N epochs (default: 10)')
    parser.add_argument('--val-batch-size', type=int, default=256,
                       help='Batch size for validation (default: 256)')
    parser.add_argument('--no-validation', action='store_true',
                       help='Disable validation during training')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f'./runs/adjusted_infonce_{args.model}_{timestamp}'
    os.makedirs(args.save_dir, exist_ok=True)

    # Calculate memory usage estimate
    forward_passes = (args.num_negatives + 2) * args.batch_size
    standard_forward_passes = 2 * 2000  # Standard SimCLR baseline

    print("=" * 80)
    print("Adjusted InfoNCE Training Configuration")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Augmentation mode: {args.aug_mode}")
    print(f"Num negatives (M): {args.num_negatives}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Save directory: {args.save_dir}")
    print(f"Forward passes per batch: {forward_passes:,} ({forward_passes/standard_forward_passes:.1f}x vs standard)")
    if not args.no_validation:
        print(f"Validation: Enabled (every {args.val_freq} epochs)")
        print(f"Validation batch size: {args.val_batch_size}")
    else:
        print("Validation: Disabled")
    print("=" * 80)

    # Get model
    model = get_model(args.model)
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get augmentation function (works on tensor batches)
    augmentation_fn = get_tensor_augmentation_fn(mode=args.aug_mode)
    print(f"Augmentation function: {args.aug_mode} mode (tensor-based)")

    # Get dataloader (single view - unaugmented)
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_single_dataloader(
        root=args.data_root,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        download=True
    )
    print(f"Dataset loaded. Number of batches: {len(dataloader)}")

    # Get validation dataloaders
    val_dataloader_train = None
    val_dataloader_test = None
    if not args.no_validation:
        print("Loading validation datasets...")
        val_dataloader_train, val_dataloader_test = get_cifar10_eval_dataloaders(
            root=args.data_root,
            batch_size=args.val_batch_size,
            num_workers=1,
            download=True
        )
        print(f"Validation datasets loaded. Train: {len(val_dataloader_train)} batches, Test: {len(val_dataloader_test)} batches")

    # Create config dictionary
    config = {
        'model_type': args.model,
        'augmentation_mode': args.aug_mode,
        'num_negatives': args.num_negatives,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'temperature': args.temperature,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'save_freq': args.save_freq,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'num_batches_per_epoch': len(dataloader),
        'loss_type': 'adjusted_infonce'
    }

    # Initialize trainer
    trainer = AdjustedSimCLRTrainer(
        model=model,
        augmentation_fn=augmentation_fn,
        num_negatives=args.num_negatives,
        temperature=args.temperature,
        dataloader=dataloader,
        save_dir=args.save_dir,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        config=config,
        val_dataloader_train=val_dataloader_train,
        val_dataloader_test=val_dataloader_test,
        val_freq=args.val_freq
    )

    # Train
    trainer.train(num_epochs=args.epochs, save_freq=args.save_freq)

    print("Training completed successfully!")
    print(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
