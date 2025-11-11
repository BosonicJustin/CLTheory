import argparse
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms
from datetime import datetime
import os

from models import ViT1x1, get_resnet50_model, MLPEncoder
from data import get_cifar10_dataloader, get_cifar10_eval_dataloaders
from simclr import SimCLRTrainer
from data_transforms import (
    get_permuted_transforms,
    get_transforms_by_ids,
    IDS_TO_TRANSFORMS,
    CROP_ID,
    CUTOUT_ID
)


def get_base_transform():
    """
    Get base transform (no augmentation, only ToTensor).
    This is for the first batch (unaugmented).
    """
    return transforms.Compose([transforms.ToTensor()])


def get_augmentation_transform(mode='all'):
    """
    Get augmentation pipeline based on mode.
    This is for the second batch (augmented).

    Args:
        mode: 'all' for all augmentations (randomly permuted order),
              'crop' for only crop and cutout,
              'all-no-crop' for all augmentations except crop and cutout

    Returns:
        Transform composition
    """
    if mode == 'all':
        # All augmentations in random permuted order
        return get_permuted_transforms()

    elif mode == 'crop':
        # Only crop and cutout
        return get_transforms_by_ids([CROP_ID, CUTOUT_ID])

    elif mode == 'all-no-crop':
        # All augmentations except crop and cutout
        excluded_ids = {CROP_ID, CUTOUT_ID}
        ids = [id for id in IDS_TO_TRANSFORMS.keys() if id not in excluded_ids]
        return get_transforms_by_ids(ids)

    else:
        raise ValueError(f"Unknown augmentation mode: {mode}. Choose 'all', 'crop', or 'all-no-crop'")


def get_model(model_type):
    """
    Get the encoder model based on type.

    Args:
        model_type: 'cnn' for ResNet50, 'vit-1' for ViT with 1x1 patches, or 'mlp' for Multi-Layer Perceptron

    Returns:
        encoder model
    """
    if model_type == 'cnn':
        print("Initializing ResNet50 encoder...")
        model = get_resnet50_model()
        # Remove the classification head, keep only the feature extractor
        model.fc = nn.Identity()
        return model

    elif model_type == 'vit-1':
        print("Initializing ViT-1x1 encoder...")
        model = ViT1x1(
            img_size=32,
            embed_dim=256,
            hidden_dim=512,
            msa_heads=8,
            num_layers=6
        )
        return model

    elif model_type == 'mlp':
        print("Initializing MLP encoder...")
        model = MLPEncoder(
            hidden_dim=2048,
            num_hidden_layers=3,
            output_dim=256,
            dropout=0.1
        )
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn', 'vit-1', or 'mlp'")


def main():
    parser = argparse.ArgumentParser(description='SimCLR Training on CIFAR-10')
    parser.add_argument('--model', type=str, choices=['cnn', 'vit-1', 'mlp'], required=True,
                       help='Model architecture: cnn (ResNet50), vit-1 (ViT with 1x1 patches), or mlp (Multi-Layer Perceptron)')
    parser.add_argument('--aug-mode', type=str, choices=['all', 'crop', 'all-no-crop'], default='all',
                       help='Augmentation mode: all (all transforms permuted), crop (only crop+cutout), or all-no-crop (all except crop+cutout) (default: all)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=2000,
                       help='Batch size for training (default: 2000)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature parameter for NT-Xent loss (default: 0.5)')
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
        args.save_dir = f'./runs/simclr_{args.model}_{timestamp}'
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("SimCLR Training Configuration")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Augmentation mode: {args.aug_mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Save directory: {args.save_dir}")
    if not args.no_validation:
        print(f"Validation: Enabled (every {args.val_freq} epochs)")
        print(f"Validation batch size: {args.val_batch_size}")
    else:
        print("Validation: Disabled")
    print("=" * 80)

    # Get model
    model = get_model(args.model)
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get transforms
    # First batch: unaugmented (only ToTensor)
    # Second batch: augmented based on mode
    transform1 = get_base_transform()
    transform2 = get_augmentation_transform(mode=args.aug_mode)

    print(f"Transform 1 (unaugmented): ToTensor only")
    print(f"Transform 2 (augmented): {args.aug_mode} mode")

    # Get dataloader
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_dataloader(
        root=args.data_root,
        train=True,
        transform1=transform1,
        transform2=transform2,
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
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'temperature': args.temperature,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'save_freq': args.save_freq,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'num_batches_per_epoch': len(dataloader)
    }

    # Initialize trainer
    trainer = SimCLRTrainer(
        model=model,
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
