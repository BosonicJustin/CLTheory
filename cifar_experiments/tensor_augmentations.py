"""
Tensor-based augmentation functions for Adjusted InfoNCE loss.

These augmentation functions work on tensor batches (B, C, H, W) using
torchvision.transforms.v2 which supports tensor inputs.

Matches the augmentation modes from data_transforms.py:
- 'all': All augmentations (jitter, crop, flip, gray, blur, cutout)
- 'crop': Only spatial augmentations (crop, cutout)
- 'all-no-crop': All except spatial (jitter, flip, gray, blur)
"""

import torch
from torchvision.transforms import v2


def get_tensor_augmentation_fn(mode='all'):
    """
    Get augmentation function that works on tensor batches.

    Args:
        mode: 'all' for all augmentations,
              'crop' for only crop and cutout,
              'all-no-crop' for all except crop and cutout

    Returns:
        Function that takes (B, C, H, W) tensor and returns augmented (B, C, H, W) tensor
    """
    if mode == 'all':
        # All augmentations - matches data_transforms.py
        transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=32, scale=(0.08, 1.0), antialias=True),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])

    elif mode == 'crop':
        # Only spatial augmentations (crop and cutout)
        transform = v2.Compose([
            v2.RandomResizedCrop(size=32, scale=(0.08, 1.0), antialias=True),
            v2.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])

    elif mode == 'all-no-crop':
        # All augmentations except crop and cutout
        transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    else:
        raise ValueError(f"Unknown augmentation mode: {mode}. Choose 'all', 'crop', or 'all-no-crop'")

    def augment(images):
        """
        Apply stochastic augmentation to a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W)

        Returns:
            Augmented tensor of shape (B, C, H, W)
        """
        # Apply transform to each image individually to ensure
        # independent random augmentations per image
        augmented = []
        for i in range(images.shape[0]):
            augmented.append(transform(images[i]))
        return torch.stack(augmented)

    return augment


def get_tensor_augmentation_fn_batched(mode='all'):
    """
    Get augmentation function that applies same random transform to entire batch.

    This is faster but all images in batch get same augmentation.
    Use get_tensor_augmentation_fn() for independent augmentations.

    Args:
        mode: 'all', 'crop', or 'all-no-crop'

    Returns:
        Function that takes (B, C, H, W) tensor and returns augmented (B, C, H, W) tensor
    """
    if mode == 'all':
        transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=32, scale=(0.08, 1.0), antialias=True),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    elif mode == 'crop':
        transform = v2.Compose([
            v2.RandomResizedCrop(size=32, scale=(0.08, 1.0), antialias=True),
            v2.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    elif mode == 'all-no-crop':
        transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")

    return transform
