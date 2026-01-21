"""
Tensor-based augmentation functions for Adjusted InfoNCE loss.

Uses kornia for GPU-accelerated augmentations that apply independent
random transforms to each image in a batch (true parallel augmentation).

Matches the augmentation modes from data_transforms.py:
- 'all': All augmentations (jitter, crop, flip, gray, blur, cutout)
- 'crop': Only crop (spatial transform only)
- 'all-no-crop': All except crop (jitter, flip, gray, blur, cutout)
"""

import torch
import kornia.augmentation as K


def get_tensor_augmentation_fn(mode='all', device='cuda'):
    """
    Get GPU-accelerated augmentation function using kornia.

    Kornia applies independent random transforms to each image in the batch,
    running entirely on GPU for maximum speed.

    Args:
        mode: 'all' for all augmentations,
              'crop' for only crop and cutout,
              'all-no-crop' for all except crop and cutout
        device: Device to run augmentations on

    Returns:
        Function that takes (B, C, H, W) tensor and returns augmented (B, C, H, W) tensor
    """
    if mode == 'all':
        # All augmentations - matches data_transforms.py
        transform = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomResizedCrop(size=(32, 32), scale=(0.08, 1.0)),
            K.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            K.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0, p=1.0),
            same_on_batch=False,  # Independent random augmentations per image
        ).to(device)

    elif mode == 'crop':
        # Only crop (spatial transform only)
        transform = K.AugmentationSequential(
            K.RandomResizedCrop(size=(32, 32), scale=(0.08, 1.0)),
            same_on_batch=False,
        ).to(device)

    elif mode == 'all-no-crop':
        # All augmentations except crop (includes cutout)
        transform = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
            K.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0, p=1.0),
            same_on_batch=False,
        ).to(device)

    else:
        raise ValueError(f"Unknown augmentation mode: {mode}. Choose 'all', 'crop', or 'all-no-crop'")

    def augment(images):
        """
        Apply stochastic augmentation to a batch of images on GPU.

        Args:
            images: Tensor of shape (B, C, H, W) on GPU

        Returns:
            Augmented tensor of shape (B, C, H, W) on GPU
        """
        return transform(images)

    return augment


def get_batch_augmentation_fn(mode='all', num_augmentations=512, device='cuda'):
    """
    Get augmentation function that generates multiple augmented views at once.

    This is optimized for Adjusted InfoNCE where we need M augmentations
    of each image. Instead of calling augment() M times, this generates
    all M views in a single batched operation.

    Args:
        mode: 'all', 'crop', or 'all-no-crop'
        num_augmentations: Number of augmented views M to generate per image
        device: Device to run augmentations on

    Returns:
        Function that takes (B, C, H, W) tensor and returns (M, B, C, H, W) tensor
    """
    single_augment = get_tensor_augmentation_fn(mode, device)

    def augment_batch(images):
        """
        Generate M augmented views for each image in batch.

        Args:
            images: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (M, B, C, H, W) with M augmented views per image
        """
        B, C, H, W = images.shape
        M = num_augmentations

        # Expand images to (M, B, C, H, W) and reshape to (M*B, C, H, W)
        # This way kornia processes all M*B images in parallel
        images_expanded = images.unsqueeze(0).expand(M, -1, -1, -1, -1)
        images_flat = images_expanded.reshape(M * B, C, H, W)

        # Apply augmentation to all M*B images at once
        augmented_flat = single_augment(images_flat)

        # Reshape back to (M, B, C, H, W)
        augmented = augmented_flat.reshape(M, B, C, H, W)

        return augmented

    return augment_batch
