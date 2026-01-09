"""
Adjusted InfoNCE Loss Implementation

In Adjusted InfoNCE, negatives come from different augmentations of the SAME
source image, rather than from other images in the batch.

This corresponds to sampling negatives from a constrained submanifold K(z)
where "content" (source image identity) is fixed but "style" (augmentation) varies.

Mathematical formulation:
                          exp(z_a · z_p / τ)
L = -log ─────────────────────────────────────────────────
          exp(z_a · z_p / τ) + Σᵢ exp(z_a · z_nᵢ / τ)

Where:
- z_a = anchor embedding (unaugmented)
- z_p = positive embedding (augmented view of same image)
- z_nᵢ = negative embeddings (M different augmented views of same image)
- τ = temperature parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjustedInfoNCELoss(nn.Module):
    """
    Adjusted InfoNCE loss where negatives come from different augmentations
    of the SAME source image (not different images).

    Args:
        temperature: Temperature parameter τ for scaling similarities (default: 0.5)
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Compute Adjusted InfoNCE loss.

        Args:
            anchor: Anchor embeddings (B, D) - from unaugmented images
            positive: Positive embeddings (B, D) - from augmented images
            negatives: Negative embeddings (M, B, D) - from M different augmentations

        Returns:
            loss: Scalar tensor, mean loss over batch
        """
        batch_size = anchor.shape[0]
        device = anchor.device

        # L2-normalize all embeddings
        anchor = F.normalize(anchor, dim=1)       # (B, D)
        positive = F.normalize(positive, dim=1)   # (B, D)
        negatives = F.normalize(negatives, dim=2) # (M, B, D)

        # Compute positive similarity: anchor · positive
        # (B, D) * (B, D) -> (B,) after sum
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # (B,)

        # Compute negative similarities: anchor · each negative
        # anchor: (B, D), negatives: (M, B, D)
        # Result: (B, M)
        neg_sim = torch.einsum('bd,mbd->bm', anchor, negatives) / self.temperature  # (B, M)

        # Concatenate positive and negative similarities
        # Positive is at index 0
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1 + M)

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class AdjustedInfoNCELossIntegrated(nn.Module):
    """
    Adjusted InfoNCE loss that handles augmentation and encoding internally.

    This version takes raw images and applies augmentation internally,
    matching the interface described in the theoretical framework.

    Args:
        temperature: Temperature parameter τ for scaling similarities
        num_negatives: Number of negative augmentations M per image
    """

    def __init__(self, temperature=0.5, num_negatives=512):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives

    def forward(self, encoder, images, augmentation_fn):
        """
        Compute Adjusted InfoNCE loss with integrated augmentation.

        Args:
            encoder: Neural network that maps images to embeddings
            images: Batch of source images (B, C, H, W) - already ToTensor'd but unaugmented
            augmentation_fn: Stochastic augmentation function that takes
                            (B, C, H, W) and returns (B, C, H, W)

        Returns:
            loss: Scalar tensor, mean loss over batch
        """
        batch_size = images.shape[0]
        device = images.device

        # Anchor: encode unaugmented images
        anchor = encoder(images)  # (B, D)

        # Generate M+1 augmented views (1 positive + M negatives)
        all_views = []
        for _ in range(self.num_negatives + 1):
            augmented = augmentation_fn(images)  # (B, C, H, W)
            all_views.append(augmented)

        # Stack: (num_views, B, C, H, W)
        all_views = torch.stack(all_views, dim=0)
        num_views = all_views.shape[0]

        # Reshape and encode all views
        # (num_views, B, C, H, W) -> (num_views * B, C, H, W)
        all_views_flat = all_views.view(-1, *images.shape[1:])

        # Encode all views
        embeddings = encoder(all_views_flat)  # (num_views * B, D)

        # Reshape: (num_views * B, D) -> (num_views, B, D)
        embed_dim = embeddings.shape[1]
        embeddings = embeddings.view(num_views, batch_size, embed_dim)

        # Split into positive and negatives
        positive = embeddings[0]      # (B, D)
        negatives = embeddings[1:]    # (M, B, D)

        # L2-normalize all embeddings
        anchor = F.normalize(anchor, dim=1)       # (B, D)
        positive = F.normalize(positive, dim=1)   # (B, D)
        negatives = F.normalize(negatives, dim=2) # (M, B, D)

        # Compute positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # (B,)

        # Compute negative similarities
        neg_sim = torch.einsum('bd,mbd->bm', anchor, negatives) / self.temperature  # (B, M)

        # Concatenate: positive at index 0
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1 + M)

        # Labels: positive is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
