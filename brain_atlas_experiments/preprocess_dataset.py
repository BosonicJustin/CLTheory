"""
Preprocessing script to generate and save matched pairs for contrastive learning.

Usage:
    python preprocess_pairs.py --input atlas_brain_638850_CCF.h5ad --output pairs_output --batch_size 20000
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from datetime import datetime

import anndata as ad
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Configure logging
def setup_logging(output_dir):
    """Setup logging to both file and console."""
    log_file = output_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class BrainAtlasDataset(Dataset):
    def __init__(self, path_to_file: str):
        self.adata = ad.read_h5ad(path_to_file, backed="r")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        c_x, c_y = self.adata.obs['center_x'][idx], self.adata.obs['center_y'][idx]

        return dict(
            idx=idx,
            position=torch.tensor([c_x.item(), c_y.item()])
        )


def collate_as_list(batch):
    """Collate function that returns batch as a list."""
    return batch


@torch.no_grad()
def match_pairs(batch_, batch_size: int, logger):
    """
    Match pairs within a batch using global greedy assignment.
    
    Args:
        batch_: list of dicts with 'position' and 'idx' keys
        batch_size: size of the batch (must be even)
        logger: logger instance for progress tracking
    
    Returns:
        list[tuple[int, int]]: pairs of (idx_from_partition1, idx_from_partition2)
    """
    assert len(batch_) == batch_size, (len(batch_), batch_size)
    assert batch_size % 2 == 0, "Batch size must be even"

    # Stack positions and extract indices
    pos = torch.stack([b["position"] for b in batch_], dim=0)
    idxs = [int(b["idx"]) for b in batch_]

    half = batch_size // 2
    p1 = pos[:half]          # First partition
    p2 = pos[half:]          # Second partition
    idxs1 = idxs[:half]
    idxs2 = idxs[half:]

    logger.info(f"  Computing pairwise distances for {half} x {half} points")
    dists = torch.cdist(p1, p2, p=2)

    # Global greedy 1-1 assignment
    H = half
    flat = dists.reshape(-1)
    order = torch.argsort(flat)

    logger.info("  Performing greedy matching")
    used_i = [False] * H
    used_j = [False] * H
    pairs = []

    for k in order.tolist():
        i = k // H
        j = k % H

        if not used_i[i] and not used_j[j]:
            used_i[i] = True
            used_j[j] = True
            pairs.append((idxs1[i], idxs2[j]))

            if len(pairs) == H:
                break

    logger.info(f"  Matched {len(pairs)} pairs")
    return pairs


def preprocess_dataset(input_path, output_dir, batch_size, num_workers=0):
    """
    Preprocess the dataset and save matched pairs for each batch.
    
    Args:
        input_path: path to the h5ad file
        output_dir: directory to save the preprocessed pairs
        batch_size: batch size for processing
        num_workers: number of worker processes for data loading
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info("Starting preprocessing")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("="*80)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = BrainAtlasDataset(input_path)
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_as_list
    )
    
    total_batches = len(loader)
    logger.info(f"Total batches: {total_batches}")
    logger.info(f"Samples per batch: {batch_size}")
    logger.info(f"Total samples to process: {total_batches * batch_size}")
    
    # Process batches
    all_pairs = []
    batch_metadata = []
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches")):
        logger.info(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
        
        pairs = match_pairs(batch, batch_size, logger)
        all_pairs.extend(pairs)
        
        # Store metadata for this batch
        batch_info = {
            'batch_idx': batch_idx,
            'batch_size': batch_size,
            'num_pairs': len(pairs),
            'pair_start_idx': len(all_pairs) - len(pairs),
            'pair_end_idx': len(all_pairs)
        }
        batch_metadata.append(batch_info)
        
        logger.info(f"  Total pairs so far: {len(all_pairs)}")
    
    # Save all pairs
    pairs_file = output_dir / "pairs.pkl"
    logger.info(f"\nSaving {len(all_pairs)} pairs to {pairs_file}")
    with open(pairs_file, 'wb') as f:
        pickle.dump(all_pairs, f)
    
    # Save batch metadata
    metadata_file = output_dir / "batch_metadata.pkl"
    logger.info(f"Saving batch metadata to {metadata_file}")
    with open(metadata_file, 'wb') as f:
        pickle.dump(batch_metadata, f)
    
    # Save processing configuration
    config = {
        'input_path': str(input_path),
        'batch_size': batch_size,
        'total_batches': total_batches,
        'total_pairs': len(all_pairs),
        'dataset_size': len(dataset),
        'num_workers': num_workers,
        'timestamp': datetime.now().isoformat()
    }
    config_file = output_dir / "config.pkl"
    logger.info(f"Saving configuration to {config_file}")
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)
    
    logger.info("="*80)
    logger.info("Preprocessing complete!")
    logger.info(f"Total pairs generated: {len(all_pairs)}")
    logger.info(f"Output files:")
    logger.info(f"  - Pairs: {pairs_file}")
    logger.info(f"  - Metadata: {metadata_file}")
    logger.info(f"  - Config: {config_file}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess brain atlas dataset for contrastive learning"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input h5ad file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pairs_output',
        help='Output directory for preprocessed pairs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20000,
        help='Batch size (must be even)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loading workers'
    )
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch_size % 2 != 0:
        raise ValueError("Batch size must be even")
    
    # Run preprocessing
    preprocess_dataset(
        input_path=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()