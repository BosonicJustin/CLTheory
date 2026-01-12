"""
Dataset classes for brain atlas data.
"""

import torch
from torch.utils.data import Dataset
import anndata as ad


class BrainAtlasDataset(Dataset):
    """Dataset for loading brain atlas gene expression data."""
    
    def __init__(self, path_to_file: str):
        """
        Args:
            path_to_file: path to h5ad file
        """
        self.adata = ad.read_h5ad(path_to_file, backed="r")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        """
        Returns:
            dict with 'idx' and 'features' keys
        """
        # Load gene expression features from log2p layer
        features = self.adata.layers['log2p'][idx]
        
        return dict(
            idx=idx,
            features=torch.tensor(features.toarray(), dtype=torch.float32).squeeze(0)
        )


def collate_as_list(batch):
    """
    Collate function that returns batch as a list.
    This maintains compatibility with preprocessing.
    """
    return batch