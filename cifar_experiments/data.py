import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


def download_cifar10(train=True, root="./data"):
    """
    Download the CIFAR10 dataset using torchvision.datasets.CIFAR10.

    Args:
        train (bool): If True, downloads the training set, otherwise the test set.
        root (str): Directory where the dataset will be saved.

    Returns:
        torchvision.datasets.CIFAR10: The downloaded CIFAR10 dataset.
    """

    dataset = CIFAR10(root=root, train=train, download=True)
    return dataset


class CIFAR10PairDataset(Dataset):
    """
    CIFAR10 dataset that returns pairs of augmented views.

    Args:
        root: Directory where the dataset will be saved.
        train: If True, creates dataset from training set, otherwise from test set.
        transform1: First transformation to apply (for first view)
        transform2: Second transformation to apply (for second view)
        download: If True, downloads the dataset if not already present.
    """
    def __init__(self, root="./data", train=True, transform1=None, transform2=None, download=True):
        self.dataset = CIFAR10(root=root, train=train, download=download)
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # We don't need labels for SimCLR

        # Apply first transformation
        if self.transform1 is not None:
            img1 = self.transform1(img)
        else:
            img1 = img

        # Apply second transformation
        if self.transform2 is not None:
            img2 = self.transform2(img)
        else:
            img2 = img

        return img1, img2


def get_cifar10_dataloader(root="./data", train=True, transform1=None, transform2=None,
                           batch_size=256, shuffle=True, num_workers=4, download=True):
    """
    Create a DataLoader for CIFAR10 that returns pairs of augmented views.

    Args:
        root: Directory where the dataset will be saved.
        train: If True, creates dataset from training set, otherwise from test set.
        transform1: First transformation to apply (for first view)
        transform2: Second transformation to apply (for second view)
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        download: If True, downloads the dataset if not already present.

    Returns:
        DataLoader that yields (batch1, batch2) tuples where each is (B, 3, H, W)
    """
    dataset = CIFAR10PairDataset(
        root=root,
        train=train,
        transform1=transform1,
        transform2=transform2,
        download=download
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for consistent batch sizes
    )

    return dataloader

