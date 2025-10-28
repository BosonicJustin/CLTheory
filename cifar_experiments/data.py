import torch
from torchvision.datasets import CIFAR10


# TODO: ADD TRANSFORMS
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


print(torch.transforms.v2.ToTensor(download_cifar10(train=True)[0]))