from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split, DataLoader


def load_cifar_10(transforms=transforms.Compose([transforms.ToTensor()]), split_validation=False, add_test=False, batch_size=64, num_workers=4):
    train_set = CIFAR10(root='./datasets', train=True, download=True, transform=transforms)

    datasets = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    }

    if split_validation:
        total_size = len(train_set)
        train_size = int(0.8 * total_size)
        val_size = int(0.2 * total_size)

        train_set, val_set = random_split(train_set, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        datasets['val'] = val_loader
        datasets['train'] = train_loader

    if add_test:
        test_set = CIFAR10(root='./datasets', train=False, download=True, transform=transforms)
        datasets['test'] = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return datasets