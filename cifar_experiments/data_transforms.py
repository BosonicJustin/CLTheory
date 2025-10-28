import torch
from torchvision.transforms import v2 as transforms
import numpy as np

JITTER_ID = 1
CROP_ID = 2
FLIP_ID = 3
GRAY_ID = 4
BLUR_ID = 5

def get_color_jitter(s=1.0, p=0.8):
    return transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=p)


def get_random_sized_crop(size=32):
    return transforms.RandomResizedCrop(size=size)


def get_flip():
    return transforms.RandomHorizontalFlip()


def get_random_gray_scale():
    return transforms.RandomGrayscale(p=0.2)


def get_gaussian_blur(size=32):
    return transforms.GaussianBlur(kernel_size=int(0.1 * size))


IDS_TO_TRANSFORMS = {
    JITTER_ID: get_color_jitter(),
    CROP_ID: get_random_sized_crop(),
    FLIP_ID: get_flip(),
    GRAY_ID: get_random_gray_scale(),
    BLUR_ID: get_gaussian_blur(),
}


def get_permuted_transforms():
    ids = np.random.permutation(list(IDS_TO_TRANSFORMS.keys()))

    return transforms.Compose([IDS_TO_TRANSFORMS[id] for id in ids] + [transforms.ToTensor()])


def get_transforms_by_ids(ids):
    return transforms.Compose([IDS_TO_TRANSFORMS[id] for id in ids] + [transforms.ToTensor()])
