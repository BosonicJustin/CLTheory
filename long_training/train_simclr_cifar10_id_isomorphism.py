import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import torchvision.models as models

from torch import nn

from simclr.simclr import SimCLRImages

# Transformation: resize + convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
    transforms.ToTensor()
])

# Load CIFAR-10 dataset
trainset = CIFAR10(root='./datasets', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=120, shuffle=True, num_workers=2)

resnet50 = models.resnet50(pretrained=False)

#  (self, encoder, training_dataset, image_h, image_w,
#                  isomorphism=None, epochs=10, temperature=0.5,
#                  checkpoint_dir='checkpoints', resume_from=None, save_every=5):
trained_resnet50_identity, losses = SimCLRImages(resnet50, train_loader, 224, 224, isomorphism=nn.Identity(), epochs=400, temperature=0.5, save_every=5).train()
