import torchvision.models as models
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import pickle

from image_transforms.image_loader_transform import LoaderTransformImagePermutation

from simclr.simclr import SimCLRImages
from transformations.image_permutations import ImagePermutationTransform

# Create a sample image to determine dimensions for the permutation transform
temp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
temp_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=temp_transform)
sample_images, _ = next(iter(DataLoader(temp_dataset, batch_size=1)))
sample_image = sample_images[0]
image_height, image_width = sample_image.shape[-2], sample_image.shape[-1]

# Create the permutation transform
perm_transform = ImagePermutationTransform(image_height, image_width)

# Save the permutation transformation to file for later reuse
os.makedirs('./saved_transforms', exist_ok=True)
with open('./saved_transforms/image_permutation.pkl', 'wb') as f:
    pickle.dump(perm_transform, f)
print(f"Permutation transform saved to ./saved_transforms/image_permutation.pkl")

# Transformation pipeline with permutation included
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
    transforms.ToTensor(),
    LoaderTransformImagePermutation(perm_transform)  # Apply the permutation as part of the transform pipeline
])

# Load CIFAR-10 dataset with permutation transform
trainset = CIFAR10(root='./datasets', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

trainset_test = CIFAR10(root='./datasets', train=False, download=True, transform=transform)
test_loader = DataLoader(trainset_test, batch_size=256, shuffle=True, num_workers=2)

# Initialize ResNet50 model
resnet50 = models.resnet50(weights=None)
resnet50.fc = nn.Identity()

# Train the model with Identity as isomorphism (since permutation is now in transforms)
trained_resnet50_identity, losses = SimCLRImages(
    resnet50,
    train_loader,
    224,
    224,
    isomorphism=nn.Identity(),  # Using Identity as isomorphism since permutation is in transforms
    epochs=100,
    temperature=0.5,
    save_every=10,
    checkpoint_dir='./checkpoints_isomorphic_training_permutation_cifar10_sgd_run1',
    val_dataset=test_loader,
    eval_every=10
).train()