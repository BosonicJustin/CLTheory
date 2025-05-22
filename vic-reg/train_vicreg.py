import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os

from vic_reg import VicReg
from models import create_vicreg_components


def get_cifar10_dataloaders(batch_size=256, num_workers=2):
    """
    Create train and test dataloaders for CIFAR10.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Create datasets
    train_dataset = CIFAR10(root='./cifar_10_data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./cifar_10_data', train=False, download=True, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create checkpoint directory
    checkpoint_dir = './checkpoints/vicreg_cifar10'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, test_loader = get_cifar10_dataloaders()
    
    # Create model components
    encoder, expander = create_vicreg_components(pretrained=False)
    
    # Create VicReg model
    vicreg = VicReg(
        model=encoder,
        expander=expander,
    )
    
    # Training parameters
    epochs = 100
    lr = 0.5
    momentum = 0.9
    weight_decay = 1e-4
    save_every = 10
    eval_every = 5
    
    print("Starting VicReg training on CIFAR10...")
    
    # Train the model
    trained_model, history = vicreg.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        save_every=save_every,
        eval_every=eval_every,
        checkpoint_dir=checkpoint_dir
    )
    
    # Save final model and training history
    torch.save({
        'model_state': trained_model.state_dict(),
        'expander_state': vicreg.expander.state_dict(),
        'history': history
    }, os.path.join(checkpoint_dir, 'vicreg_final.pt'))
    
    print("Training completed!")
    print(f"Final checkpoint saved at {os.path.join(checkpoint_dir, 'vicreg_final.pt')}")
    
    # Print final metrics
    final_metrics = history['validation_metrics'][-1]
    print("\nFinal Validation Metrics:")
    print(f"VicReg Loss: {final_metrics['vicreg_loss']:.4f}")
    print(f"KNN Accuracy: {final_metrics['knn_accuracy']:.2f}%")


if __name__ == "__main__":
    main() 