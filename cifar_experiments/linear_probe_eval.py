import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import json
import os
import glob
import re
from tqdm import tqdm

from models import ViT1x1, get_resnet50_model


class LinearProbe(nn.Module):
    """
    Linear probe for evaluating learned representations.
    Freezes encoder and trains only a linear classifier.
    """
    def __init__(self, encoder, num_classes=10, device='cpu'):
        super().__init__()
        self.encoder = encoder

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 32, 32).to(device)
            encoder_output = encoder(dummy_input)
            self.feature_dim = encoder_output.shape[-1]

        # Linear classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.classifier(features)
        return logits


class LinearProbeValidator:
    """
    Validator class for linear probing evaluation.

    Args:
        checkpoint_path: Path to the model checkpoint
        config_path: Path to the config.json file
        device: Device to run on
        batch_size: Batch size for evaluation
        num_epochs: Number of epochs to train linear probe
        learning_rate: Learning rate for linear probe training
    """
    def __init__(
        self,
        checkpoint_path,
        config_path,
        device='cuda',
        batch_size=256,
        num_epochs=100,
        learning_rate=0.1
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        print(f"Loaded config from {config_path}")
        print(f"Model type: {self.config['model_type']}")

        # Load model
        self.encoder = self._load_encoder()
        self.model = LinearProbe(self.encoder, num_classes=10, device=self.device).to(self.device)

        print(f"Model loaded from {checkpoint_path}")
        print(f"Feature dimension: {self.model.feature_dim}")

        # Setup data
        self.train_loader, self.test_loader = self._setup_data()

        # Setup optimizer and loss
        self.optimizer = optim.SGD(
            self.model.classifier.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        self.criterion = nn.CrossEntropyLoss()

        # History tracking
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        self.best_test_acc = 0.0

    def _load_encoder(self):
        """Load the encoder model from checkpoint"""
        model_type = self.config['model_type']

        # Create encoder
        if model_type == 'vit-1':
            encoder = ViT1x1(
                img_size=32,
                embed_dim=256,
                hidden_dim=512,
                msa_heads=8,
                num_layers=6
            )
        elif model_type == 'cnn':
            encoder = get_resnet50_model()
            encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        encoder.eval()

        return encoder.to(self.device)

    def _setup_data(self):
        """Setup CIFAR-10 train and test dataloaders"""
        # Simple transform: ToTensor only (no normalization, matching SimCLR training)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = CIFAR10(
            root='./data',
            train=True,
            transform=transform_train,
            download=True
        )

        test_dataset = CIFAR10(
            root='./data',
            train=False,
            transform=transform_test,
            download=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, test_loader

    def train_epoch(self):
        """Train linear probe for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'acc': f'{100. * correct / total:.2f}%'
                })

        accuracy = 100. * correct / total
        return accuracy

    def run(self):
        """Run linear probe training and evaluation"""
        print("=" * 80)
        print("Linear Probe Evaluation")
        print("=" * 80)
        print(f"Training epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Save directory: {os.path.dirname(self.checkpoint_path)}")
        print("=" * 80)

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            # Evaluate
            test_acc = self.evaluate()
            print(f"Test Acc: {test_acc:.2f}%")

            # Track history
            self.history['epochs'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)

            # Track best accuracy
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                print(f"  → New best test accuracy: {self.best_test_acc:.2f}%")

        # Save history
        self.save_history()

        print("=" * 80)
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print(f"Best Test Accuracy: {self.best_test_acc:.2f}%")
        print("=" * 80)

        return self.best_test_acc

    def save_history(self):
        """
        Save training history to JSON files.
        Saves both full history and summary.
        """
        save_dir = os.path.dirname(self.checkpoint_path)

        # Save full history
        history_path = os.path.join(save_dir, 'linear_probe_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"\nHistory saved to: {history_path}")

        # Save summary with final metrics
        summary = {
            'checkpoint': os.path.basename(self.checkpoint_path),
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None,
            'final_test_acc': self.history['test_acc'][-1] if self.history['test_acc'] else None,
            'best_test_acc': self.best_test_acc,
            'total_epochs': len(self.history['epochs']),
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

        summary_path = os.path.join(save_dir, 'linear_probe_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Summary saved to: {summary_path}")


def find_latest_checkpoint(run_dir):
    """
    Find the latest checkpoint in the run directory.

    Args:
        run_dir: Directory containing checkpoints

    Returns:
        str: Path to latest checkpoint
    """
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(run_dir, 'checkpoint_epoch_*.pt')
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    # Extract epoch numbers and find the latest
    def get_epoch_num(checkpoint_path):
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', checkpoint_path)
        return int(match.group(1)) if match else -1

    latest_checkpoint = max(checkpoints, key=get_epoch_num)
    epoch_num = get_epoch_num(latest_checkpoint)

    print(f"Found latest checkpoint: checkpoint_epoch_{epoch_num}.pt")
    return latest_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Linear Probe Evaluation for SimCLR')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Path to run directory (e.g., runs/simclr_cnn_20251111_083646)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for evaluation (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train linear probe (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate for linear probe (default: 0.1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Validate run directory exists
    if not os.path.isdir(args.run_dir):
        raise ValueError(f"Run directory does not exist: {args.run_dir}")

    # Find config.json
    config_path = os.path.join(args.run_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {args.run_dir}")

    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint(args.run_dir)

    print("=" * 80)
    print(f"Run directory: {args.run_dir}")
    print(f"Config: {os.path.basename(config_path)}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print("=" * 80)

    # Run linear probe
    validator = LinearProbeValidator(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )

    best_acc = validator.run()

    print(f"\nFinal Result: {best_acc:.2f}% test accuracy")


if __name__ == '__main__':
    main()
