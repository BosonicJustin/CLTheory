from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm
import json
import os

class LinearProbeEvaluator(nn.Module):
    def __init__(self, encoder, embedding_dim, num_classes, device=None, save_dir=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Freeze encoder weights
        self.encoder = encoder.to(self.device).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Projects from the latent space into the classification space
        self.classifier = nn.Linear(embedding_dim, num_classes).to(self.device)
        # Loss for classification
        self.loss_fn = nn.CrossEntropyLoss()

        # Optimizer for training the linear encoder
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)

        # History tracking
        self.save_dir = save_dir
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.best_val_acc = 0.0

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 10, eval_freq: int = 1):
        """
        Train the linear probe classifier.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of training epochs
            eval_freq: Evaluate on val_loader every N epochs (default: 1 = every epoch)

        Returns:
            float: Final validation accuracy (or None if no validation)
        """
        self.classifier.train()
        print("Starting linear probe training...")

        final_val_acc = None

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)

                # Making sure no gradient is calculated here
                with torch.no_grad():
                    z = self.encoder(x)
                    z = nn.functional.normalize(z, dim=1)

                # Get logits
                preds = self.classifier(z)

                # Compute the cross-entropy loss on the logits
                loss = self.loss_fn(preds, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(correct / total) * 100:.2f}%'})

            avg_loss = total_loss / len(train_loader)
            train_acc = correct / total * 100
            print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}, Train Accuracy = {train_acc:.2f}%")

            # Track training metrics
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(train_acc)

            # Periodic evaluation
            val_acc = None
            if val_loader and ((epoch + 1) % eval_freq == 0 or (epoch + 1) == epochs):
                val_acc = self.evaluate(val_loader)
                final_val_acc = val_acc  # Keep track of latest validation

                # Track best validation accuracy
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    print(f"  → New best validation accuracy: {self.best_val_acc:.2f}%")

            # Track validation accuracy (None for epochs without evaluation)
            self.history['val_acc'].append(val_acc)

        # Save history to file
        if self.save_dir:
            self.save_history()

        print(f"\n{'='*60}")
        print(f"Training completed!")
        if final_val_acc is not None:
            print(f"Final validation accuracy: {final_val_acc:.2f}%")
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")

        return final_val_acc

    def evaluate(self, dataloader: DataLoader):
        """
        Evaluate the linear probe on a validation/test set.

        Args:
            dataloader: DataLoader for evaluation data

        Returns:
            float: Accuracy percentage
        """
        self.classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Evaluating')
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                z = self.encoder(x)
                z = nn.functional.normalize(z, dim=1)
                preds = self.classifier(z)
                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)

                # Update progress bar
                pbar.set_postfix({'acc': f'{(correct / total) * 100:.2f}%'})

        acc = correct / total * 100
        print(f"[Validation] Accuracy: {acc:.2f}%")
        self.classifier.train()
        return acc

    def save_history(self):
        """
        Save training history to JSON file.
        Saves both raw history and final summary.
        """
        if not self.save_dir:
            return

        os.makedirs(self.save_dir, exist_ok=True)

        # Save full history
        history_path = os.path.join(self.save_dir, 'linear_probe_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"History saved to: {history_path}")

        # Save summary with final metrics
        summary = {
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None,
            'final_val_acc': [v for v in self.history['val_acc'] if v is not None][-1] if any(self.history['val_acc']) else None,
            'best_val_acc': self.best_val_acc,
            'total_epochs': len(self.history['epochs'])
        }

        summary_path = os.path.join(self.save_dir, 'linear_probe_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Summary saved to: {summary_path}")