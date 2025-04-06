from torch import nn, DataLoader
import torch
import tqdm


class LinearProbeEvaluator(nn.Module):
    def __init__(self, encoder, embedding_dim, num_classes, device=None):
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

    def train_probe(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 10):
        self.classifier.train()
        print("Starting linear probe training...")

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
            acc = correct / total * 100
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")

            if val_loader:
                self.evaluate(val_loader)

    def evaluate(self, dataloader: DataLoader):
        self.classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                z = self.encoder(x)
                preds = self.classifier(z)
                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        acc = correct / total * 100
        print(f"[Validation] Accuracy: {acc:.2f}%")
        self.classifier.train()
        return acc