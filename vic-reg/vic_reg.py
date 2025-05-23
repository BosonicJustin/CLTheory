import torch
from loss import VicRegLoss
from tqdm import tqdm
import os
from evals.knn_eval import run_knn_eval
import kornia.augmentation as K


class VicReg(torch.nn.Module):
    def __init__(self, model, expander, gamma=1.0, eps=1e-6, lambda_c=25, mu_c=25, nu_c=1):
        super(VicReg, self).__init__()
        self.model = model
        self.expander = expander
        self.gamma = gamma
        self.eps = eps
        self.lambda_c = lambda_c
        self.mu_c = mu_c
        self.nu_c = nu_c
        
        self.loss_fn = VicRegLoss(gamma, eps, lambda_c, mu_c, nu_c)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.expander = self.expander.to(self.device)
        
        # Initialize augmentations
        self.crop_aug = K.RandomResizedCrop(
            size=(224, 224),
            scale=(0.2, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ).to(self.device)
        
        self.color_aug = K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=1.0
        ).to(self.device)
        
    def train(self, train_loader, val_loader=None, epochs=100, lr=0.5, momentum=0.9, weight_decay=1e-4,
              save_every=10, eval_every=10, checkpoint_dir='checkpoints', resume_from=None):
        """
        Train the VicReg model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate
            momentum: Momentum for SGD optimizer
            weight_decay: Weight decay for SGD optimizer
            save_every: Save checkpoint every N epochs
            eval_every: Evaluate on validation set every N epochs
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to checkpoint to resume from
        """
        self.model.train()
        self.expander.train()
        
        optimizer = torch.optim.SGD(
            list(self.model.parameters()) + list(self.expander.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        start_epoch = 0
        losses = []
        epoch_losses = []
        validation_metrics = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if resume_from:
            checkpoint = self._load_checkpoint(resume_from)
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint.get('losses', [])
            epoch_losses = checkpoint.get('epoch_losses', [])
            validation_metrics = checkpoint.get('validation_metrics', [])
        
        print(f"Using device: {self.device}")
        print("Starting VicReg training")
        
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                x, _ = batch
                x = x.to(self.device)
                
                # Create two augmented views
                x_crop = self.crop_aug(x)
                x_color = self.color_aug(x)
                
                # Get encodings for both views
                z_crop = self.model(x_crop)
                z_color = self.model(x_color)
                
                # Expand both encodings
                z_crop_expanded = self.expander(z_crop)
                z_color_expanded = self.expander(z_color)
                
                # Compute loss between the two views
                loss = self.loss_fn(z_crop_expanded, z_color_expanded)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record loss
                loss_value = loss.item()
                losses.append(loss_value)
                epoch_loss += loss_value
                batch_count += 1
                
                pbar.set_postfix({'loss': f"{loss_value:.4f}"})
            
            # End of epoch
            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)
            scheduler.step()
            
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                self._save_checkpoint(epoch + 1, optimizer, losses, epoch_losses, validation_metrics, checkpoint_dir)
            
            # Evaluate on validation set if provided
            if val_loader and (epoch + 1) % eval_every == 0:
                metrics = self.validate(train_loader, val_loader)
                validation_metrics.append(metrics)
        
        self.model.eval()
        self.expander.eval()
        return self.model, {
            'batch_losses': losses, 
            'epoch_losses': epoch_losses, 
            'validation_metrics': validation_metrics
        }
    
    def validate(self, train_loader, val_loader):
        """
        Comprehensive validation function that evaluates both VicReg loss and KNN accuracy.
        
        Args:
            train_loader: DataLoader for training data (used for KNN evaluation)
            val_loader: DataLoader for validation data
            
        Returns:
            dict: Dictionary containing validation metrics
        """
        self.model.eval()
        self.expander.eval()
        
        # Evaluate VicReg loss
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                x = x.to(self.device)
                
                z = self.model(x)
                z_expanded = self.expander(z)
                loss = self.loss_fn(z_expanded)
                
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        # Run KNN evaluation
        knn_acc = run_knn_eval(self.model, train_loader, val_loader, self.device)
        
        # Print metrics
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"KNN Accuracy: {knn_acc:.2f}%")
        
        # Return to training mode
        self.model.train()
        self.expander.train()
        
        return {
            'vicreg_loss': avg_loss,
            'knn_accuracy': knn_acc
        }
    
    def _save_checkpoint(self, epoch, optimizer, losses, epoch_losses, validation_metrics, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, f"vicreg_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'expander_state': self.expander.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'losses': losses,
            'epoch_losses': epoch_losses,
            'validation_metrics': validation_metrics
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    def _load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.expander.load_state_dict(checkpoint['expander_state'])
        return checkpoint

