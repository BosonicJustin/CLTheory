import torch
import torch.nn as nn


class OrthogonalityRegularizer(nn.Module):
    def __init__(self, latent_dimension):
        super().__init__()
        self.d = latent_dimension
        self.register_buffer('I', torch.eye(latent_dimension))

    def forward(self, z):
        return (z @ z.T - self.I).norm(p=2)


class FitEncoderToMatrix(nn.Module):
    def __init__(self, latent_dimension, encoder, orthogonality_regularizer=False, device='cpu'):
        super().__init__()

        self.encoder = encoder
        self.d = latent_dimension
        self.device = device
        
        # Register matrix as a parameter so it gets optimized
        self.m = nn.Parameter(torch.randn(latent_dimension, latent_dimension, device=device))

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
    
        if orthogonality_regularizer:
            self.ortho_reg = OrthogonalityRegularizer(latent_dimension)
            self.ortho_lambda = 0.1  # regularization strength
        else:
            self.ortho_reg = lambda z: 0
            self.ortho_lambda = 0

        # Optimizer should optimize the matrix parameter, not encoder
        self.optimizer = torch.optim.Adam([self.m], lr=0.001)
        self.loss_fn = nn.MSELoss()

    def compute_loss(self, z):
        """Compute the regression loss with optional orthogonality regularization."""
        z_rec = self.encoder(z)
        target = (self.m @ z.T).T  # Equivalent to z @ self.m.T but more efficient
        
        mse_loss = self.loss_fn(z_rec, target)
        ortho_loss = self.ortho_reg(self.m)
        
        total_loss = mse_loss + self.ortho_lambda * ortho_loss
        return total_loss, mse_loss, ortho_loss

    def train(self, batch_size, iterations, train_sampler, val_sampler=None, log_every=100):
        """
        Fit the matrix regression model.
        
        Args:
            batch_size: Size of training batches
            iterations: Number of training iterations
            train_sampler: Function that returns training samples
            val_sampler: Optional function that returns validation samples
            log_every: Log progress every N iterations
        
        Returns:
            Dictionary with training and validation losses
        """
        train_losses = []
        train_mse_losses = []
        train_ortho_losses = []
        val_losses = []
        val_mse_losses = []
        val_ortho_losses = []
        
        for i in range(iterations):
            # Training step
            super().train()  # Set module to training mode
            z_train = train_sampler(batch_size).to(self.device)
            
            total_loss, mse_loss, ortho_loss = self.compute_loss(z_train)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            train_losses.append(total_loss.item())
            train_mse_losses.append(mse_loss.item())
            train_ortho_losses.append(ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else 0.0)
            
            # Validation step
            if val_sampler is not None:
                self.eval()
                with torch.no_grad():
                    z_val = val_sampler(batch_size).to(self.device)
                    val_total_loss, val_mse_loss, val_ortho_loss = self.compute_loss(z_val)
                    
                    val_losses.append(val_total_loss.item())
                    val_mse_losses.append(val_mse_loss.item())
                    val_ortho_losses.append(val_ortho_loss.item() if isinstance(val_ortho_loss, torch.Tensor) else 0.0)
            
            # Logging
            if (i + 1) % log_every == 0:
                print(f"Iteration {i+1}/{iterations}")
                print(f"  Train - Total: {total_loss.item():.6f}, MSE: {mse_loss.item():.6f}, Ortho: {train_ortho_losses[-1]:.6f}")
                if val_sampler is not None:
                    print(f"  Val   - Total: {val_losses[-1]:.6f}, MSE: {val_mse_losses[-1]:.6f}, Ortho: {val_ortho_losses[-1]:.6f}")
        
        results = {
            'train_losses': train_losses,
            'train_mse_losses': train_mse_losses,
            'train_ortho_losses': train_ortho_losses,
        }
        
        if val_sampler is not None:
            results.update({
                'val_losses': val_losses,
                'val_mse_losses': val_mse_losses,
                'val_ortho_losses': val_ortho_losses,
            })
        
        return results

    def get_learned_matrix(self):
        """Return the learned transformation matrix."""
        return self.m.detach().cpu()

    def forward(self, z):
        """Apply the learned transformation to input z."""
        return (self.m @ z.T).T




if __name__ == "__main__":
    """Test script for the FitEncoderToMatrix class."""
    import torch
    import numpy as np
    from pathlib import Path
    
    # Test parameters
    batch_size = 1024
    iterations = 1000
    log_every = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing FitEncoderToMatrix on device: {device}")
    
    # Create a random target matrix for testing
    target_matrix = torch.randn(3, 3, device=device)
    target_matrix = target_matrix / torch.norm(target_matrix, dim=0, keepdim=True)  # Normalize columns
    
    # Create a simple data sampler that generates random 3D vectors
    def sample_data(batch_size):
        return torch.randn(batch_size, 3, device=device)
    
    # Create validation sampler
    def val_sampler(batch_size):
        return torch.randn(batch_size, 3, device=device)
    
    # Create a dummy encoder (identity function for testing)
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
        def forward(self, x):
            return self.linear(x)
    
    dummy_encoder = DummyEncoder().to(device)
    
    # Initialize the fitter
    fitter = FitEncoderToMatrix(
        latent_dimension=3,
        encoder=dummy_encoder,
        orthogonality_regularizer=True,
        device=device
    )
    
    print(f"Target matrix shape: {target_matrix.shape}")
    print(f"Target matrix condition number: {torch.linalg.cond(target_matrix):.4f}")
    
    # Train the model
    print("\nStarting training...")
    results = fitter.train(
        batch_size=batch_size,
        iterations=iterations, 
        train_sampler=sample_data,
        val_sampler=val_sampler,
        log_every=log_every
    )
    
    # Get the learned matrix
    learned_matrix = fitter.get_learned_matrix()
    print(f"\nLearned matrix shape: {learned_matrix.shape}")
    print(f"Learned matrix condition number: {torch.linalg.cond(learned_matrix):.4f}")
    
    # Test forward pass
    test_input = torch.randn(10, 3, device=device)
    test_output = fitter(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    
    # Compute final metrics
    mse_error = torch.mean((learned_matrix - target_matrix) ** 2)
    ortho_error = torch.norm(learned_matrix.T @ learned_matrix - torch.eye(3, device=device))
    
    print(f"\n=== Final Results ===")
    print(f"MSE between learned and target matrix: {mse_error:.6f}")
    print(f"Orthogonality error: {ortho_error:.6f}")
    print(f"Final training loss: {results['train_losses'][-1]:.6f}")
    if 'val_losses' in results:
        print(f"Final validation loss: {results['val_losses'][-1]:.6f}")
    
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training losses
        axes[0, 0].plot(results['train_losses'])
        axes[0, 0].set_title('Training Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        
        axes[0, 1].plot(results['train_mse_losses'])
        axes[0, 1].set_title('Training MSE Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('MSE Loss')
        
        axes[1, 0].plot(results['train_ortho_losses'])
        axes[1, 0].set_title('Training Orthogonality Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Ortho Loss')
        
        if 'val_losses' in results:
            axes[1, 1].plot(results['val_losses'], label='Validation')
            axes[1, 1].plot(results['train_losses'], label='Training')
            axes[1, 1].set_title('Training vs Validation Loss')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
        else:
            axes[1, 1].plot(results['train_losses'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('fit_encoder_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Training curves saved to 'fit_encoder_test_results.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    print("\n=== Test Summary ===")
    print("Test completed successfully!")
    print(f"Final MSE error: {torch.mean((learned_matrix - target_matrix) ** 2):.6f}")
