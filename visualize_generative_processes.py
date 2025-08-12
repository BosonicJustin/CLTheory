import torch
import numpy as np
import matplotlib.pyplot as plt
from data.generation import SphereDecoderIdentity, SphereDecoderLinear, InjectiveLinearDecoder, SpiralRotation, Patches
from invertible_network_utils import construct_invertible_mlp
from visualization_utils.spheres import scatter3d_sphere, visualize_spheres_side_by_side

def sample_uniform_sphere(n_samples=1000, dim=3):
    """Sample uniformly from a sphere surface in dim dimensions."""
    # Sample from standard normal and normalize
    z = torch.randn(n_samples, dim)
    z = z / z.norm(dim=-1, keepdim=True)
    return z

def create_colored_sphere_plot(z, colors, title):
    """Create a sphere plot with colored wireframe like scatter3d_sphere."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Normalize points to sphere surface
    z_norm = z / z.norm(dim=-1, keepdim=True)
    # Apply same color normalization as scatter3d_sphere
    color_norm = 0.5 * colors.detach() + 0.5
    ax.scatter(z_norm[:, 0], z_norm[:, 1], z_norm[:, 2], c=color_norm, s=10, alpha=0.8, zorder=1)
    
    # Add wireframe sphere
    u, v = np.mgrid[-0.25*np.pi:0.75*np.pi:256j, 0:np.pi:256j]
    r = 1.03
    ax.plot_wireframe(r * np.cos(u) * np.sin(v),
                      r * np.sin(u) * np.sin(v),
                      r * np.cos(v), color='gray', alpha=0.4,
                      rstride=32, cstride=64)
    
    # Style like scatter3d_sphere
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_aspect('equal')
    ax.set_xticks([-1., 1.], minor=False)
    ax.set_yticks([-1., 1.], minor=False)
    ax.set_zticks([-1., 1.], minor=False)
    ax.view_init(10, 45)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax

def create_side_by_side_visualization(z, x, colors, name, output_dir, is_sphere_output=True):
    """Create side-by-side visualization."""
    
    # Turn off interactive mode to prevent display
    plt.ioff()
    
    # Non-sphere outputs (Injective Linear, Invertible MLP) - simple scatter plots
    if name in {"Injective Linear", "Invertible MLP"}:
        z_3d = z[:, :3].detach().cpu().numpy()
        x_3d = x[:, :3].detach().cpu().numpy()
        
        n_points = z_3d.shape[0]
        
        # Create maximally distinct colors
        hsv_colors = plt.cm.hsv(np.linspace(0, 1, n_points))
        np.random.seed(42)
        np.random.shuffle(hsv_colors)
        colors_plot = hsv_colors[:, :3]
        
        fig = plt.figure(figsize=(12, 6))
        
        # Original sphere (left)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2],
                    c=colors_plot, s=20, alpha=0.9)
        
        # Transformed data space (right)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(x_3d[:, 0], x_3d[:, 1], x_3d[:, 2],
                    c=colors_plot, s=20, alpha=0.9)
        
        plt.tight_layout()
        
    else:
        # Sphere outputs (Identity, Spiral, Patches) - use coordinates as RGB colors
        fig = plt.figure(figsize=(12, 6))
        
        # Left sphere: scatter3d_sphere(plt, z, z) - plot z coordinates, color by z coordinates
        ax1 = fig.add_subplot(121, projection='3d')
        
        z_detached = z.detach()
        # Ensure proper normalization and handle edge cases
        z_norms = z_detached.norm(dim=-1, keepdim=True)
        z_norms = torch.clamp(z_norms, min=1e-8)  # Avoid division by zero
        z_norm = z_detached / z_norms
        
        # Check for any invalid values and filter them out
        valid_mask = torch.isfinite(z_norm).all(dim=-1)
        z_norm_valid = z_norm[valid_mask]
        
        # Color by z coordinates (same as scatter3d_sphere with z as color)
        z_color = 0.5 * z_norm_valid + 0.5  # Map from [-1,1] to [0,1] for RGB
        ax1.scatter(z_norm_valid[:, 0], z_norm_valid[:, 1], z_norm_valid[:, 2], c=z_color, s=10, alpha=0.8, zorder=1)
        
        # Add wireframe sphere
        u, v = np.mgrid[-0.25*np.pi:0.75*np.pi:256j, 0:np.pi:256j]
        r = 1.03
        ax1.plot_wireframe(r * np.cos(u) * np.sin(v),
                          r * np.sin(u) * np.sin(v),
                          r * np.cos(v), color='gray', alpha=0.4,
                          rstride=32, cstride=64)
        
        # Apply scatter3d_sphere styling to ax1
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.grid(False)
        ax1.set_xlim((-1, 1))
        ax1.set_ylim((-1, 1))
        ax1.set_zlim((-1, 1))
        ax1.set_aspect('equal')
        ax1.set_xticks([-1., 1.], minor=False)
        ax1.set_yticks([-1., 1.], minor=False)
        ax1.set_zticks([-1., 1.], minor=False)
        ax1.view_init(10, 45)
        
        # Right sphere: scatter3d_sphere(plt, z, g(z)) - plot z coordinates, color by g(z) coordinates  
        ax2 = fig.add_subplot(122, projection='3d')
        
        x_detached = x.detach()
        # Ensure proper normalization and handle edge cases
        x_norms = x_detached.norm(dim=-1, keepdim=True)
        x_norms = torch.clamp(x_norms, min=1e-8)  # Avoid division by zero
        x_norm = x_detached / x_norms
        
        # Check for any invalid values and filter them out
        x_valid_mask = torch.isfinite(x_norm).all(dim=-1)
        
        # Use combined mask to ensure correspondence between z and x
        combined_mask = valid_mask & x_valid_mask
        z_norm_plot = z_norm[combined_mask]
        x_norm_plot = x_norm[combined_mask]
        
        # Plot z coordinates but color by x (transformed) coordinates
        x_color = 0.5 * x_norm_plot + 0.5  # Map transformed coordinates to RGB colors
        ax2.scatter(z_norm_plot[:, 0], z_norm_plot[:, 1], z_norm_plot[:, 2], c=x_color, s=10, alpha=0.8, zorder=1)
        
        # Add wireframe sphere
        ax2.plot_wireframe(r * np.cos(u) * np.sin(v),
                          r * np.sin(u) * np.sin(v),
                          r * np.cos(v), color='gray', alpha=0.4,
                          rstride=32, cstride=64)
        
        # Apply scatter3d_sphere styling to ax2
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax2.grid(False)
        ax2.set_xlim((-1, 1))
        ax2.set_ylim((-1, 1))
        ax2.set_zlim((-1, 1))
        ax2.set_aspect('equal')
        ax2.set_xticks([-1., 1.], minor=False)
        ax2.set_yticks([-1., 1.], minor=False)
        ax2.set_zticks([-1., 1.], minor=False)
        ax2.view_init(10, 45)
        
        plt.tight_layout()
    
    # Save the plot
    safe_name = name.lower().replace(' ', '_')
    output_path = f'{output_dir}/{safe_name}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {name} comparison to: {output_path}")
    return output_path

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    latent_dim = 3
    data_dim = 3
    
    # Initialize all generative processes
    print("Initializing generative processes...")
    
    # 1. Identity
    g_identity = SphereDecoderIdentity()
    
    # 2. Linear 
    g_injective_linear = SphereDecoderLinear(latent_dim, data_dim)
    
    # # 3. Injective Linear
    # g_injective_linear = InjectiveLinearDecoder(latent_dim, data_dim)
    
    # 4. Spiral
    g_spiral = SpiralRotation(period_n=2)
    
    # 5. Patches
    g_patches = Patches(slice_number=4, device='cpu')
    
    # 6. Invertible MLP
    g_invertible_mlp = construct_invertible_mlp(n=latent_dim, n_layers=3)
    
    # Store all processes with names
    generative_processes = [
        ("Identity", g_identity),
        ("Injective Linear", g_injective_linear),
        ("Spiral", g_spiral),
        ("Patches", g_patches),
        ("Invertible MLP", g_invertible_mlp)
    ]
    
    # Output directory
    output_dir = 'sphere_visualization_outputs'
    
    print("Applying generative processes and creating individual visualizations...")
    
    # Specify which processes output sphere vs arbitrary data space
    sphere_outputs = {"Identity", "Spiral", "Patches"}  # These preserve sphere structure
    non_sphere_outputs = {"Injective Linear", "Invertible MLP"}  # These don't
    
    # Apply each generative process and create individual side-by-side comparison
    for name, g_process in generative_processes:
        print(f"Processing: {name}")
        
        # Use different sample sizes based on process type
        if name in {"Injective Linear", "Invertible MLP"}:
            n_samples = 3000  # 3x bigger than the original 1000
        else:
            n_samples = 200000  # For sphere processes
        
        print(f"  Sampling {n_samples} points from {latent_dim}D sphere...")
        z = sample_uniform_sphere(n_samples, latent_dim)
        
        # Generate colors for points
        colors = torch.rand(n_samples, 3)
        
        with torch.no_grad():
            try:
                x = g_process(z)
                
                # Ensure output is 3D for visualization
                if x.shape[-1] > 3:
                    x = x[:, :3]  # Take first 3 dimensions
                elif x.shape[-1] < 3:
                    # Pad with zeros if less than 3D
                    padding = torch.zeros(x.shape[0], 3 - x.shape[-1])
                    x = torch.cat([x, padding], dim=-1)
                
                # Determine if output should be treated as sphere or general data space
                is_sphere_output = name in sphere_outputs
                
                # Create side-by-side visualization
                create_side_by_side_visualization(z, x, colors, name, output_dir, is_sphere_output)
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue
    
    print("All visualizations complete!")

if __name__ == "__main__":
    main()