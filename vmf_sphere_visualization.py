import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sampling.sphere_sampling import sample_vMF

def create_sphere_surface(resolution=50):
    """Create sphere surface coordinates."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def visualize_vmf_sphere(num_samples=1000, kappa=5.0):
    """Visualize vMF distribution on sphere with north pole at (0, 0, 1)."""
    
    # North pole as center
    mu = np.array([0.0, 0.0, 1.0])
    
    # Sample points from vMF distribution
    samples = sample_vMF(mu, kappa, num_samples)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create and plot sphere surface with grid lines
    x_sphere, y_sphere, z_sphere = create_sphere_surface()
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='lightsteelblue',
                   linewidth=0.5, edgecolors='navy', antialiased=True)
    
    # Add wireframe for grid lines
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.2, color='steelblue', linewidth=0.3)
    
    # Plot vMF samples
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], 
              c='darkblue', s=25, alpha=0.8, edgecolors='navy', linewidth=0.5)
    
    # Set equal aspect ratio
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # Add axis labels
    ax.set_xlabel('X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z', fontsize=14, fontweight='bold')
    
    # Style axis panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3)
    
    # Set viewing angle for nice perspective
    ax.view_init(elev=20, azim=45)
    
    # Mark north pole LAST to ensure it's visible on top
    ax.scatter([0], [0], [1], c='red', s=400, marker='*', 
              edgecolors='darkred', linewidth=3, zorder=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create visualization with 2000 samples and lower concentration
    visualize_vmf_sphere(num_samples=2000, kappa=4.0)