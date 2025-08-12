import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_sphere(radius=1, resolution=50):
    """
    Plot a 3D sphere using matplotlib.
    
    Args:
        radius: Radius of the sphere (default: 1)
        resolution: Number of points for sphere surface (default: 50)
    """
    # Create parameter arrays
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    # Create sphere coordinates
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sphere surface
    ax.plot_surface(x, y, z, alpha=0.7, color='lightblue', 
                   linewidth=0.5, edgecolors='blue', antialiased=True)
    
    # Set equal aspect ratio
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Sphere')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_3d_sphere()