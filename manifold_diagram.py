import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_torus_surface(R=2, r=0.8, resolution=30):
    """Create a torus surface for manifold Z."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, 2 * np.pi, resolution)
    u, v = np.meshgrid(u, v)
    
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

def create_sphere_surface(radius=1.5, center=(0, 0, 0), resolution=30):
    """Create a sphere surface for manifold X."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)
    
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    return x, y, z

def create_saddle_surface(size=2, resolution=30):
    """Create a saddle surface for manifold Z'."""
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    x, y = np.meshgrid(x, y)
    z = 0.3 * (x**2 - y**2)
    return x, y, z

def draw_3d_arrow(ax, start, end, color='black', linewidth=2, mutation_scale=20):
    """Draw a 3D arrow between two points."""
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            return np.min(zs)
    
    arrow = Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   mutation_scale=mutation_scale, lw=linewidth, 
                   arrowstyle="-|>", color=color)
    ax.add_artist(arrow)

def create_manifold_diagram():
    """Create the complete manifold diagram."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Position manifolds in 3D space
    z_pos = (-6, -2, 2)      # Manifold Z (torus)
    x_pos = (0, 4, 0)        # Manifold X (sphere) 
    zp_pos = (6, -2, -1)     # Manifold Z' (saddle)
    
    # Create and plot manifold Z (torus)
    x_torus, y_torus, z_torus = create_torus_surface()
    x_torus += z_pos[0]
    y_torus += z_pos[1] 
    z_torus += z_pos[2]
    ax.plot_surface(x_torus, y_torus, z_torus, alpha=0.7, color='lightcoral', 
                   linewidth=0.5, edgecolors='darkred')
    
    # Create and plot manifold X (sphere)
    x_sphere, y_sphere, z_sphere = create_sphere_surface(center=x_pos)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.7, color='lightblue',
                   linewidth=0.5, edgecolors='darkblue')
    
    # Create and plot manifold Z' (saddle)
    x_saddle, y_saddle, z_saddle = create_saddle_surface()
    x_saddle += zp_pos[0]
    y_saddle += zp_pos[1]
    z_saddle += zp_pos[2]
    ax.plot_surface(x_saddle, y_saddle, z_saddle, alpha=0.7, color='lightgreen',
                   linewidth=0.5, edgecolors='darkgreen')
    
    # Draw arrows between manifolds
    # Arrow from Z to X (generative process g)
    draw_3d_arrow(ax, (-4, -1, 2), (-1.5, 2.5, 0), color='red', linewidth=3)
    
    # Arrow from X to Z' (mapping f)
    draw_3d_arrow(ax, (1.5, 2.5, 0), (4, -1, -1), color='blue', linewidth=3)
    
    # Arrow from Z' to Z (mapping j)
    draw_3d_arrow(ax, (4, -3, -1), (-4, -3, 2), color='green', linewidth=3)
    
    # Add text labels for manifolds
    ax.text(z_pos[0], z_pos[1]-1, z_pos[2]+2, r'$\mathcal{Z}$', fontsize=24, 
            fontweight='bold', ha='center', va='center')
    ax.text(x_pos[0], x_pos[1]+2, x_pos[2]+2, r'$\mathcal{X}$', fontsize=24,
            fontweight='bold', ha='center', va='center')
    ax.text(zp_pos[0], zp_pos[1]-1, zp_pos[2]+2, r"$\mathcal{Z}'$", fontsize=24,
            fontweight='bold', ha='center', va='center')
    
    # Add text labels for mappings
    ax.text(-2.5, 1, 1.5, r'$g$', fontsize=20, fontweight='bold', 
            ha='center', va='center', color='red')
    ax.text(2.5, 1, -0.5, r'$f$', fontsize=20, fontweight='bold',
            ha='center', va='center', color='blue')
    ax.text(0, -4, 0, r'$j$', fontsize=20, fontweight='bold',
            ha='center', va='center', color='green')
    
    # Set axis properties
    ax.set_xlim([-8, 8])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-3, 4])
    
    # Remove axis labels and ticks for cleaner look
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Make axis panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    
    # Set viewing angle
    ax.view_init(elev=15, azim=45)
    
    plt.title('Manifold Mappings: Generative Process and Representation Learning', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_manifold_diagram()