import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from matplotlib.path import Path
import matplotlib.patches as mpatches

def create_blob_shape(center, width, height, wobble=0.3):
    """Create an irregular blob shape similar to the example."""
    angles = np.linspace(0, 2*np.pi, 20)
    # Create base ellipse with random wobbles
    x_base = center[0] + (width/2) * np.cos(angles)
    y_base = center[1] + (height/2) * np.sin(angles)
    
    # Add wobbles to make it look like a manifold
    wobbles_x = wobble * np.random.normal(0, 0.3, len(angles))
    wobbles_y = wobble * np.random.normal(0, 0.3, len(angles))
    
    x_blob = x_base + wobbles_x
    y_blob = y_base + wobbles_y
    
    # Smooth the shape
    vertices = list(zip(x_blob, y_blob))
    return Polygon(vertices, closed=True)

def draw_curved_arrow(ax, start, end, curve_height=0.5, color='black', linewidth=2):
    """Draw a smooth curved arrow between two points."""
    # Calculate control points for bezier curve
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2 + curve_height
    
    # Create bezier curve
    t = np.linspace(0, 1, 100)
    x_curve = (1-t)**2 * start[0] + 2*(1-t)*t * mid_x + t**2 * end[0]
    y_curve = (1-t)**2 * start[1] + 2*(1-t)*t * mid_y + t**2 * end[1]
    
    # Plot the entire curve
    ax.plot(x_curve, y_curve, color=color, linewidth=linewidth)
    
    # Calculate direction for arrowhead
    dx = x_curve[-1] - x_curve[-5]
    dy = y_curve[-1] - y_curve[-5]
    
    # Draw arrowhead at the end
    ax.annotate('', xy=end, xytext=(x_curve[-5], y_curve[-5]),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                              shrinkA=0, shrinkB=0))

def create_manifold_diagram_2d():
    """Create a 2D manifold diagram in the style of the example."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Set random seed for reproducible blob shapes
    np.random.seed(42)
    
    # Define positions
    z_center = (2, 4)
    x_center = (6, 4) 
    zp_center = (10, 4)
    
    # Create manifold Z (circle)
    z_circle = Circle(z_center, 1.2, facecolor='lightgray', 
                     edgecolor='black', linewidth=2)
    ax.add_patch(z_circle)
    
    # Create manifold X (square)
    square_size = 1.5
    x_square = patches.Rectangle((x_center[0] - square_size/2, x_center[1] - square_size/2), 
                                square_size, square_size, 
                                facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(x_square)
    
    # Create manifold Z' (irregular blob)
    np.random.seed(42)
    zp_blob = create_blob_shape(zp_center, 2.5, 1.8, wobble=0.4)
    zp_blob.set_facecolor('lightgray')
    zp_blob.set_edgecolor('black')
    zp_blob.set_linewidth(2)
    ax.add_patch(zp_blob)
    
    # Add arrows between manifolds - from inside shape to inside shape
    # Arrow from Z to X (g) - from inside circle to inside square
    draw_curved_arrow(ax, (z_center[0] + 0.5, z_center[1]), (x_center[0] - 0.3, x_center[1]), curve_height=0.3, color='black', linewidth=2)
    
    # Arrow from X to Z' (f) - from inside square to inside blob
    draw_curved_arrow(ax, (x_center[0] + 0.3, x_center[1]), (zp_center[0] - 0.5, zp_center[1]), curve_height=0.3, color='black', linewidth=2)
    
    # Arrow from Z' to Z (j) - from inside blob to inside circle (more curved)
    draw_curved_arrow(ax, (zp_center[0] - 0.3, zp_center[1] - 0.5), (z_center[0] + 0.3, z_center[1] - 0.5), curve_height=-1.2, color='black', linewidth=2)
    
    # Arrow from Z to Z' (h) - from inside circle to inside blob (upper arc)
    draw_curved_arrow(ax, (z_center[0] + 0.3, z_center[1] + 0.5), (zp_center[0] - 0.3, zp_center[1] + 0.5), curve_height=1.0, color='black', linewidth=2)
    
    # Add labels for manifolds
    ax.text(z_center[0], z_center[1], r'$\mathcal{Z}$', fontsize=24, 
            ha='center', va='center', fontweight='bold')
    ax.text(x_center[0], x_center[1], r'$\mathcal{X}$', fontsize=24,
            ha='center', va='center', fontweight='bold')
    ax.text(zp_center[0], zp_center[1], r"$\mathcal{Z}'$", fontsize=24,
            ha='center', va='center', fontweight='bold')
    
    # Add mapping labels
    ax.text(4, 4.5, r'$g$', fontsize=18, ha='center', va='center', fontweight='bold')
    ax.text(8, 4.5, r'$f$', fontsize=18, ha='center', va='center', fontweight='bold')
    ax.text(6, 2.2, r'$j$', fontsize=18, ha='center', va='center', fontweight='bold')
    ax.text(6, 5.2, r'$h$', fontsize=18, ha='center', va='center', fontweight='bold')
    
    # Add space labels
    ax.text(2, 6, 'Latent space', fontsize=16, ha='center', va='center')
    ax.text(6, 6, 'Data space', fontsize=16, ha='center', va='center')
    ax.text(10, 6, 'Latent space (Recovered)', fontsize=16, ha='center', va='center')
    
    # Set axis properties
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    
    # Remove axes
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_manifold_diagram_2d()