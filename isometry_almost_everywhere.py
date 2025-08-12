import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as mpatches

def create_isometry_visualization():
    """Create a visualization showing isometry almost everywhere on generic normed spaces."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create 12 points in a somewhat regular pattern
    np.random.seed(42)  # For reproducibility
    
    # Create points in first space - mix of structured and random
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    radius1 = 2.0
    x1_outer = radius1 * np.cos(angles)
    y1_outer = radius1 * np.sin(angles)
    
    # Add some inner points
    x1_inner = np.array([0.8, -0.8, 0.5, -0.5])
    y1_inner = np.array([0.5, -0.8, -1.2, 1.0])
    
    x1 = np.concatenate([x1_outer, x1_inner])
    y1 = np.concatenate([y1_outer, y1_inner])
    
    # For second space, most points are rotated versions (isometric)
    # Apply rotation matrix (30 degrees)
    theta = np.pi/6  # 30 degrees
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    x2_iso = cos_theta * x1 - sin_theta * y1
    y2_iso = sin_theta * x1 + cos_theta * y1
    
    # Create the non-isometric pair - stretch one pair
    x2 = x2_iso.copy()
    y2 = y2_iso.copy()
    
    # Choose indices for the non-isometric pair (stretch these)
    stretch_indices = [2, 7]  # Two points to form the stretched pair
    stretch_factor = 1.8
    
    for idx in stretch_indices:
        x2[idx] *= stretch_factor
        y2[idx] *= stretch_factor
    
    # Define pairs for distance visualization (fewer for clarity)
    pairs = [(0, 4), (1, 5), (8, 9)]  # 3 isometric pairs
    non_iso_pair = (2, 7)  # 1 non-isometric pair
    
    # Plot the spaces as clean abstract regions
    theta_shape = np.linspace(0, 2*np.pi, 50)
    x_shape1 = 3.5 * np.cos(theta_shape) + 0.2 * np.cos(2*theta_shape)
    y_shape1 = 3.5 * np.sin(theta_shape) + 0.1 * np.sin(2*theta_shape)
    
    x_shape2 = 3.5 * np.cos(theta_shape) + 0.15 * np.cos(3*theta_shape)
    y_shape2 = 3.5 * np.sin(theta_shape) + 0.1 * np.sin(3*theta_shape)
    
    shape1 = Polygon(list(zip(x_shape1, y_shape1)), fill=False, 
                    edgecolor='darkblue', linewidth=2, alpha=0.6)
    shape2 = Polygon(list(zip(x_shape2, y_shape2)), fill=False, 
                    edgecolor='darkgreen', linewidth=2, alpha=0.6)
    
    ax1.add_patch(shape1)
    ax2.add_patch(shape2)
    
    # Plot all points
    ax1.scatter(x1, y1, c='black', s=100, alpha=0.8, edgecolors='black', 
               linewidth=1.5, zorder=5)
    ax2.scatter(x2, y2, c='black', s=100, alpha=0.8, edgecolors='black', 
               linewidth=1.5, zorder=5)
    
    # Draw isometric pairs (blue lines)
    for i, j in pairs:
        # Distance in first space
        ax1.plot([x1[i], x1[j]], [y1[i], y1[j]], 'b-', linewidth=3, alpha=0.7)
        
        # Corresponding preserved distance in second space
        ax2.plot([x2[i], x2[j]], [y2[i], y2[j]], 'b-', linewidth=3, alpha=0.7)
    
    # Draw non-isometric pair (red lines)
    i, j = non_iso_pair
    ax1.plot([x1[i], x1[j]], [y1[i], y1[j]], 'r-', linewidth=4, alpha=0.9)
    ax2.plot([x2[i], x2[j]], [y2[i], y2[j]], 'r-', linewidth=4, alpha=0.9)
    
    # Add curved arrow between spaces
    arrow = mpatches.FancyArrowPatch((4.5, 0), (7.5, 0),
                                   connectionstyle="arc3,rad=0.3", 
                                   arrowstyle='->', mutation_scale=25, 
                                   color='black', linewidth=3)
    fig.add_artist(arrow)
    
    # Add mapping label
    fig.text(0.5, 0.55, r'$h: (X, \|\cdot\|_X) \to (Y, \|\cdot\|_Y)$', 
             ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Style the plots
    for ax in [ax1, ax2]:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#fafafa')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add titles
    ax1.set_title(r'Normed Space $(X, \|\cdot\|_X)$', fontsize=18, fontweight='bold', pad=20)
    ax2.set_title(r'Normed Space $(Y, \|\cdot\|_Y)$', fontsize=18, fontweight='bold', pad=20)
    
    # Add legend
    blue_patch = mpatches.Patch(color='blue', alpha=0.8, 
                               label='Distance-preserving pairs (measure 1)')
    red_patch = mpatches.Patch(color='red', alpha=0.8, 
                              label='Non-isometric pair (measure 0)')
    fig.legend(handles=[blue_patch, red_patch], loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
    
    # Add mathematical annotation
    fig.text(0.5, 0.92, r'Isometry Almost Everywhere', 
             ha='center', va='center', fontsize=18, fontweight='bold')
    fig.text(0.5, 0.88, r'$\|h(x) - h(y)\|_Y = \|x - y\|_X$ a.e.', 
             ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15)
    plt.show()

if __name__ == "__main__":
    create_isometry_visualization()