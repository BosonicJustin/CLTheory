import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

def create_grid_on_circle(center, radius, n_points=8):
    """Create a grid of points on a circle."""
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return x, y

def create_isometry_visualization():
    """Create a visualization showing isometry almost everywhere."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define centers and radii
    center_z = (0, 0)
    center_zp = (0, 0)
    radius = 2.5
    
    # Create circles for the spaces
    circle_z = Circle(center_z, radius, fill=False, edgecolor='darkblue', linewidth=3)
    circle_zp = Circle(center_zp, radius, fill=False, edgecolor='darkgreen', linewidth=3)
    ax1.add_patch(circle_z)
    ax2.add_patch(circle_zp)
    
    # Create grid points
    x_z, y_z = create_grid_on_circle(center_z, radius * 0.7, 12)
    x_zp, y_zp = create_grid_on_circle(center_zp, radius * 0.7, 12)
    
    # Add some inner points
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    x_z_inner = center_z[0] + radius * 0.3 * np.cos(inner_angles)
    y_z_inner = center_z[1] + radius * 0.3 * np.sin(inner_angles)
    x_zp_inner = center_zp[0] + radius * 0.3 * np.cos(inner_angles)
    y_zp_inner = center_zp[1] + radius * 0.3 * np.sin(inner_angles)
    
    # Combine all points
    x_z_all = np.concatenate([x_z, x_z_inner])
    y_z_all = np.concatenate([y_z, y_z_inner])
    x_zp_all = np.concatenate([x_zp, x_zp_inner])
    y_zp_all = np.concatenate([y_zp, y_zp_inner])
    
    # Plot points - most are blue (isometric), few are red (non-isometric)
    n_total = len(x_z_all)
    n_bad = 2  # Only 2 points are non-isometric
    
    # Good points (isometric)
    good_indices = list(range(n_total))
    bad_indices = [3, 15]  # Choose specific indices for bad points
    for idx in bad_indices:
        good_indices.remove(idx)
    
    # Plot good points in both spaces
    ax1.scatter(x_z_all[good_indices], y_z_all[good_indices], 
               c='blue', s=80, alpha=0.8, edgecolors='darkblue', linewidth=1.5, zorder=5)
    ax2.scatter(x_zp_all[good_indices], y_zp_all[good_indices], 
               c='blue', s=80, alpha=0.8, edgecolors='darkblue', linewidth=1.5, zorder=5)
    
    # Plot bad points (slightly offset to show imperfection)
    ax1.scatter(x_z_all[bad_indices], y_z_all[bad_indices], 
               c='red', s=80, alpha=0.8, edgecolors='darkred', linewidth=1.5, zorder=5)
    # Add small perturbation to show non-isometry
    x_bad_perturbed = x_zp_all[bad_indices] + np.array([0.2, -0.15])
    y_bad_perturbed = y_zp_all[bad_indices] + np.array([0.15, 0.2])
    ax2.scatter(x_bad_perturbed, y_bad_perturbed, 
               c='red', s=80, alpha=0.8, edgecolors='darkred', linewidth=1.5, zorder=5)
    
    # Draw some distance preservation lines for good points
    for i in range(0, len(good_indices), 3):
        idx1 = good_indices[i]
        idx2 = good_indices[(i+1) % len(good_indices)]
        
        # Distance in Z
        ax1.plot([x_z_all[idx1], x_z_all[idx2]], [y_z_all[idx1], y_z_all[idx2]], 
                'b--', alpha=0.6, linewidth=2)
        
        # Corresponding distance in Z' (preserved)
        ax2.plot([x_zp_all[idx1], x_zp_all[idx2]], [y_zp_all[idx1], y_zp_all[idx2]], 
                'b--', alpha=0.6, linewidth=2)
    
    # Draw non-preserved distances for bad points
    if len(bad_indices) >= 2:
        idx1, idx2 = bad_indices[0], bad_indices[1]
        ax1.plot([x_z_all[idx1], x_z_all[idx2]], [y_z_all[idx1], y_z_all[idx2]], 
                'r:', alpha=0.8, linewidth=3)
        ax2.plot([x_bad_perturbed[0], x_bad_perturbed[1]], 
                [y_bad_perturbed[0], y_bad_perturbed[1]], 
                'r:', alpha=0.8, linewidth=3)
    
    # Add curved arrow between spaces
    arrow = mpatches.FancyArrowPatch((3.5, 0), (6.5, 0),
                                   connectionstyle="arc3,rad=0.3", 
                                   arrowstyle='->', mutation_scale=25, 
                                   color='black', linewidth=3)
    fig.add_artist(arrow)
    
    # Add mapping label
    fig.text(0.5, 0.55, r'$h: \mathcal{Z} \to \mathcal{Z}^{\prime}$', 
             ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Style the plots
    for ax in [ax1, ax2]:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#fafafa')
    
    # Add titles and labels
    ax1.set_title(r'Original Space $\mathcal{Z}$', fontsize=18, fontweight='bold', pad=20)
    ax2.set_title(r'Learned Space $\mathcal{Z}^{\prime}$', fontsize=18, fontweight='bold', pad=20)
    
    # Add legend
    blue_patch = mpatches.Patch(color='blue', alpha=0.8, label='Isometric pairs (measure 1)')
    red_patch = mpatches.Patch(color='red', alpha=0.8, label='Non-isometric pairs (measure 0)')
    fig.legend(handles=[blue_patch, red_patch], loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
    
    # Add mathematical annotation
    fig.text(0.5, 0.92, r'Isometry Almost Everywhere: $d_{\mathcal{Z}^{\prime}}(h(z), h(\tilde{z})) = d_{\mathcal{Z}}(z, \tilde{z})$ a.e.', 
             ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.show()

if __name__ == "__main__":
    create_isometry_visualization()