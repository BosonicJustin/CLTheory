import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

def create_isometry_example_plot():
    """Create a visualization for the specific isometry example in the LaTeX file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define the same unit circle for both spaces (slightly smaller to give legend room)
    circle_radius = 0.8
    circle1 = Circle((0, 0), circle_radius, fill=False, edgecolor='darkblue', linewidth=2)
    circle2 = Circle((0, 0), circle_radius, fill=False, edgecolor='darkgreen', linewidth=2)
    ax1.add_patch(circle1)
    ax2.add_patch(circle2)
    
    # Generate multiple points for better illustration
    n_points = 8
    
    # Create points at different radii
    blue_radius = 0.35  # Distance for blue points
    red_radius = 0.25   # Distance for red points (closer to center, so 2x scaling = 0.5)
    
    # Generate blue points (regular points) - random each time
    n_blue = 5
    blue_angles = np.random.uniform(0, 2*np.pi, n_blue)
    blue_radii = np.full(n_blue, blue_radius)
    
    # Generate red points (points in S) - random each time
    n_red = 3
    red_angles = np.random.uniform(0, 2*np.pi, n_red)
    red_radii = np.full(n_red, red_radius)
    
    # Combine all points
    all_radii = np.concatenate([blue_radii, red_radii])
    all_angles = np.concatenate([blue_angles, red_angles])
    
    x_points = all_radii * np.cos(all_angles)
    y_points = all_radii * np.sin(all_angles)
    
    # Define finite set S (the red points)
    S_indices = list(range(n_blue, n_blue + n_red))  # Indices for red points
    
    # Rotation matrix (π/6 = 30 degrees)
    angle = np.pi/6
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    
    # Apply transformations properly
    points_original = np.column_stack([x_points, y_points])
    points_transformed = np.zeros_like(points_original)
    
    # Apply the transformation h(x) for each point
    for i in range(n_points):
        x_point = points_original[i]  # This is a 2D vector
        if i in S_indices:
            # For points in S: h(x) = 2Rx
            points_transformed[i] = 2 * (R @ x_point)
        else:
            # For points not in S: h(x) = Rx
            points_transformed[i] = R @ x_point
    
    # Print debug info to verify transformation
    print(f"Rotation angle: {angle*180/np.pi:.1f} degrees")
    print(f"Rotation matrix R:")
    print(R)
    print(f"Matrix determinant (should be 1): {np.linalg.det(R):.6f}")
    print(f"S_indices (points that get 2x scaling): {S_indices}")
    
    # Check all points
    for i in range(n_points):
        orig_dist = np.linalg.norm(points_original[i])
        trans_dist = np.linalg.norm(points_transformed[i])
        ratio = trans_dist / orig_dist if orig_dist > 0 else 0
        point_type = "S point" if i in S_indices else "Regular"
        print(f"Point {i} ({point_type}): Original dist={orig_dist:.6f}, Transformed dist={trans_dist:.6f}, Ratio={ratio:.6f}")
    
    # Check pairwise distances (this is what isometry should preserve)
    print("\nPairwise distance preservation:")
    for i in range(n_points):
        for j in range(i+1, n_points):
            orig_pair_dist = np.linalg.norm(points_original[i] - points_original[j])
            trans_pair_dist = np.linalg.norm(points_transformed[i] - points_transformed[j])
            ratio = trans_pair_dist / orig_pair_dist if orig_pair_dist > 0 else 0
            both_regular = i not in S_indices and j not in S_indices
            expected = "1.0 (preserved)" if both_regular else "≠1.0 (not preserved)"
            print(f"Distance {i}-{j}: Original={orig_pair_dist:.6f}, Transformed={trans_pair_dist:.6f}, Ratio={ratio:.6f} ({expected})")
    
    # Test rotation matrix
    test_point = np.array([1.0, 0.0])
    rotated_test = R @ test_point
    print(f"\nRotation test: [1,0] -> [{rotated_test[0]:.6f}, {rotated_test[1]:.6f}], distance: {np.linalg.norm(rotated_test):.6f}")
    
    # Plot points in X space
    regular_indices = [i for i in range(n_points) if i not in S_indices]
    ax1.scatter(x_points[regular_indices], y_points[regular_indices], 
               c='blue', s=80, alpha=0.8, edgecolors='darkblue', linewidth=1.5, 
               label='Regular points', zorder=5)
    ax1.scatter(x_points[S_indices], y_points[S_indices], 
               c='red', s=120, alpha=0.9, edgecolors='darkred', linewidth=2, 
               marker='s', label='Points in $S$', zorder=5)
    
    # Plot points in Y space
    ax2.scatter(points_transformed[regular_indices, 0], points_transformed[regular_indices, 1], 
               c='blue', s=80, alpha=0.8, edgecolors='darkblue', linewidth=1.5, 
               label='$h(x) = Rx$', zorder=5)
    ax2.scatter(points_transformed[S_indices, 0], points_transformed[S_indices, 1], 
               c='red', s=120, alpha=0.9, edgecolors='darkred', linewidth=2, 
               marker='s', label='$h(x) = 2Rx$', zorder=5)
    
    # Show two distance lines from one blue anchor point
    if len(regular_indices) >= 2 and len(S_indices) >= 1:
        anchor_idx = regular_indices[0]  # Blue anchor point
        
        # 1. Blue anchor to another blue point (preserved distance)
        other_blue_idx = regular_indices[1]
        
        # Distance in X
        ax1.plot([x_points[anchor_idx], x_points[other_blue_idx]], 
                [y_points[anchor_idx], y_points[other_blue_idx]], 
                'b--', linewidth=2, alpha=0.7, label='Preserved distances')
        
        # Corresponding distance in Y (should be same length)
        ax2.plot([points_transformed[anchor_idx, 0], points_transformed[other_blue_idx, 0]], 
                [points_transformed[anchor_idx, 1], points_transformed[other_blue_idx, 1]], 
                'b--', linewidth=2, alpha=0.7, label='Preserved distances')
        
        # 2. Blue anchor to red point (non-preserved distance)
        red_idx = S_indices[0]  # First red point
        
        # Distance in X
        ax1.plot([x_points[anchor_idx], x_points[red_idx]], 
                [y_points[anchor_idx], y_points[red_idx]], 
                'r:', linewidth=3, alpha=0.8, label='Non-preserved distance')
        
        # Corresponding distance in Y (different length)
        ax2.plot([points_transformed[anchor_idx, 0], points_transformed[red_idx, 0]], 
                [points_transformed[anchor_idx, 1], points_transformed[red_idx, 1]], 
                'r:', linewidth=3, alpha=0.8, label='Non-preserved distance')
    
    # Add curved arrow between spaces (using figure coordinates)
    arrow = mpatches.FancyArrowPatch((0.35, 0.5), (0.65, 0.5),
                                   connectionstyle="arc3,rad=0.3", 
                                   arrowstyle='->', mutation_scale=20, 
                                   color='black', linewidth=2,
                                   transform=fig.transFigure)
    fig.add_artist(arrow)
    
    # Add mapping label
    fig.text(0.5, 0.55, r'$h: X \to Y$', ha='center', va='center', 
             fontsize=16, fontweight='bold')
    
    # Style the plots with same scale for both spaces
    for ax in [ax1, ax2]:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#fafafa')
    
    # Add titles
    ax1.set_title(r'Space $X = \mathbb{R}^2$', fontsize=14, fontweight='bold', pad=15)
    ax2.set_title(r'Space $Y = \mathbb{R}^2$', fontsize=14, fontweight='bold', pad=15)
    
    # Add legends
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add main title (simplified to avoid LaTeX parsing issues)
    fig.suptitle(r'Isometry Almost Everywhere: $h(x) = Rx$ if $x \notin S$, $h(x) = 2Rx$ if $x \in S$', 
                 fontsize=13, fontweight='bold', y=0.95)
    
    # Add annotation about measure
    fig.text(0.5, 0.08, r'Finite set $S$ has measure zero: $\mu(S) = 0$', 
             ha='center', va='center', fontsize=12, style='italic',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.18)
    plt.show()

if __name__ == "__main__":
    create_isometry_example_plot()