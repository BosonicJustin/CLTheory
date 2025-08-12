import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

def create_sphere(radius=1, resolution=50):
    """Create sphere coordinates"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)
    
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v) 
    z = radius * np.cos(v)
    
    return x, y, z

def create_ring(radius=1.2, ring_radius=0.15, theta_offset=0, phi=np.pi/3, resolution=100):
    """Create a ring around the sphere at angle phi from the z-axis"""
    theta = np.linspace(0, 2*np.pi, resolution) + theta_offset
    
    # Ring center circle on the sphere
    ring_center_x = radius * np.cos(theta) * np.sin(phi)
    ring_center_y = radius * np.sin(theta) * np.sin(phi)
    ring_center_z = radius * np.cos(phi) * np.ones_like(theta)
    
    # Create ring by adding small circles perpendicular to the sphere surface
    ring_points = []
    for i, t in enumerate(theta):
        # Normal vector at this point on sphere
        normal = np.array([ring_center_x[i], ring_center_y[i], ring_center_z[i]]) / radius
        
        # Two orthogonal vectors to the normal
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Small circle around the ring
        small_circle_angles = np.linspace(0, 2*np.pi, 20)
        for angle in small_circle_angles:
            offset = ring_radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
            point = np.array([ring_center_x[i], ring_center_y[i], ring_center_z[i]]) + offset
            ring_points.append(point)
    
    return np.array(ring_points)

def create_mapping_arrows(ring1_center, ring2_center, num_arrows=8):
    """Create arrows showing the mapping between rings"""
    arrows = []
    theta = np.linspace(0, 2*np.pi, num_arrows, endpoint=False)
    
    for t in theta:
        # Points on ring centers
        start = ring1_center + 0.2 * np.array([np.cos(t), np.sin(t), 0])
        end = ring2_center + 0.2 * np.array([np.cos(t + np.pi/4), np.sin(t + np.pi/4), 0])
        arrows.append((start, end))
    
    return arrows

def plot_theorem_visualization():
    """Create the main visualization"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create sphere
    sphere_x, sphere_y, sphere_z = create_sphere(radius=1.3, resolution=60)
    
    # Plot sphere with transparency
    ax.plot_surface(sphere_x, sphere_y, sphere_z, 
                   alpha=0.25, color='lightgray', 
                   linewidth=0, antialiased=True)
    
    # Create two rings at different latitudes
    phi1 = np.pi/3  # Upper ring
    phi2 = 2*np.pi/3  # Lower ring
    
    ring1_points = create_ring(radius=1.3, ring_radius=0.08, phi=phi1, resolution=150)
    ring2_points = create_ring(radius=1.3, ring_radius=0.08, phi=phi2, theta_offset=np.pi/4, resolution=150)
    
    # Plot rings with better visibility
    ax.scatter(ring1_points[:, 0], ring1_points[:, 1], ring1_points[:, 2], 
              c='red', s=1.0, alpha=0.9, label='Feature space 1')
    ax.scatter(ring2_points[:, 0], ring2_points[:, 1], ring2_points[:, 2], 
              c='blue', s=1.0, alpha=0.9, label='Feature space 2')
    
    # Add mapping arrows between rings
    theta_arrows = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for i, t in enumerate(theta_arrows[::2]):  # Use every other arrow to avoid clutter
        # Points on ring circumference
        start_point = 1.3 * np.array([np.cos(t) * np.sin(phi1), 
                                     np.sin(t) * np.sin(phi1), 
                                     np.cos(phi1)])
        end_point = 1.3 * np.array([np.cos(t + np.pi/4) * np.sin(phi2), 
                                   np.sin(t + np.pi/4) * np.sin(phi2), 
                                   np.cos(phi2)])
        
        # Create curved arrow
        arrow = Arrow3D([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       [start_point[2], end_point[2]], 
                       mutation_scale=8, lw=1, arrowstyle="-|>", 
                       color="green", alpha=0.7)
        ax.add_artist(arrow)
    
    # Add mathematical annotations
    ax.text(-3.2, -0.8, np.cos(phi1) - 0.3, r'$h_1 = f_1 \circ g$', fontsize=14, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    ax.text(3.2, 0.8, np.cos(phi2) + 0.3, r'$h_2 = f_2 \circ g$', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3))
    
    # Add mapping label in a better position
    ax.text(1.8, 1.8, 0.5, r'$m: \mathcal{Z} \to \mathcal{Z}$' + '\n' + r'$h_2(z) = h_1(m(z))$', 
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.25", facecolor="lightgreen", alpha=0.8))
    
    # Set equal aspect ratio and clean up axes
    ax.set_xlim([-3.8, 3.8])
    ax.set_ylim([-3.8, 3.8])
    ax.set_zlim([-3.8, 3.8])
    
    # Remove axis labels but keep the structure clean
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # Remove tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set viewing angle for best perspective
    ax.view_init(elev=20, azim=45)
    
    # Force equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    
    plt.tight_layout()
    
    # Save high-resolution version for publication
    plt.savefig('feature_extractor_equivalence_theorem.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('feature_extractor_equivalence_theorem.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    print("Creating visualization for Feature Extractor Equivalence Theorem...")
    fig, ax = plot_theorem_visualization()
    print("Visualization saved as:")
    print("- feature_extractor_equivalence_theorem.pdf (for LaTeX)")
    print("- feature_extractor_equivalence_theorem.png (for preview)")
    print("\nTheorem: Two feature extractors h₁ and h₂ are equivalent under")
    print("invertible mapping m that preserves the conditional distribution.")