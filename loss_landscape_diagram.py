import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import patches
import matplotlib.patheffects as path_effects

def create_loss_function(x, y):
    """Create a loss function with multiple local minima"""
    # Base function with multiple valleys
    z = 0.3 * (x**2 + y**2) + \
        0.5 * np.exp(-((x-2)**2 + (y-1)**2)/0.5) + \
        0.5 * np.exp(-((x+1.5)**2 + (y+0.5)**2)/0.5) + \
        0.8 * np.exp(-((x-0.5)**2 + (y+2)**2)/0.8) + \
        0.6 * np.exp(-((x+2)**2 + (y-1.5)**2)/0.6) + \
        0.2 * np.sin(2*x) * np.sin(2*y) + \
        0.1 * np.cos(3*x + 2*y)
    
    # Invert to create minima instead of maxima
    z = 2.5 - z
    
    return z

def plot_loss_landscape():
    """Create the loss landscape visualization"""
    # Create meshgrid
    x = np.linspace(-3.5, 3.5, 300)
    y = np.linspace(-3.5, 3.5, 300)
    X, Y = np.meshgrid(x, y)
    
    # Compute loss function
    Z = create_loss_function(X, Y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot contour map
    contour = ax.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, Z, levels=25, colors='black', alpha=0.3, linewidths=0.5)
    
    # Define positions for h1 and h2 (local minima)
    h1_pos = (2.0, 1.0)
    h2_pos = (-1.5, -0.5)
    
    # Mark h1 and h2 positions
    ax.plot(h1_pos[0], h1_pos[1], 'ro', markersize=12, markeredgecolor='white', 
            markeredgewidth=2, zorder=10)
    ax.plot(h2_pos[0], h2_pos[1], 'bo', markersize=12, markeredgecolor='white', 
            markeredgewidth=2, zorder=10)
    
    # Add labels with white outlines for visibility
    h1_text = ax.text(h1_pos[0] + 0.3, h1_pos[1] + 0.2, r'$h_1$', fontsize=16, 
                      color='red', fontweight='bold', zorder=15)
    h1_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    h2_text = ax.text(h2_pos[0] - 0.5, h2_pos[1] - 0.3, r'$h_2$', fontsize=16, 
                      color='blue', fontweight='bold', zorder=15)
    h2_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Add inductive bias constraint region around h1
    constraint_ellipse = Ellipse(h1_pos, width=1.8, height=1.4, 
                                angle=25, facecolor='none', 
                                edgecolor='red', linewidth=3, 
                                linestyle='--', alpha=0.9, zorder=12)
    ax.add_patch(constraint_ellipse)
    
    # Add inductive bias label
    bias_text = ax.text(h1_pos[0] + 0.8, h1_pos[1] - 0.8, 
                       'Inductive\nBias', fontsize=12, 
                       color='red', fontweight='bold', 
                       ha='center', va='center', zorder=15)
    bias_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Add arrow pointing to the constraint boundary
    arrow = patches.FancyArrowPatch((h1_pos[0] + 0.5, h1_pos[1] - 0.5),
                                   (h1_pos[0] + 0.9, h1_pos[1] + 0.1),
                                   connectionstyle="arc3,rad=0.2", 
                                   arrowstyle='->', 
                                   mutation_scale=15, 
                                   color='red', linewidth=2, zorder=14)
    ax.add_patch(arrow)
    
    # Set equal aspect ratio and clean up
    ax.set_aspect('equal')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    
    # Remove ticks and labels for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add minimal axis labels
    ax.set_xlabel('Parameter Space', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parameter Space', fontsize=14, fontweight='bold')
    
    # Add colorbar with loss label
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Loss Value', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save high-resolution versions
    plt.savefig('loss_landscape_inductive_bias.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('loss_landscape_inductive_bias.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    return fig, ax

def plot_simple_loss_landscape():
    """Create a cleaner, simpler version"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a simpler loss landscape
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    
    # Simpler function with clear minima
    Z = (X**2 + Y**2) * 0.1 + \
        np.exp(-((X-2)**2 + (Y-0.5)**2)/0.8) + \
        np.exp(-((X+2)**2 + (Y+0.5)**2)/0.8) + \
        0.3 * np.sin(1.5*X) * np.sin(1.5*Y)
    
    Z = 2.0 - Z
    
    # Plot with fewer contour lines for clarity
    contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.9)
    ax.contour(X, Y, Z, levels=15, colors='gray', alpha=0.4, linewidths=0.8)
    
    # Mark equivalent minima
    h1_pos = (2.0, 0.5)
    h2_pos = (-2.0, -0.5)
    
    ax.plot(h1_pos[0], h1_pos[1], 'ko', markersize=15, markerfacecolor='red', 
            markeredgecolor='white', markeredgewidth=3, zorder=10)
    ax.plot(h2_pos[0], h2_pos[1], 'ko', markersize=15, markerfacecolor='blue', 
            markeredgecolor='white', markeredgewidth=3, zorder=10)
    
    # Labels
    ax.text(h1_pos[0], h1_pos[1] + 0.4, r'$h_1$', fontsize=18, 
            ha='center', va='bottom', fontweight='bold', color='red')
    ax.text(h2_pos[0], h2_pos[1] - 0.4, r'$h_2$', fontsize=18, 
            ha='center', va='top', fontweight='bold', color='blue')
    
    # Inductive bias constraint region
    constraint_circle = plt.Circle(h1_pos, 1.2, facecolor='none', 
                                  edgecolor='red', linewidth=4, 
                                  linestyle='--', alpha=0.8, zorder=12)
    ax.add_patch(constraint_circle)
    
    # Add accessible hypothesis space label
    ax.text(h1_pos[0], h1_pos[1] + 1.6, 'Accessible\nHypothesis Space', 
            fontsize=11, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontweight='bold', color='red')
    
    # Clean up
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    
    # Save
    plt.savefig('simple_loss_landscape.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('simple_loss_landscape.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    print("Creating loss landscape visualization...")
    
    # Create the simple version
    fig, ax = plot_simple_loss_landscape()
    
    print("Simple visualization saved as:")
    print("- simple_loss_landscape.pdf")
    print("- simple_loss_landscape.png")
    
    print("\nVisualization shows:")
    print("- Multiple equivalent local minima (h₁ and h₂)")
    print("- Inductive bias constraint region around h₁")
    print("- Both solutions achieve same loss value")