import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_constraint_ratio_diagram():
    """Create a diagram showing the constraint ratio experimental design."""
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Colors for constrained vs unconstrained
    constrained_color = '#FF6B6B'    # Red/coral for constrained
    unconstrained_color = '#4ECDC4'  # Teal for unconstrained
    
    # Constraint ratios to show
    ratios = [0.0, 0.2, 0.5, 0.8, 1.0]
    titles = [r'$\rho = 0.0$', r'$\rho = 0.2$', r'$\rho = 0.5$', r'$\rho = 0.8$', r'$\rho = 1.0$']
    descriptions = [
        'Fully Unconstrained\n(Original InfoNCE)',
        'Mostly Unconstrained\n(20% Constrained)',
        'Mixed Sampling\n(50% Constrained)',
        'Mostly Constrained\n(80% Constrained)',
        'Fully Constrained\n(Proposed InfoNCE)'
    ]
    
    for i, (ax, ratio, title, desc) in enumerate(zip(axes, ratios, titles, descriptions)):
        
        # Create a circular manifold representation
        circle = patches.Circle((0.5, 0.5), 0.35, linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(circle)
        
        if ratio == 0.0:
            # Fully unconstrained - entire circle in unconstrained color
            inner_circle = patches.Circle((0.5, 0.5), 0.35, facecolor=unconstrained_color, alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(inner_circle)
            
        elif ratio == 1.0:
            # Fully constrained - entire circle in constrained color
            inner_circle = patches.Circle((0.5, 0.5), 0.35, facecolor=constrained_color, alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(inner_circle)
            
        else:
            # Mixed ratios - create wedges based on ratio
            # Constrained portion (red) - from bottom going clockwise
            constrained_angle = ratio * 360
            wedge_constrained = patches.Wedge((0.5, 0.5), 0.35, 270, 270 + constrained_angle, 
                                            facecolor=constrained_color, alpha=0.7, edgecolor='black', linewidth=1)
            ax.add_patch(wedge_constrained)
            
            # Unconstrained portion (teal) - remaining part
            wedge_unconstrained = patches.Wedge((0.5, 0.5), 0.35, 270 + constrained_angle, 270 + 360, 
                                              facecolor=unconstrained_color, alpha=0.7, edgecolor='black', linewidth=1)
            ax.add_patch(wedge_unconstrained)
            
            # Add dividing lines for clarity
            if ratio in [0.2, 0.5, 0.8]:
                # Calculate end points of dividing line(s)
                angle_rad = np.radians(270 + constrained_angle)
                x_end = 0.5 + 0.35 * np.cos(angle_rad)
                y_end = 0.5 + 0.35 * np.sin(angle_rad)
                ax.plot([0.5, x_end], [0.5, y_end], 'k-', linewidth=2)
        
        # Add mathematical notation labels on the colored regions
        if ratio == 0.0:
            # Only unconstrained region
            ax.text(0.5, 0.5, r'$\mathcal{Z} \setminus K(z)$', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='black',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black'))
        elif ratio == 1.0:
            # Only constrained region
            ax.text(0.5, 0.5, r'$K(z)$', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='black',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black'))
        else:
            # Mixed regions - add labels in appropriate areas
            # For constrained region (red) - position in the red area
            if ratio >= 0.2:  # Only add if there's enough red area
                # Calculate position in the middle of constrained wedge
                mid_angle_rad = np.radians(270 + constrained_angle/2)
                x_constrained = 0.5 + 0.18 * np.cos(mid_angle_rad)  # 0.18 is about half radius
                y_constrained = 0.5 + 0.18 * np.sin(mid_angle_rad)
                ax.text(x_constrained, y_constrained, r'$K(z)$', ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='black',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='black'))
            
            # For unconstrained region (teal) - position in the teal area
            if ratio <= 0.8:  # Only add if there's enough teal area
                # Calculate position in the middle of unconstrained wedge
                unconstrained_angle = (1 - ratio) * 360
                mid_angle_unconstrained_rad = np.radians(270 + constrained_angle + unconstrained_angle/2)
                x_unconstrained = 0.5 + 0.18 * np.cos(mid_angle_unconstrained_rad)
                y_unconstrained = 0.5 + 0.18 * np.sin(mid_angle_unconstrained_rad)
                ax.text(x_unconstrained, y_unconstrained, r'$\mathcal{Z} \setminus K(z)$', ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='black',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Add title and description
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.text(0.5, 0.05, desc, ha='center', va='center', fontsize=11, 
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        # Clean up axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Remove overall title as requested
    
    # Create legend
    legend_elements = [
        patches.Patch(facecolor=unconstrained_color, alpha=0.7, label=r'$\mathcal{N}_{\mathrm{unconstrained}}$ (Full Sphere $\mathbb{S}^2$)'),
        patches.Patch(facecolor=constrained_color, alpha=0.7, label=r'$\mathcal{N}_{\mathrm{constrained}}$ (Restricted Submanifold)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=2, fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Remove yellow explanation text as requested
    
    plt.tight_layout()
    
    # Save the diagram
    plt.savefig('constraint_ratio_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('constraint_ratio_diagram.pdf', bbox_inches='tight', facecolor='white')  # Also save as PDF for LaTeX
    
    print("📊 Constraint ratio diagram created!")
    print("   💾 Saved as: constraint_ratio_diagram.png (for viewing)")
    print("   💾 Saved as: constraint_ratio_diagram.pdf (for LaTeX inclusion)")
    
    plt.show()

def create_alternative_rectangular_diagram():
    """Alternative version with rectangular manifolds as requested."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Colors
    constrained_color = '#FF6B6B'    # Red for constrained
    unconstrained_color = '#4ECDC4'  # Teal for unconstrained
    
    ratios = [0.0, 0.5, 1.0]
    titles = [r'$\rho = 0.0$', r'$\rho = 0.5$', r'$\rho = 1.0$']
    
    for i, (ax, ratio, title) in enumerate(zip(axes, ratios, titles)):
        
        # Create rectangle representing the sampling space
        if ratio == 0.0:
            # Fully unconstrained
            rect = patches.Rectangle((0.1, 0.3), 0.8, 0.4, facecolor=unconstrained_color, 
                                   alpha=0.8, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
        elif ratio == 0.5:
            # Half and half
            # Left half (constrained)
            rect1 = patches.Rectangle((0.1, 0.3), 0.4, 0.4, facecolor=constrained_color, 
                                    alpha=0.8, edgecolor='black', linewidth=2)
            ax.add_patch(rect1)
            # Right half (unconstrained)
            rect2 = patches.Rectangle((0.5, 0.3), 0.4, 0.4, facecolor=unconstrained_color, 
                                    alpha=0.8, edgecolor='black', linewidth=2)
            ax.add_patch(rect2)
            
        else:  # ratio == 1.0
            # Fully constrained
            rect = patches.Rectangle((0.1, 0.3), 0.8, 0.4, facecolor=constrained_color, 
                                   alpha=0.8, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        
        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.text(0.5, 0.1, f'Constraint Ratio = {ratio}', ha='center', va='center', 
                fontsize=12, transform=ax.transAxes)
        
        # Clean up
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    fig.suptitle('Constraint Ratio: Negative Sampling Strategy', fontsize=18, fontweight='bold')
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor=constrained_color, alpha=0.8, label='Constrained Negatives'),
        patches.Patch(facecolor=unconstrained_color, alpha=0.8, label='Unconstrained Negatives')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
               ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('constraint_ratio_rectangular.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("📊 Alternative rectangular diagram created: constraint_ratio_rectangular.png")
    plt.show()

if __name__ == "__main__":
    print("🎨 Creating constraint ratio diagrams...")
    print("\n1. Creating circular manifold diagram (recommended):")
    create_constraint_ratio_diagram()
    
    print("\n2. Creating rectangular version:")
    create_alternative_rectangular_diagram()
    
    print("\n✅ Both diagrams created! Use the PDF version for LaTeX inclusion.")