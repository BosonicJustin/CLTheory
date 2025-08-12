import numpy as np
import matplotlib.pyplot as plt

def create_double_well_function(x):
    """Create a function with two local minima."""
    # Double well potential: combination of quartic and quadratic terms
    return 0.1 * (x**4 - 8*x**2 + 16) + 0.5

def find_local_minima():
    """Find the approximate locations of local minima."""
    # For our function, minima are approximately at x = ±2
    x1 = -2.0
    x2 = 2.0
    y1 = create_double_well_function(x1)
    y2 = create_double_well_function(x2)
    return (x1, y1), (x2, y2)

def plot_function_with_minima():
    """Plot the function with highlighted local minima."""
    # Create x values
    x = np.linspace(-4, 4, 1000)
    y = create_double_well_function(x)
    
    # Find minima
    (x1, y1), (x2, y2) = find_local_minima()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the function
    plt.plot(x, y, 'b-', linewidth=3, label='$f(x)$')
    
    # Highlight the local minima
    plt.plot(x1, y1, 'ro', markersize=12, markerfacecolor='red', 
             markeredgecolor='darkred', markeredgewidth=2, zorder=5)
    plt.plot(x2, y2, 'ro', markersize=12, markerfacecolor='red', 
             markeredgecolor='darkred', markeredgewidth=2, zorder=5)
    
    # Add labels for the minima
    plt.annotate(r'$h_1$', xy=(x1, y1), xytext=(x1, y1+0.8),
                fontsize=20, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.annotate(r'$h_2$', xy=(x2, y2), xytext=(x2, y2+0.8),
                fontsize=20, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Add grid and styling
    plt.grid(True, alpha=0.3)
    plt.xlabel(r'$\theta$', fontsize=18, fontweight='bold')
    plt.ylabel(r'$\mathcal{L}_{CL}(h, M, \tau)$', fontsize=18, fontweight='bold')
    plt.title('Contrastive Loss with Two Local Minima', fontsize=20, fontweight='bold', pad=20)
    
    # Set axis limits for better visualization
    plt.xlim(-4, 4)
    plt.ylim(-1, 5)
    
    # Style the axes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add a subtle background
    plt.gca().set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_function_with_minima()