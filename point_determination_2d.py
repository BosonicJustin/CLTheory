import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': False,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.figsize': (10, 8)
})

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Define three linearly independent points (vertices of a triangle)
A = np.array([1, 1])
B = np.array([5, 2])
C = np.array([3, 5])

# Points to be determined
P1 = np.array([3, 2.8])
P2 = np.array([2.5, 3.5])

# Create triangle
triangle = Polygon([A, B, C], fill=False, edgecolor='blue', linewidth=2)
ax.add_patch(triangle)

# Plot the three reference points
ax.plot(A[0], A[1], 'ro', markersize=10, label='Point A')
ax.plot(B[0], B[1], 'go', markersize=10, label='Point B')  
ax.plot(C[0], C[1], 'bo', markersize=10, label='Point C')

# Plot the points to be determined
ax.plot(P1[0], P1[1], 'ko', markersize=8)
ax.plot(P2[0], P2[1], 'mo', markersize=8)

# Add labels for points
ax.annotate('A', A, xytext=(A[0]-0.3, A[1]-0.3), fontsize=14, fontweight='bold')
ax.annotate('B', B, xytext=(B[0]+0.1, B[1]-0.3), fontsize=14, fontweight='bold')
ax.annotate('C', C, xytext=(C[0]-0.3, C[1]+0.2), fontsize=14, fontweight='bold')
ax.annotate('P₁', P1, xytext=(P1[0]+0.1, P1[1]+0.2), fontsize=14, fontweight='bold')
ax.annotate('P₂', P2, xytext=(P2[0]+0.1, P2[1]+0.2), fontsize=14, fontweight='bold')

# Draw lines from points to each vertex
ax.plot([P1[0], A[0]], [P1[1], A[1]], 'r--', alpha=0.6, linewidth=1.5)
ax.plot([P1[0], B[0]], [P1[1], B[1]], 'g--', alpha=0.6, linewidth=1.5)
ax.plot([P1[0], C[0]], [P1[1], C[1]], 'b--', alpha=0.6, linewidth=1.5)

ax.plot([P2[0], A[0]], [P2[1], A[1]], 'r:', alpha=0.6, linewidth=1.5)
ax.plot([P2[0], B[0]], [P2[1], B[1]], 'g:', alpha=0.6, linewidth=1.5)
ax.plot([P2[0], C[0]], [P2[1], C[1]], 'b:', alpha=0.6, linewidth=1.5)

# Remove all text and captions

# Set up the plot
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_aspect('equal')

# Add arrows to show the coordinate system
ax.annotate('', xy=(0.5, 0), xytext=(0, 0), 
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
ax.annotate('', xy=(0, 0.5), xytext=(0, 0), 
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

plt.tight_layout()
plt.savefig('/Users/justin/Desktop/CLTheory/point_determination_2d.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/justin/Desktop/CLTheory/point_determination_2d.pdf', bbox_inches='tight')
plt.show()

print("Point determination diagram saved as point_determination_2d.png and point_determination_2d.pdf")