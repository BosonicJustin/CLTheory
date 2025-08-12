import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches

plt.rcParams.update({
    'font.size': 11,
    'text.usetex': False,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.figsize': (14, 8)
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Define the finite set S (triangle vertices + one interior point)
S = np.array([
    [1, 1],     # a0
    [3, 1],     # a1  
    [2, 3],     # a2
    [2, 1.5]    # s (additional point)
])

# Define the isometry f on S (rotation + translation)
theta = np.pi/4  # 45 degree rotation
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
translation = np.array([6, 2])

f_S = np.array([R @ point + translation for point in S])

# Plot original configuration
ax1.set_title('Original Set S', fontsize=14, pad=15)
ax1.plot(S[:3, 0], S[:3, 1], 'bo-', markersize=8, linewidth=2, label='Affine frame')
ax1.plot(S[3, 0], S[3, 1], 'ro', markersize=8, label='Additional point s')

# Add labels
labels = ['a₀', 'a₁', 'a₂', 's']
for i, (point, label) in enumerate(zip(S, labels)):
    ax1.annotate(label, point, xytext=(point[0]-0.1, point[1]+0.15), 
                fontsize=12, fontweight='bold')

# Show distances
for i in range(len(S)):
    for j in range(i+1, len(S)):
        dist = np.linalg.norm(S[i] - S[j])
        midpoint = (S[i] + S[j]) / 2
        ax1.plot([S[i, 0], S[j, 0]], [S[i, 1], S[j, 1]], 'k--', alpha=0.3)
        ax1.text(midpoint[0], midpoint[1], f'{dist:.2f}', fontsize=9, 
                ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

ax1.set_xlim(0, 4)
ax1.set_ylim(0, 4)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot image under isometry f
ax2.set_title('Image f(S) and Extension F', fontsize=14, pad=15)
ax2.plot(f_S[:3, 0], f_S[:3, 1], 'go-', markersize=8, linewidth=2, label='f(affine frame)')
ax2.plot(f_S[3, 0], f_S[3, 1], 'mo', markersize=8, label='f(s)')

# Add labels for image points
image_labels = ['f(a₀)', 'f(a₁)', 'f(a₂)', 'f(s)']
for i, (point, label) in enumerate(zip(f_S, image_labels)):
    ax2.annotate(label, point, xytext=(point[0]-0.1, point[1]+0.15), 
                fontsize=12, fontweight='bold')

# Show that distances are preserved
for i in range(len(f_S)):
    for j in range(i+1, len(f_S)):
        dist = np.linalg.norm(f_S[i] - f_S[j])
        midpoint = (f_S[i] + f_S[j]) / 2
        ax2.plot([f_S[i, 0], f_S[j, 0]], [f_S[i, 1], f_S[j, 1]], 'k--', alpha=0.3)
        ax2.text(midpoint[0], midpoint[1], f'{dist:.2f}', fontsize=9, 
                ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# Show the extended isometry F on additional points
test_points = np.array([[0.5, 0.5], [3.5, 2], [1.5, 3.5], [2.5, 0.5]])
extended_points = np.array([R @ point + translation for point in test_points])

ax1.plot(test_points[:, 0], test_points[:, 1], 'c^', markersize=6, alpha=0.7, label='Test points')
ax2.plot(extended_points[:, 0], extended_points[:, 1], 'c^', markersize=6, alpha=0.7, label='F(test points)')

# Draw arrows showing the transformation
for i, (orig, img) in enumerate(zip(test_points, extended_points)):
    # Draw curved arrow from left plot to right plot
    if i == 0:  # Only draw one arrow to avoid clutter
        ax1.annotate('', xy=(3.8, 2), xytext=(orig[0], orig[1]),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                  color='purple', lw=2))
        ax2.annotate('F', xy=(extended_points[0]), xytext=(5, 1.5),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2))

# Add coordinate axes
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

ax2.set_xlim(4, 10)
ax2.set_ylim(0, 6)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Add legends
ax1.legend(loc='upper right', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/justin/Desktop/CLTheory/isometry_extension_demo.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/justin/Desktop/CLTheory/isometry_extension_demo.pdf', bbox_inches='tight')
plt.show()

print("Isometry extension demonstration saved as isometry_extension_demo.png and isometry_extension_demo.pdf")

# Verify distances are preserved
print("\nDistance verification:")
for i in range(len(S)):
    for j in range(i+1, len(S)):
        orig_dist = np.linalg.norm(S[i] - S[j])
        image_dist = np.linalg.norm(f_S[i] - f_S[j])
        print(f"d({labels[i]}, {labels[j]}) = {orig_dist:.3f}, d(f({labels[i]}), f({labels[j]})) = {image_dist:.3f}")