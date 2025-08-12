import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    'text.usetex': False,
    'font.size': 12
})

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create sphere coordinates
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 30)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere surface with transparency
ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.4, color='lightblue', edgecolor='none')
# Add wireframe for structure
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.2, color='blue', linewidth=0.3)

# Fixed Z value for the circle
z_fixed = 0.3
radius_circle = np.sqrt(1 - z_fixed**2)  # radius of circle at height z_fixed

# Create circle at fixed Z
theta = np.linspace(0, 2*np.pi, 100)
x_circle = radius_circle * np.cos(theta)
y_circle = radius_circle * np.sin(theta)
z_circle = z_fixed * np.ones_like(theta)

# Plot the circle with higher z-order to ensure it's on top
ax.plot(x_circle, y_circle, z_circle, 'red', linewidth=5, zorder=10)

# Add larger points to mark the circle more clearly
ax.scatter(x_circle[::8], y_circle[::8], z_circle[::8], c='red', s=50, zorder=10)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(r'Unit Sphere with Circle at Fixed Z-level', fontsize=14, pad=20)

# Set equal aspect ratio and limits
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.set_box_aspect([1,1,1])

# Remove legend

# Add grid
ax.grid(True, alpha=0.3)

plt.savefig('/Users/justin/Desktop/CLTheory/sphere_with_circle.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/justin/Desktop/CLTheory/sphere_with_circle.pdf', bbox_inches='tight')
plt.show()

print("Sphere with circle plot saved.")