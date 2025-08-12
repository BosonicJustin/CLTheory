import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    'text.usetex': False,
    'font.size': 12
})

fig = plt.figure(figsize=(12, 5))

# Create sphere coordinates
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Original sphere
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, alpha=0.7, color='lightblue', edgecolor='navy', linewidth=0.3)
ax1.set_title(r'Latent Space $\mathcal{Z} = A\mathcal{Z}^{\prime}$')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Rotation matrix (45 degrees around z-axis, 30 degrees around x-axis)
theta_z = np.pi/4
theta_x = np.pi/6
Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
               [np.sin(theta_z), np.cos(theta_z), 0],
               [0, 0, 1]])
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta_x), -np.sin(theta_x)],
               [0, np.sin(theta_x), np.cos(theta_x)]])
R = Rz @ Rx

# Apply rotation
points = np.stack([x, y, z], axis=-1)
rotated_points = points @ R.T
x_rot = rotated_points[:, :, 0]
y_rot = rotated_points[:, :, 1]
z_rot = rotated_points[:, :, 2]

# Rotated sphere
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_rot, y_rot, z_rot, alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=0.3)
ax2.set_title(r"Recovered Latent Space $\mathcal{Z}^{\prime}$")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Set equal aspect ratio and limits
for ax in [ax1, ax2]:
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1,1,1])

# Remove tight_layout to avoid 3D rendering issues
# plt.tight_layout()
plt.savefig('/Users/justin/Desktop/CLTheory/sphere_rotation.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/justin/Desktop/CLTheory/sphere_rotation.pdf', bbox_inches='tight')
plt.show()

print("Sphere rotation plot saved as sphere_rotation.png and sphere_rotation.pdf")