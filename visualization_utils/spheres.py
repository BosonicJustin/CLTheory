import numpy as np


def visualize_spheres_side_by_side(plt, original_latents, encoded_latents):
    z_3d = original_latents[:, :3].detach().numpy()
    encoded_data_3d = encoded_latents[:, :3].detach().numpy()

    c_original = 0.5 * z_3d[:, 2] + 0.5
    c_encoded = 0.5 * encoded_data_3d[:, 2] + 0.5

    fig = plt.figure(figsize=(12, 6))

    # Original points
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2],
                           c=c_original, cmap='viridis', s=20, alpha=0.8)
    ax1.set_title('Sphere 1')
    fig.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=10, label='Color by Z-axis')

    # Encoded points
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(encoded_data_3d[:, 0], encoded_data_3d[:, 1], encoded_data_3d[:, 2],
                           c=c_encoded, cmap='viridis', s=20, alpha=0.8)
    ax2.set_title('Sphere 2')
    fig.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=10, label='Color by Z-axis')

    plt.tight_layout()
    plt.show()


def scatter3d_sphere(plt, z, color, s=10, a=0.8, r=1.03):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    z = z.detach()
    z = z / z.norm(dim=-1, keepdim=True)
    color = 0.5 * color.detach() + 0.5
    ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=color, s=s, alpha=a, zorder=1)

    u, v = np.mgrid[-0.25*np.pi:0.75*np.pi:256j, 0:np.pi:256j]
    ax.plot_wireframe(r * np.cos(u) * np.sin(v),
                      r * np.sin(u) * np.sin(v),
                      r * np.cos(v), color='gray', alpha=0.4,
                      rstride=32, cstride=64)

    # Hide everything else
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # ax.set_proj_type('ortho')

    plt.tight_layout()
    ax.grid(False)

    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_aspect('equal')

    ax.set_xticks([-1., 1.], minor=False)
    ax.set_yticks([-1., 1.], minor=False)
    ax.set_zticks([-1., 1.], minor=False)

    ax.view_init(10, 45)

    return fig