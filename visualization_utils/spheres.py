import numpy as np


def visualize_spheres_side_by_side(plt, original_latents, encoded_latents):
    z_3d = original_latents[:, :3].detach().cpu().numpy()
    encoded_data_3d = encoded_latents[:, :3].detach().cpu().numpy()

    n_points = z_3d.shape[0]

    # Create maximally distinct colors by shuffling HSV hues
    hsv_colors = plt.cm.hsv(np.linspace(0, 1, n_points))
    np.random.seed(42)  # make it reproducible
    np.random.shuffle(hsv_colors)
    colors = hsv_colors[:, :3]  # remove alpha

    fig = plt.figure(figsize=(12, 6))

    # Original latent space
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2],
                c=colors, s=20, alpha=0.9)
    ax1.set_title('Original Latents')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Encoded latent space
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(encoded_data_3d[:, 0], encoded_data_3d[:, 1], encoded_data_3d[:, 2],
                c=colors, s=20, alpha=0.9)
    ax2.set_title('Encoded Latents')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

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