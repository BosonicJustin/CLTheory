import numpy as np
import colorsys
from sklearn import linear_model


def spherical_gradient_colors(points):
    """
    Generate gradient colors based on spherical coordinates.
    Maps azimuthal angle (phi) to hue and polar angle (theta) to saturation/lightness.

    Args:
        points: numpy array of shape (n, 3) with x, y, z coordinates

    Returns:
        numpy array of shape (n, 3) with RGB colors
    """
    # Normalize points to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    points_normalized = points / norms

    x, y, z = points_normalized[:, 0], points_normalized[:, 1], points_normalized[:, 2]

    # Compute spherical coordinates
    # Azimuthal angle (phi): angle in x-y plane from x-axis
    phi = np.arctan2(y, x)  # range: [-pi, pi]
    # Polar angle (theta): angle from z-axis
    theta = np.arccos(np.clip(z, -1, 1))  # range: [0, pi]

    # Map to HSL color space
    # Hue: based on azimuthal angle (full rainbow around equator)
    hue = (phi + np.pi) / (2 * np.pi)  # normalize to [0, 1]

    # Saturation: high everywhere for vivid colors
    saturation = np.ones_like(hue) * 0.9

    # Lightness: varies with polar angle (brighter at poles, medium at equator)
    lightness = 0.3 + 0.4 * (1 - np.abs(z))  # range roughly [0.3, 0.7]

    # Convert HSL to RGB
    colors = np.array([colorsys.hls_to_rgb(h, l, s)
                       for h, l, s in zip(hue, lightness, saturation)])

    return colors


def linear_unrotation(z, z_enc):
    """
    Compute the linear unrotation of encoded representations back to the original latent space.

    Args:
        z: Original latent samples (numpy array or torch.Tensor)
        z_enc: Encoded representations (numpy array or torch.Tensor)

    Returns:
        numpy array: Unrotated representations projected back onto the unit sphere
    """
    # Handle torch tensors
    if hasattr(z, 'detach'):
        z = z.detach().cpu().numpy()
    if hasattr(z_enc, 'detach'):
        z_enc = z_enc.detach().cpu().numpy()

    model = linear_model.LinearRegression()
    model.fit(z_enc, z)

    unrotated = model.predict(z_enc)
    unrotated = unrotated / np.linalg.norm(unrotated, axis=1, keepdims=True)

    return unrotated


def visualize_spheres_side_by_side(plt, original_latents, encoded_latents, apply_unrotation=True):
    """
    Visualize original and encoded latent spaces side by side with gradient coloring.

    Args:
        plt: matplotlib.pyplot module
        original_latents: Original latent samples (torch.Tensor)
        encoded_latents: Encoded representations (torch.Tensor)
        apply_unrotation: If True, apply linear unrotation to encoded latents
    """
    z_3d = original_latents[:, :3].detach().cpu().numpy()
    encoded_data_3d = encoded_latents[:, :3].detach().cpu().numpy()

    # Apply linear unrotation to align encoded space with original
    if apply_unrotation:
        encoded_data_3d = linear_unrotation(z_3d, encoded_data_3d)

    # Generate gradient colors based on original latent positions
    colors = spherical_gradient_colors(z_3d)

    fig = plt.figure(figsize=(12, 6))

    # Original latent space
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2],
                c=colors, s=20, alpha=0.9)
    ax1.set_title('Original Latents')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Encoded latent space (with unrotation applied)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(encoded_data_3d[:, 0], encoded_data_3d[:, 1], encoded_data_3d[:, 2],
                c=colors, s=20, alpha=0.9)
    ax2.set_title('Encoded Latents' + (' (unrotated)' if apply_unrotation else ''))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.show()


def scatter3d_sphere(plt, z, color_source=None, s=10, a=0.8, r=1.03,
                     use_gradient=True, apply_unrotation=False, original_z=None):
    """
    Create a 3D scatter plot on a sphere with gradient coloring.

    Args:
        plt: matplotlib.pyplot module
        z: Points to plot (torch.Tensor or numpy array)
        color_source: Points to use for color computation (torch.Tensor or numpy array).
                      If None, uses z. For encoded data, pass original latents here.
        s: Point size
        a: Alpha (transparency)
        r: Wireframe sphere radius
        use_gradient: If True, use spherical gradient colors. If False, use color_source as RGB.
        apply_unrotation: If True, apply linear unrotation to z using original_z
        original_z: Original latent points for unrotation (required if apply_unrotation=True)

    Returns:
        matplotlib figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Convert to numpy
    if hasattr(z, 'detach'):
        z = z.detach().cpu().numpy()
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)

    # Apply unrotation if requested
    if apply_unrotation and original_z is not None:
        if hasattr(original_z, 'detach'):
            original_z = original_z.detach().cpu().numpy()
        z = linear_unrotation(original_z, z)

    # Compute colors
    if use_gradient:
        if color_source is None:
            color_source = z
        elif hasattr(color_source, 'detach'):
            color_source = color_source.detach().cpu().numpy()
        colors = spherical_gradient_colors(color_source)
    else:
        # Legacy behavior: use color_source directly as color values
        if hasattr(color_source, 'detach'):
            color_source = color_source.detach().cpu().numpy()
        colors = 0.5 * color_source + 0.5

    ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=colors, s=s, alpha=a, zorder=1)

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

