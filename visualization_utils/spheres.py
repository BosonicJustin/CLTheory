
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