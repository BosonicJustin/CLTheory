
def plot_scores(plt, scores):
    linear_scores = scores['linear_scores']
    perm_scores = scores['perm_scores']

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot linear scores
    axes[0].plot(linear_scores, label='Linear Scores', color='blue')
    axes[0].set_xlabel('Checkpoints')
    axes[0].set_ylabel('Scores')
    axes[0].set_title('Linear Scores')
    axes[0].legend()
    axes[0].grid(True)

    # Plot permutation scores
    axes[1].plot(perm_scores, label='Permutation Scores', color='orange')
    axes[1].set_xlabel('Checkpoints')
    axes[1].set_ylabel('Scores')
    axes[1].set_title('Permutation Scores')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()