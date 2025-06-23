
def plot_scores(plt, scores):
    linear_scores = scores['linear_scores']
    perm_scores = scores['perm_scores']
    angle_preservation_errors = scores['angle_preservation_errors']
    eval_losses = scores['eval_losses']

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))

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

    # Plot angle preservation errors
    axes[2].plot(angle_preservation_errors, label='Angle Preservation Errors', color='red')
    axes[2].set_xlabel('Checkpoints')
    axes[2].set_ylabel('Error')
    axes[2].set_title('Angle Preservation Errors')
    axes[2].legend()
    axes[2].grid(True)

    # Plot eval losses
    axes[3].plot(eval_losses, label='Eval Losses', color='green')
    axes[3].set_xlabel('Checkpoints')
    axes[3].set_ylabel('Loss')
    axes[3].set_title('Evaluation Losses')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()