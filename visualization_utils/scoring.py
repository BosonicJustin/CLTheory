def plot_scores(plt, scores):
    linear_scores = scores['linear_scores']
    perm_scores = scores['perm_scores']
    angle_preservation_errors = scores['angle_preservation_errors']
    eval_losses = scores['eval_losses']
    
    # Check if positive and negative losses exist
    has_separate_losses = 'eval_pos_losses' in scores and 'eval_neg_losses' in scores
    eval_pos_losses = scores.get('eval_pos_losses', [])
    eval_neg_losses = scores.get('eval_neg_losses', [])

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

    # Plot eval losses (total, positive, and negative if available)
    axes[3].plot(eval_losses, label='Total Loss', color='green', linewidth=2)
    
    if has_separate_losses and eval_pos_losses and eval_neg_losses:
        axes[3].plot(eval_pos_losses, label='Positive Loss', color='blue', linestyle='--', alpha=0.7)
        axes[3].plot(eval_neg_losses, label='Negative Loss', color='red', linestyle='--', alpha=0.7)
    
    axes[3].set_xlabel('Checkpoints')
    axes[3].set_ylabel('Loss')
    axes[3].set_title('Evaluation Losses')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()