"""
Panel (B): Line plot showing Angular Preservation Error vs Constraint Ratio.
Shows MLP encoder performance converging to Inverse encoder as rho -> 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

BASE_PATH = "/Users/justin/Desktop/ML_research/CLTheory/experiments/diversity_violated_adjusted/raw_results"

# File paths
MLP_FILES = [
    f"{BASE_PATH}/g_identity_f_mlp.csv",
    f"{BASE_PATH}/g_linear_f_mlp.csv",
    f"{BASE_PATH}/g_mlp_f_mlp.csv",
    f"{BASE_PATH}/g_patches_f_mlp.csv",
    f"{BASE_PATH}/g_spiral_f_mlp.csv",
]

INVERSE_FILES = [
    f"{BASE_PATH}/g_identity_f_linear.csv",
    f"{BASE_PATH}/g_linear_f_linear.csv",
    f"{BASE_PATH}/g_patches_f_inverse_patches.csv",
    f"{BASE_PATH}/g_spiral_f_inverse_spiral.csv",
]


def parse_mean_std(value):
    """Parse 'mean±std' format or return float directly."""
    if isinstance(value, str) and '±' in value:
        parts = value.split('±')
        return float(parts[0]), float(parts[1])
    else:
        return float(value), 0.0


def load_angle_data(filepath):
    """Load CSV and extract Angle (APE) data for each constraint ratio."""
    df = pd.read_csv(filepath)

    ratios = []
    means = []
    stds = []

    for _, row in df.iterrows():
        ratio = float(row['Ratio'])

        # Handle two CSV formats
        if 'Angle_Std' in df.columns:
            # Format 2: separate columns
            mean = float(row['Angle'])
            std = float(row['Angle_Std'])
        else:
            # Format 1: combined mean±std
            mean, std = parse_mean_std(row['Angle'])

        ratios.append(ratio)
        means.append(mean)
        stds.append(std)

    return np.array(ratios), np.array(means), np.array(stds)


def aggregate_across_processes(file_list):
    """Load data from multiple files and compute mean/std across processes."""
    all_data = []

    for filepath in file_list:
        ratios, means, stds = load_angle_data(filepath)
        all_data.append(means)

    # Stack: shape (n_processes, n_ratios)
    all_data = np.array(all_data)

    # Compute mean and std across processes (axis=0)
    process_mean = np.mean(all_data, axis=0)
    process_std = np.std(all_data, axis=0)

    return ratios, process_mean, process_std


# Load and aggregate data
print("Loading MLP encoder data (5 processes)...")
ratios, mlp_mean, mlp_std = aggregate_across_processes(MLP_FILES)
print(f"  Ratios: {ratios}")
print(f"  MLP APE means: {mlp_mean}")

print("\nLoading Inverse encoder data (4 processes)...")
_, inverse_mean, inverse_std = aggregate_across_processes(INVERSE_FILES)
print(f"  Inverse APE means: {inverse_mean}")

# Print summary
print("\n" + "="*60)
print("Summary: Angular Preservation Error vs Constraint Ratio")
print("="*60)
print(f"{'Ratio':<8} {'MLP Mean':<12} {'MLP Std':<12} {'Inv Mean':<12} {'Inv Std':<12}")
print("-"*60)
for i, r in enumerate(ratios):
    print(f"{r:<8.1f} {mlp_mean[i]:<12.4f} {mlp_std[i]:<12.4f} {inverse_mean[i]:<12.4f} {inverse_std[i]:<12.4f}")

# Create figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
})

fig, ax = plt.subplots(figsize=(8, 5))

# Colors (matching Panel A)
COLOR_MLP = '#1f77b4'      # Blue
COLOR_INVERSE = '#ff7f0e'  # Orange

# Plot MLP line with shaded error region
ax.fill_between(ratios, mlp_mean - mlp_std, mlp_mean + mlp_std,
                color=COLOR_MLP, alpha=0.2)
ax.plot(ratios, mlp_mean, color=COLOR_MLP, linewidth=2, marker='o', markersize=4, label='MLP')

# Plot Inverse line with shaded error region
ax.fill_between(ratios, inverse_mean - inverse_std, inverse_mean + inverse_std,
                color=COLOR_INVERSE, alpha=0.2)
ax.plot(ratios, inverse_mean, color=COLOR_INVERSE, linewidth=2, marker='o', markersize=4, label='Inverse')

# Vertical line at ρ=1.0 to emphasize convergence point
ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Styling
ax.set_xlabel('Constraint Ratio (ρ)')
ax.set_ylabel('Angular Preservation Error (APE)')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 0.35)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# No grid to match Panel (a)

# Legend
ax.legend(loc='upper right', framealpha=0.95, fontsize=12)

plt.tight_layout()

# Save figures
OUTPUT_PATH = "/Users/justin/Desktop/ML_research/CLTheory/experiments"
plt.savefig(f"{OUTPUT_PATH}/constraint_ratio_ape_lineplot.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_PATH}/constraint_ratio_ape_lineplot.pdf", bbox_inches='tight')
print(f"\nSaved: {OUTPUT_PATH}/constraint_ratio_ape_lineplot.png")
print(f"Saved: {OUTPUT_PATH}/constraint_ratio_ape_lineplot.pdf")

plt.show()
