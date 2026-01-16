"""
Panel (C): Progressive Diversity Violation - Linear Identifiability vs Fixed Dimensions
Shows how performance degrades as more dimensions are fixed in augmentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data path
CSV_PATH = "/Users/justin/Desktop/ML_research/CLTheory/experiments/diversity_violated/experiment_results_10d/progressive_d_fixed_10d_70d_summary_20250814_061808.csv"
OUTPUT_PATH = "/Users/justin/Desktop/ML_research/CLTheory/experiments"

# Load data
df = pd.read_csv(CSV_PATH, header=[0, 1], index_col=[0, 1])

# Extract data for each process
linear_data = df.loc['InjectiveLinearDecoder_10D_70D']
monomial_data = df.loc['MonomialEmbedding_10D_70D']

# Get d_fixed values and metrics
d_fixed = linear_data.index.astype(int).values
linear_mean = linear_data[('linear_score', 'mean')].values
linear_std = linear_data[('linear_score', 'std')].values
monomial_mean = monomial_data[('linear_score', 'mean')].values
monomial_std = monomial_data[('linear_score', 'std')].values

# Print data for verification
print("Progressive Diversity Violation Data:")
print(f"{'d_fixed':<10} {'Linear Mean':<12} {'Linear Std':<12} {'Monomial Mean':<14} {'Monomial Std':<12}")
print("-" * 60)
for i, d in enumerate(d_fixed):
    print(f"{d:<10} {linear_mean[i]:<12.4f} {linear_std[i]:<12.4f} {monomial_mean[i]:<14.4f} {monomial_std[i]:<12.4f}")

# Create figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
})

fig, ax = plt.subplots(figsize=(8, 5))

# Colors (matching Panels A and B)
COLOR_LINEAR = '#1f77b4'    # Blue
COLOR_MONOMIAL = '#ff7f0e'  # Orange

# Plot Linear process with shaded error region
ax.fill_between(d_fixed, linear_mean - linear_std, linear_mean + linear_std,
                color=COLOR_LINEAR, alpha=0.2)
ax.plot(d_fixed, linear_mean, color=COLOR_LINEAR, linewidth=2, marker='o', markersize=4, label='Linear')

# Plot Monomial process with shaded error region
ax.fill_between(d_fixed, monomial_mean - monomial_std, monomial_mean + monomial_std,
                color=COLOR_MONOMIAL, alpha=0.2)
ax.plot(d_fixed, monomial_mean, color=COLOR_MONOMIAL, linewidth=2, marker='o', markersize=4, label='Monomial')

# Styling
ax.set_xlabel('Number of Fixed Dimensions ($d_{fixed}$)')
ax.set_ylabel('Linear Identifiability (R²)')
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(0, 1.05)
ax.set_xticks(range(0, 11))

# Legend
ax.legend(loc='upper right', framealpha=0.95, fontsize=12)

plt.tight_layout()

# Save figures
plt.savefig(f"{OUTPUT_PATH}/progressive_violation_lineplot.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_PATH}/progressive_violation_lineplot.pdf", bbox_inches='tight')
print(f"\nSaved: {OUTPUT_PATH}/progressive_violation_lineplot.png")
print(f"Saved: {OUTPUT_PATH}/progressive_violation_lineplot.pdf")
