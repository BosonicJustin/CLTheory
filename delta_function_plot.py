import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': False,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'figure.figsize': (8, 6)
})

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

x = np.linspace(-3, 3, 1000)
y = np.zeros_like(x)

ax.plot(x, y, 'k-', linewidth=2, label='δ(x) = 0 for x ≠ 0')

arrow = FancyArrowPatch((0, 0), (0, 1.5), 
                       arrowstyle='->', 
                       mutation_scale=20, 
                       linewidth=3, 
                       color='red')
ax.add_patch(arrow)

ax.text(0.1, 1.6, 'δ(0) = ∞', fontsize=14, color='red')
ax.text(0.1, 0.8, '∫_{-∞}^{∞} δ(x) dx = 1', fontsize=12, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

ax.axhline(y=0, color='black', linewidth=1.5)
ax.axvline(x=0, color='black', linewidth=1.5)

ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 2)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('δ(x)', fontsize=14)
ax.set_title('Dirac Delta Function δ(x)', fontsize=16, pad=20)

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([0, 0.5, 1, 1.5])

plt.tight_layout()
plt.savefig('/Users/justin/Desktop/CLTheory/delta_function.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/justin/Desktop/CLTheory/delta_function.pdf', bbox_inches='tight')
plt.show()

print("Delta function plot saved as delta_function.png and delta_function.pdf")