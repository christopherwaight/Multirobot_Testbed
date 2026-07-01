"""
Demo: All Available Saddle Field Types

Non-interactive visualization showing:
- 6 saddle field variants with contours
- Critical point locations
- Eigenvalue analysis at saddle points
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.fields.scalar_utils import analyze_critical_point
from src.fields.scalar_environments.Saddle import (
    bimodal_saddle1, bimodal_saddle2, bimodal_saddle3,
    quadrimodal_saddle1, offset_saddle1, asymmetric_saddle1
)

# Grid for evaluation
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Field configurations
fields = [
    (bimodal_saddle1, "Bimodal Saddle 1\n(Standard)", (0, 0)),
    (bimodal_saddle2, "Bimodal Saddle 2\n(Closer Peaks)", (0, 0)),
    (bimodal_saddle3, "Bimodal Saddle 3\n(Vertical)", (0, 0)),
    (quadrimodal_saddle1, "Quadrimodal Saddle\n(4 Peaks)", (0, 0)),
    (offset_saddle1, "Offset Saddle\n(Not at origin)", (0.5, 0.5)),
    (asymmetric_saddle1, "Asymmetric Saddle\n(Different heights)", (0, 0))
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (field_func, title, saddle_pos) in enumerate(fields):
    ax = axes[idx]

    # Evaluate field
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = field_func(X[i, j], Y[i, j])

    # Contour plot
    contours = ax.contour(X, Y, Z, levels=20, cmap='RdBu_r', alpha=0.6)
    ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r', alpha=0.3)

    # Analyze saddle point
    analysis = analyze_critical_point(field_func, saddle_pos[0], saddle_pos[1])

    # Mark saddle location
    ax.plot(saddle_pos[0], saddle_pos[1], 'k*', markersize=15, label='Saddle')

    # Eigenvalue info
    eigenvals = analysis['eigenvalues']
    classification = analysis['classification']
    grad_norm = np.linalg.norm(analysis['gradient'])

    # Add text info
    info_text = (f"{classification.upper()}\n"
                f"λ₁={eigenvals[0]:.2f}, λ₂={eigenvals[1]:.2f}\n"
                f"|∇z|={grad_norm:.2e}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Eigenvector directions (if saddle)
    if classification == 'saddle':
        eigenvecs = analysis['eigenvectors']
        scale = 0.5
        # Positive eigenvalue direction (repulsive)
        ax.arrow(saddle_pos[0], saddle_pos[1],
                eigenvecs[0, 1] * scale, eigenvecs[1, 1] * scale,
                head_width=0.1, head_length=0.1, fc='red', ec='red',
                alpha=0.7, linewidth=2)
        # Negative eigenvalue direction (attractive)
        ax.arrow(saddle_pos[0], saddle_pos[1],
                eigenvecs[0, 0] * scale, eigenvecs[1, 0] * scale,
                head_width=0.1, head_length=0.1, fc='green', ec='green',
                alpha=0.7, linewidth=2)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

# Legend for eigenvectors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Attractive (λ < 0)'),
    Patch(facecolor='red', alpha=0.7, label='Repulsive (λ > 0)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('demo_scalar_fields.png', dpi=300, bbox_inches='tight')
print("Saved: demo_scalar_fields.png")
print("\nField Comparison Complete!")
print("="*50)
for field_func, title, saddle_pos in fields:
    analysis = analyze_critical_point(field_func, saddle_pos[0], saddle_pos[1])
    eigenvals = analysis['eigenvalues']
    print(f"{title.split(chr(10))[0]:25} → λ₁={eigenvals[0]:6.2f}, λ₂={eigenvals[1]:6.2f}")
print("="*50)

plt.show()
