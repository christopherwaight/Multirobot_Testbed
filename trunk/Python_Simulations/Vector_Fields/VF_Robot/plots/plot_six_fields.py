"""
Plot 6 vector field environments in a 2x3 grid.
Displays: vortex1, sinking_vortex1, sink1, source1, spewing_vortex1, saddle1

This creates figure 1 in the paper.
Figure 2 and 4 are in powerpoint
figure 3 is borrowed from another paper
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all 6 vector field environments
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Sinking_Vortex import sinking_vortex1
from src.fields.environments.Sink import sink1
from src.fields.environments.Source import source1
from src.fields.environments.Spewing_Vortex import spewing_vortex1
from src.fields.environments.Saddle import saddle1


def plot_vector_fields():
    """Create a 2x3 grid of vector field plots."""

    # Define the grid for plotting
    x = np.linspace(-0.6, 0.6, 12)
    y = np.linspace(-0.6, 0.6, 12)
    X, Y = np.meshgrid(x, y)

    # Define the environments to plot
    environments = [
        (vortex1, "Vortex", "blue"),
        (sinking_vortex1, "Sinking Vortex", "purple"),
        (sink1, "Sink", "darkred"),
        (source1, "Source", "darkgreen"),
        (spewing_vortex1, "Spewing Vortex", "darkorange"),
        (saddle1, "Saddle", "darkblue")
    ]

    # Create figure with 2x3 grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    axes = axes.flatten()

    # Plot each environment
    for idx, (env_func, title, color) in enumerate(environments):
        ax = axes[idx]

        # Calculate the vector field
        U, V = env_func(X, Y)

        # Plot the vector field using quiver with black arrows
        # Make arrows bigger for sinking/spewing vortex
        if title in ["Sinking Vortex", "Spewing Vortex"]:
            quiver = ax.quiver(X, Y, U, V, color='black', alpha=0.7, scale=4, width=0.005)
        else:
            quiver = ax.quiver(X, Y, U, V, color='black', alpha=0.7, scale=8, width=0.005)

        # Mark the critical point
        ax.plot(0, 0, 'ro', markersize=10, label='Critical Point', zorder=5)

        # Set equal aspect ratio and add grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Title and legend
        ax.set_title(title, fontsize=24, fontweight='bold')
        ax.legend(loc='upper right', fontsize=24)

        # Set axis limits
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)

        # Remove tick marks
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), 'six_vector_fields.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    # Also save as PDF
    pdf_path = os.path.join(os.path.dirname(__file__), 'six_vector_fields.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    plot_vector_fields()
