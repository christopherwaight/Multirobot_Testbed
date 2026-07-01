"""
Plot 6 vector field environments in a 2x3 grid.
Displays: vortex1, sinking_vortex1, sink1, source1, spewing_vortex1, saddle1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    x = np.linspace(-0.6, 0.6, 20)
    y = np.linspace(-0.6, 0.6, 20)
    X, Y = np.meshgrid(x, y)

    # Define the environments to plot
    environments = [
        (vortex1, "Vortex 1", "blue"),
        (sinking_vortex1, "Sinking Vortex 1", "purple"),
        (sink1, "Sink 1", "darkred"),
        (source1, "Source 1", "darkgreen"),
        (spewing_vortex1, "Spewing Vortex 1", "darkorange"),
        (saddle1, "Saddle 1", "darkblue")
    ]

    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Plot each environment
    for idx, (env_func, title, color) in enumerate(environments):
        ax = axes[idx]

        # Calculate the vector field
        U, V = env_func(X, Y)

        # Calculate angle (direction) and magnitude
        angles = np.arctan2(V, U)  # Angle in radians [-pi, pi]
        magnitude = np.sqrt(U**2 + V**2)

        # Map angle to hue [0, 1]
        hue = (angles + np.pi) / (2 * np.pi)  # Convert [-pi, pi] to [0, 1]

        # Normalize magnitude for saturation [0, 1]
        mag_max = np.max(magnitude)
        if mag_max > 0:
            saturation = magnitude / mag_max
        else:
            saturation = np.ones_like(magnitude)

        # Set value (brightness) to maximum
        value = np.ones_like(magnitude)

        # Create HSV array and convert to RGB
        hsv = np.stack([hue, saturation, value], axis=-1)
        rgb = mcolors.hsv_to_rgb(hsv)

        # Flatten for quiver (quiver expects colors in shape matching X,Y flattened)
        colors = rgb.reshape(-1, 3)

        # Plot the vector field using quiver with custom colors
        quiver = ax.quiver(X, Y, U, V, color=colors, alpha=0.8, scale=10, width=0.003)

        # Mark the center point
        ax.plot(0, 0, 'ro', markersize=10, label='Center', zorder=5)

        # Set equal aspect ratio and add grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Labels and title
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

        # Set axis limits
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), 'six_vector_fields_color.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    # Also save as PDF
    pdf_path = os.path.join(os.path.dirname(__file__), 'six_vector_fields_color.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    plot_vector_fields()
