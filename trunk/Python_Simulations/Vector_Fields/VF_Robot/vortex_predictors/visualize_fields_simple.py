"""
Vortex Field Visualization - Simple 1x2 Comparison

Creates a side-by-side comparison:
- Left: Raw training data (quiver plot)
- Right: Analytical field (hue+sat colorwheel with quiver overlay)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fields.environments.Vortex import vortex1


# ==================== DATA LOADING ====================

def grid_based_uniform_sample(df, grid_size=25, max_per_cell=1):
    """
    Fast grid-based uniform sampling for cleaner visualization.

    Args:
        df: DataFrame with x, y columns
        grid_size: Number of grid cells per dimension
        max_per_cell: Maximum points to keep per grid cell

    Returns:
        Indices of selected points
    """
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()

    # Create grid bins
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)

    # Assign each point to a grid cell
    x_indices = np.digitize(df['x'].values, x_bins) - 1
    y_indices = np.digitize(df['y'].values, y_bins) - 1

    # Clip to valid range
    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)

    # Group points by grid cell
    selected = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Find points in this cell
            mask = (x_indices == i) & (y_indices == j)
            cell_indices = np.where(mask)[0]

            if len(cell_indices) > 0:
                # Randomly select up to max_per_cell points
                n_select = min(max_per_cell, len(cell_indices))
                selected.extend(np.random.choice(cell_indices, n_select, replace=False))

    return selected


def load_training_data(filename='all_vortex_data_1.csv', max_points=200):
    """
    Load and process training data for visualization.

    Returns:
        df: DataFrame with columns [x, y, hue, sat]
    """
    print(f"Loading training data from {filename}...")
    df = pd.read_csv(filename)

    # Stack the three measurement sets
    stacked_data = []

    for i in range(len(df)):
        # First measurement set (negate x,y to match simulation coordinates)
        stacked_data.append([
            -df.iloc[i, 0],  # x1 (negated)
            -df.iloc[i, 1],  # y1 (negated)
            df.iloc[i, 3],   # hue1
            df.iloc[i, 4]    # sat1
        ])
        # Second measurement set
        stacked_data.append([
            -df.iloc[i, 5],  # x2 (negated)
            -df.iloc[i, 6],  # y2 (negated)
            df.iloc[i, 8],   # hue2
            df.iloc[i, 9]    # sat2
        ])
        # Third measurement set
        stacked_data.append([
            -df.iloc[i, 10], # x3 (negated)
            -df.iloc[i, 11], # y3 (negated)
            df.iloc[i, 13],  # hue3
            df.iloc[i, 14]   # sat3
        ])

    # Create stacked dataframe
    stacked_df = pd.DataFrame(stacked_data, columns=['x', 'y', 'hue', 'sat'])
    print(f"Stacked {len(df)} rows → {len(stacked_df)} data points")

    # Filter to domain
    stacked_df = stacked_df[(stacked_df['x'] > -0.6) & (stacked_df['x'] < 0.6) &
                            (stacked_df['y'] > -0.6) & (stacked_df['y'] < 0.6)].copy()
    print(f"After filtering to [-0.6, 0.6]: {len(stacked_df)} points")

    # Grid-based uniform sampling for cleaner visualization
    if len(stacked_df) > max_points:
        print(f"Applying grid-based uniform sampling...")
        selected_indices = grid_based_uniform_sample(stacked_df, grid_size=25, max_per_cell=1)
        stacked_df = stacked_df.iloc[selected_indices].reset_index(drop=True)
        print(f"Selected {len(stacked_df)} uniformly distributed points")

    return stacked_df


# ==================== FIELD EVALUATION ====================

def evaluate_analytical_field(X_grid, Y_grid):
    """Evaluate analytical vortex1 field and convert to hue/saturation"""
    print("\nEvaluating analytical field...")

    # Get vector field components
    U, V = vortex1(X_grid, Y_grid)

    # Rotate 180 degrees: (u, v) → (-u, -v)
    U_rot = -U
    V_rot = -V

    # Convert to hue (direction)
    hue = (np.arctan2(V_rot, U_rot) + np.pi) / (2 * np.pi)
    hue = np.clip(hue, 0, 1)

    # Convert to saturation (magnitude)
    magnitude = np.sqrt(U_rot**2 + V_rot**2)
    sat = magnitude / (np.max(magnitude) + 1e-10)  # Normalize to [0,1]
    sat = np.clip(sat, 0, 1)

    print("✓ Analytical evaluation complete (rotated 180°)")
    return hue, sat, U_rot, V_rot


# ==================== VISUALIZATION ====================

def create_visualization(df, hue_analytical, sat_analytical, U_analytical, V_analytical, X_grid, Y_grid):
    """Create 1x2 comparison figure"""
    print("\nCreating 1×2 visualization...")

    fig = plt.figure(figsize=(16, 8))

    # ========== PANEL 1: Analytical Field (Hue+Sat Colorwheel + Quiver) ==========
    ax1 = plt.subplot(1, 2, 1)

    # Convert hue/sat to RGB for background
    hsv_analytical = np.zeros((X_grid.shape[0], X_grid.shape[1], 3))
    hsv_analytical[:, :, 0] = hue_analytical
    hsv_analytical[:, :, 1] = sat_analytical
    hsv_analytical[:, :, 2] = 1.0
    rgb_analytical = hsv_to_rgb(hsv_analytical)

    ax1.imshow(rgb_analytical, extent=[-0.65, 0.65, -0.65, 0.65], origin='lower', aspect='auto')

    # Add quiver overlay with reduced density for clarity (negated arrows, colored by hue)
    skip = 7  # Show every 7th arrow
    quiv1 = ax1.quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip],
                       -U_analytical[::skip, ::skip], -V_analytical[::skip, ::skip],
                       hue_analytical[::skip, ::skip], cmap='hsv',
                       alpha=0.8, scale=10, width=0.004)

    ax1.set_title('Analytical Field (with Vector Overlay)', fontsize=16, fontweight='bold')
    ax1.set_xlim(-0.65, 0.65)
    ax1.set_ylim(-0.65, 0.65)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)

    # ========== PANEL 2: Raw Training Data (Quiver) ==========
    ax2 = plt.subplot(1, 2, 2)

    # Convert hue/sat to vectors for quiver plot
    angles = df['hue'].values * 2 * np.pi  # Convert hue to angle
    magnitudes = df['sat'].values

    U_data = magnitudes * np.cos(angles)
    V_data = magnitudes * np.sin(angles)

    # Color by hue
    colors_data = df['hue'].values

    quiv2 = ax2.quiver(df['x'].values, df['y'].values, U_data, V_data, colors_data,
                       cmap='hsv', scale=8, alpha=0.7, width=0.003)
    ax2.set_title('Raw Training Data', fontsize=16, fontweight='bold')
    ax2.set_xlim(-0.65, 0.65)
    ax2.set_ylim(-0.65, 0.65)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)

    plt.tight_layout(pad=2.0)

    # Save figure
    output_path = 'field_comparison_simple.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.show()


# ==================== MAIN ====================

def main():
    print("=" * 80)
    print("VORTEX FIELD VISUALIZATION - SIMPLE 1×2 COMPARISON")
    print("=" * 80)

    # Change to script directory to find data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}\n")

    # Load training data
    df = load_training_data('all_vortex_data_1.csv', max_points=200)

    # Create evaluation grid
    print("\nCreating evaluation grid...")
    x_grid = np.linspace(-0.65, 0.65, 100)
    y_grid = np.linspace(-0.65, 0.65, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    print("✓ Grid created (100×100)")

    # Evaluate analytical field
    hue_analytical, sat_analytical, U_analytical, V_analytical = evaluate_analytical_field(X_grid, Y_grid)

    # Create visualization
    create_visualization(df, hue_analytical, sat_analytical, U_analytical, V_analytical, X_grid, Y_grid)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
