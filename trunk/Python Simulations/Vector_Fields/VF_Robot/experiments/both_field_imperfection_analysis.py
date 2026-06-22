#!/usr/bin/env python3
"""
Both Field Imperfection Analysis

Lightweight script to compare vortex and saddle field reconstructions.
Creates a single 2x3 figure (2 columns: vortex/saddle, 3 rows: analytical/reconstructed/error).

Output: reconstruction_comparison.png in experiments folder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fields.field_types import AnalyticalField, RBFField
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Saddle import saddle1

# Configuration
GRID_RESOLUTION = 60  # Originally 50
ARROW_SKIP = 3          # Originally 3
MAX_RAW_POINTS = 200    # Originally 200

# Predictor directories (relative to VF_Robot root)
VORTEX_PREDICTOR_DIR = 'vortex_predictors'
SADDLE_PREDICTOR_DIR = 'saddle_predictors'

# Raw data files
VORTEX_DATA_FILE = 'all_vortex_data_1.csv'
SADDLE_DATA_FILE = 'saddle_data.csv'

# Output
OUTPUT_FILE = 'reconstruction_comparison.png'


def create_evaluation_grid(resolution=50):
    """Create evaluation grid in simulation coordinates [-0.5, 0.5]."""
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y, X.flatten(), Y.flatten()


def evaluate_field(field, x_flat, y_flat):
    """Evaluate field at given coordinates."""
    u_flat, v_flat = [], []
    for x, y in zip(x_flat, y_flat):
        u, v = field.get_value(x, y)
        u_flat.append(u)
        v_flat.append(v)
    return np.array(u_flat), np.array(v_flat)


def grid_based_uniform_sample(df, grid_size=25, max_per_cell=1):
    """Fast grid-based uniform sampling for cleaner visualization."""
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()

    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)

    x_indices = np.digitize(df['x'].values, x_bins) - 1
    y_indices = np.digitize(df['y'].values, y_bins) - 1

    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)

    selected = []
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x_indices == i) & (y_indices == j)
            cell_indices = np.where(mask)[0]
            if len(cell_indices) > 0:
                n_select = min(max_per_cell, len(cell_indices))
                selected.extend(np.random.choice(cell_indices, n_select, replace=False))

    return selected


def load_vortex_raw_data(predictor_dir, filename, max_points=200):
    """Load and process vortex raw training data."""
    predictor_path = os.path.join(os.path.dirname(__file__), '..', predictor_dir, filename)

    if not os.path.exists(predictor_path):
        print(f"Warning: Vortex data not found at {predictor_path}")
        return None

    print(f"Loading vortex raw data from {filename}...")
    df = pd.read_csv(predictor_path)

    # Stack the three measurement sets
    stacked_data = []
    for i in range(len(df)):
        # Negate x,y to match simulation coordinates
        stacked_data.append([-df.iloc[i, 0], -df.iloc[i, 1], df.iloc[i, 3], df.iloc[i, 4]])
        stacked_data.append([-df.iloc[i, 5], -df.iloc[i, 6], df.iloc[i, 8], df.iloc[i, 9]])
        stacked_data.append([-df.iloc[i, 10], -df.iloc[i, 11], df.iloc[i, 13], df.iloc[i, 14]])

    stacked_df = pd.DataFrame(stacked_data, columns=['x', 'y', 'hue', 'sat'])

    # Filter to domain
    stacked_df = stacked_df[(stacked_df['x'] > -0.5) & (stacked_df['x'] < 0.5) &
                            (stacked_df['y'] > -0.5) & (stacked_df['y'] < 0.5)].copy()

    # Uniform sampling
    if len(stacked_df) > max_points:
        selected_indices = grid_based_uniform_sample(stacked_df, grid_size=25, max_per_cell=1)
        stacked_df = stacked_df.iloc[selected_indices].reset_index(drop=True)

    # Convert hue/sat to u,v
    angles = stacked_df['hue'].values * 2 * np.pi
    magnitudes = stacked_df['sat'].values
    stacked_df['u'] = magnitudes * np.cos(angles)
    stacked_df['v'] = magnitudes * np.sin(angles)

    print(f"Loaded {len(stacked_df)} vortex data points")
    return stacked_df[['x', 'y', 'u', 'v']]


def load_saddle_raw_data(predictor_dir, filename, max_points=200):
    """Load and process saddle raw training data with SWAP_NEGATE_ROTATED transform."""
    predictor_path = os.path.join(os.path.dirname(__file__), '..', predictor_dir, filename)

    if not os.path.exists(predictor_path):
        print(f"Warning: Saddle data not found at {predictor_path}")
        return None

    print(f"Loading saddle raw data from {filename}...")
    df = pd.read_csv(predictor_path)

    # Stack with SWAP_NEGATE_ROTATED transformation
    stacked_data = []
    for i in range(len(df)):
        # Apply transformation for each measurement set
        for col_offset in [0, 5, 10]:
            x_temp = -df.iloc[i, col_offset + 1]  # -y_raw
            y_temp = -df.iloc[i, col_offset + 0]  # -x_raw
            x_new = y_temp  # -x_raw
            y_new = -x_temp  # y_raw
            hue_new = (df.iloc[i, col_offset + 3] - 0.25) % 1.0  # Rotate hue -90°
            sat = df.iloc[i, col_offset + 4]
            stacked_data.append([x_new, y_new, hue_new, sat])

    stacked_df = pd.DataFrame(stacked_data, columns=['x', 'y', 'hue', 'sat'])

    # Filter to domain
    stacked_df = stacked_df[(stacked_df['x'] > -0.5) & (stacked_df['x'] < 0.5) &
                            (stacked_df['y'] > -0.5) & (stacked_df['y'] < 0.5)].copy()

    # Uniform sampling
    if len(stacked_df) > max_points:
        selected_indices = grid_based_uniform_sample(stacked_df, grid_size=25, max_per_cell=1)
        stacked_df = stacked_df.iloc[selected_indices].reset_index(drop=True)

    # Convert hue/sat to u,v
    angles = stacked_df['hue'].values * 2 * np.pi
    magnitudes = stacked_df['sat'].values
    stacked_df['u'] = magnitudes * np.cos(angles)
    stacked_df['v'] = magnitudes * np.sin(angles)

    print(f"Loaded {len(stacked_df)} saddle data points")
    return stacked_df[['x', 'y', 'u', 'v']]


def compute_hue(U, V):
    """Convert vector field to hue values for HSV colormap."""
    angles = np.arctan2(V, U)
    hue = (angles + np.pi) / (2 * np.pi)
    return hue


def plot_comparison(X, Y,
                   vortex_analytical, vortex_raw, vortex_rbf,
                   saddle_analytical, saddle_raw, saddle_rbf,
                   filename):
    """Create 2x3 comparison plot: Vortex (left) vs Saddle (right)."""

    fig = plt.figure(figsize=(14, 21))

    # Unpack data
    u_v_anal, v_v_anal = vortex_analytical
    u_s_anal, v_s_anal = saddle_analytical
    u_v_rbf, v_v_rbf = vortex_rbf
    u_s_rbf, v_s_rbf = saddle_rbf

    # Reshape for plotting
    U_v_anal = u_v_anal.reshape(X.shape)
    V_v_anal = v_v_anal.reshape(X.shape)
    U_s_anal = u_s_anal.reshape(X.shape)
    V_s_anal = v_s_anal.reshape(X.shape)
    U_v_rbf = u_v_rbf.reshape(X.shape)
    V_v_rbf = v_v_rbf.reshape(X.shape)
    U_s_rbf = u_s_rbf.reshape(X.shape)
    V_s_rbf = v_s_rbf.reshape(X.shape)

    # Compute hues
    hue_v_anal = compute_hue(U_v_anal, V_v_anal)
    hue_s_anal = compute_hue(U_s_anal, V_s_anal)

    # Row 1: Analytical fields
    # Vortex Analytical
    ax1 = plt.subplot(3, 2, 1)
    ax1.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
               U_v_anal[::ARROW_SKIP, ::ARROW_SKIP], V_v_anal[::ARROW_SKIP, ::ARROW_SKIP],
               hue_v_anal[::ARROW_SKIP, ::ARROW_SKIP],
               cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax1.set_title('Analytical', fontsize=20, fontweight='bold')
    ax1.set_xlabel('x (m)', fontsize=18)
    ax1.set_ylabel('y (m)', fontsize=18)
    ax1.tick_params(labelsize=16)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Saddle Analytical
    ax2 = plt.subplot(3, 2, 2)
    ax2.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
               U_s_anal[::ARROW_SKIP, ::ARROW_SKIP], V_s_anal[::ARROW_SKIP, ::ARROW_SKIP],
               hue_s_anal[::ARROW_SKIP, ::ARROW_SKIP],
               cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax2.set_title('Analytical', fontsize=20, fontweight='bold')
    ax2.set_xlabel('x (m)', fontsize=18)
    ax2.set_ylabel('y (m)', fontsize=18)
    ax2.tick_params(labelsize=16)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Row 2: Raw/Reconstructed data
    # Vortex Reconstructed
    ax3 = plt.subplot(3, 2, 3)
    if vortex_raw is not None:
        angles_raw = np.arctan2(vortex_raw['v'].values, vortex_raw['u'].values)
        hue_raw = (angles_raw + np.pi) / (2 * np.pi)
        ax3.quiver(vortex_raw['x'].values, vortex_raw['y'].values,
                   vortex_raw['u'].values, vortex_raw['v'].values,
                   hue_raw, cmap='hsv', alpha=0.7, scale=8, width=0.003)
    ax3.set_title('Reconstructed', fontsize=20, fontweight='bold')
    ax3.set_xlabel('x (m)', fontsize=18)
    ax3.set_ylabel('y (m)', fontsize=18)
    ax3.tick_params(labelsize=16)
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Saddle Reconstructed
    ax4 = plt.subplot(3, 2, 4)
    if saddle_raw is not None:
        angles_raw = np.arctan2(saddle_raw['v'].values, saddle_raw['u'].values)
        hue_raw = (angles_raw + np.pi) / (2 * np.pi)
        ax4.quiver(saddle_raw['x'].values, saddle_raw['y'].values,
                   saddle_raw['u'].values, saddle_raw['v'].values,
                   hue_raw, cmap='hsv', alpha=0.7, scale=8, width=0.003)
    ax4.set_title('Reconstructed', fontsize=20, fontweight='bold')
    ax4.set_xlabel('x (m)', fontsize=18)
    ax4.set_ylabel('y (m)', fontsize=18)
    ax4.tick_params(labelsize=16)
    ax4.set_xlim(-0.5, 0.5)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # Row 3: Reconstruction Errors
    # Vortex Error
    ax5 = plt.subplot(3, 2, 5)
    U_v_error = U_v_anal - U_v_rbf
    V_v_error = V_v_anal - V_v_rbf
    hue_v_error = compute_hue(U_v_error, V_v_error)
    error_scale = 3.0
    ax5.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
               U_v_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale,
               V_v_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale,
               hue_v_error[::ARROW_SKIP, ::ARROW_SKIP],
               cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax5.set_title('Reconstruction Error', fontsize=20, fontweight='bold')
    ax5.set_xlabel('x (m)', fontsize=18)
    ax5.set_ylabel('y (m)', fontsize=18)
    ax5.tick_params(labelsize=16)
    ax5.set_xlim(-0.5, 0.5)
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # Saddle Error
    ax6 = plt.subplot(3, 2, 6)
    U_s_error = U_s_anal - U_s_rbf
    V_s_error = V_s_anal - V_s_rbf
    hue_s_error = compute_hue(U_s_error, V_s_error)
    ax6.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
               U_s_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale,
               V_s_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale,
               hue_s_error[::ARROW_SKIP, ::ARROW_SKIP],
               cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax6.set_title('Reconstruction Error', fontsize=20, fontweight='bold')
    ax6.set_xlabel('x (m)', fontsize=18)
    ax6.set_ylabel('y (m)', fontsize=18)
    ax6.tick_params(labelsize=16)
    ax6.set_xlim(-0.5, 0.5)
    ax6.set_ylim(-0.5, 0.5)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to experiments folder
    output_path = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("FIELD RECONSTRUCTION COMPARISON")
    print("=" * 80)
    print(f"\nCreating 2×3 comparison: Vortex (left) vs Saddle (right)")
    print(f"Grid Resolution: {GRID_RESOLUTION} × {GRID_RESOLUTION}\n")

    # Load analytical fields
    print("Loading analytical fields...")
    field_vortex_analytical = AnalyticalField(vortex1)
    field_saddle_analytical = AnalyticalField(saddle1)
    print("✓ Analytical fields loaded")

    # Load RBF fields
    print("Loading RBF fields...")
    field_vortex_rbf = RBFField(predictor_dir=VORTEX_PREDICTOR_DIR)
    field_saddle_rbf = RBFField(predictor_dir=SADDLE_PREDICTOR_DIR)
    print("✓ RBF fields loaded")

    # Create evaluation grid
    print("Creating evaluation grid...")
    X, Y, x_flat, y_flat = create_evaluation_grid(GRID_RESOLUTION)
    print("✓ Grid created")

    # Evaluate fields
    print("Evaluating vortex fields...")
    u_v_anal, v_v_anal = evaluate_field(field_vortex_analytical, x_flat, y_flat)
    u_v_rbf, v_v_rbf = evaluate_field(field_vortex_rbf, x_flat, y_flat)
    print("✓ Vortex fields evaluated")

    print("Evaluating saddle fields...")
    u_s_anal, v_s_anal = evaluate_field(field_saddle_analytical, x_flat, y_flat)
    u_s_rbf, v_s_rbf = evaluate_field(field_saddle_rbf, x_flat, y_flat)
    print("✓ Saddle fields evaluated")

    # Load raw training data
    print("Loading raw training data...")
    vortex_raw = load_vortex_raw_data(VORTEX_PREDICTOR_DIR, VORTEX_DATA_FILE, MAX_RAW_POINTS)
    saddle_raw = load_saddle_raw_data(SADDLE_PREDICTOR_DIR, SADDLE_DATA_FILE, MAX_RAW_POINTS)
    print("✓ Raw data loaded")

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(X, Y,
                   (u_v_anal, v_v_anal), vortex_raw, (u_v_rbf, v_v_rbf),
                   (u_s_anal, v_s_anal), saddle_raw, (u_s_rbf, v_s_rbf),
                   OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
