#!/usr/bin/env python3
"""
Field Imperfection Analysis

Compares NN and RBF field approximations against analytical ground truth,
accounting for real-world scale discrepancy.

Scale Issue:
- Simulation domain: 1m ([-0.5, 0.5])
- Physical printout: 66 inches = 1.6764 meters
- Scale factor: 1.6764

This script generates:
1. Vector field difference plots (NN vs Analytical, RBF vs Analytical)
2. Statistical analysis of errors
3. Spatial error heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fields.field_types import AnalyticalField, NNField, RBFField
from src.fields.environments.Vortex import vortex1

# Configuration
SCALE_FACTOR = 1.6764  # Physical world is 1.6764x larger than simulation
SIMULATION_DOMAIN = 1.0  # meters ([-0.5, 0.5])
PHYSICAL_DOMAIN = SIMULATION_DOMAIN * SCALE_FACTOR  # 1.6764 meters

# Grid resolution
GRID_RESOLUTION = 50  # 50x50 grid
PREDICTOR_DIR = 'vortex_predictors'

# Output directory
OUTPUT_DIR = 'vortex_field_analysis_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_evaluation_grid(resolution=50):
    """
    Create evaluation grid in simulation coordinates.

    Returns:
        X, Y: Meshgrid arrays
        x_flat, y_flat: Flattened coordinate arrays
    """
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y, X.flatten(), Y.flatten()


def evaluate_field(field, x_flat, y_flat):
    """
    Evaluate field at given coordinates.

    Returns:
        u, v: Field components (flattened)
    """
    u_flat = []
    v_flat = []

    for x, y in zip(x_flat, y_flat):
        u, v = field.get_value(x, y)
        u_flat.append(u)
        v_flat.append(v)

    return np.array(u_flat), np.array(v_flat)


def grid_based_uniform_sample(df, grid_size=25, max_per_cell=1):
    """
    Fast grid-based uniform sampling for cleaner visualization.
    (Copied from visualize_simple.py)
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


def load_raw_training_data(filename='all_vortex_data_1.csv', max_points=200):
    """
    Load and process raw training data for comparison.
    Matches visualize_simple.py processing exactly.

    Returns:
        df: DataFrame with columns [x, y, u, v]
    """
    predictor_path = os.path.join(os.path.dirname(__file__), '..', PREDICTOR_DIR, filename)

    if not os.path.exists(predictor_path):
        print(f"Warning: Training data not found at {predictor_path}")
        return None

    print(f"Loading raw training data from {filename}...")
    df = pd.read_csv(predictor_path)

    # Stack the three measurement sets (keeping hue/sat initially)
    stacked_data = []

    for i in range(len(df)):
        # First measurement set (negate x,y to match simulation coordinates)
        stacked_data.append([
            -df.iloc[i, 0],   # x1 (negated)
            -df.iloc[i, 1],   # y1 (negated)
            df.iloc[i, 3],    # hue1
            df.iloc[i, 4]     # sat1
        ])
        # Second measurement set
        stacked_data.append([
            -df.iloc[i, 5],   # x2 (negated)
            -df.iloc[i, 6],   # y2 (negated)
            df.iloc[i, 8],    # hue2
            df.iloc[i, 9]     # sat2
        ])
        # Third measurement set
        stacked_data.append([
            -df.iloc[i, 10],  # x3 (negated)
            -df.iloc[i, 11],  # y3 (negated)
            df.iloc[i, 13],   # hue3
            df.iloc[i, 14]    # sat3
        ])

    # Create stacked dataframe
    stacked_df = pd.DataFrame(stacked_data, columns=['x', 'y', 'hue', 'sat'])
    print(f"Stacked {len(df)} rows → {len(stacked_df)} data points")

    # Filter to domain (use -0.5 to 0.5 to match simulation domain)
    stacked_df = stacked_df[(stacked_df['x'] > -0.5) & (stacked_df['x'] < 0.5) &
                            (stacked_df['y'] > -0.5) & (stacked_df['y'] < 0.5)].copy()
    print(f"After filtering to [-0.5, 0.5]: {len(stacked_df)} points")

    # Grid-based uniform sampling for cleaner visualization (like visualize_simple.py)
    if len(stacked_df) > max_points:
        print(f"Applying grid-based uniform sampling...")
        selected_indices = grid_based_uniform_sample(stacked_df, grid_size=25, max_per_cell=1)
        stacked_df = stacked_df.iloc[selected_indices].reset_index(drop=True)
        print(f"Selected {len(stacked_df)} uniformly distributed points")

    # Convert hue/sat to u,v after sampling
    angles = stacked_df['hue'].values * 2 * np.pi
    magnitudes = stacked_df['sat'].values
    stacked_df['u'] = magnitudes * np.cos(angles)
    stacked_df['v'] = magnitudes * np.sin(angles)

    return stacked_df[['x', 'y', 'u', 'v']]


def compute_error_metrics(u_pred, v_pred, u_true, v_true):
    """
    Compute various error metrics.

    Returns:
        dict: Error metrics including magnitude, direction, and component errors
    """
    # Magnitude errors
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    mag_true = np.sqrt(u_true**2 + v_true**2)
    mag_error = np.abs(mag_pred - mag_true)
    mag_error_relative = mag_error / (mag_true + 1e-10)

    # Direction errors (angle difference)
    angle_pred = np.arctan2(v_pred, u_pred)
    angle_true = np.arctan2(v_true, u_true)
    angle_error = np.abs(angle_pred - angle_true)
    # Wrap to [0, pi]
    angle_error = np.minimum(angle_error, 2*np.pi - angle_error)
    angle_error_degrees = np.rad2deg(angle_error)

    # Component errors
    u_error = np.abs(u_pred - u_true)
    v_error = np.abs(v_pred - v_true)

    # Vector magnitude error
    vector_error = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)

    metrics = {
        'magnitude_error': {
            'mean': np.mean(mag_error),
            'std': np.std(mag_error),
            'max': np.max(mag_error),
            'median': np.median(mag_error),
        },
        'magnitude_error_relative': {
            'mean': np.mean(mag_error_relative),
            'std': np.std(mag_error_relative),
            'max': np.max(mag_error_relative),
            'median': np.median(mag_error_relative),
        },
        'direction_error_degrees': {
            'mean': np.mean(angle_error_degrees),
            'std': np.std(angle_error_degrees),
            'max': np.max(angle_error_degrees),
            'median': np.median(angle_error_degrees),
        },
        'component_u_error': {
            'mean': np.mean(u_error),
            'std': np.std(u_error),
            'max': np.max(u_error),
        },
        'component_v_error': {
            'mean': np.mean(v_error),
            'std': np.std(v_error),
            'max': np.max(v_error),
        },
        'vector_error': {
            'mean': np.mean(vector_error),
            'std': np.std(vector_error),
            'max': np.max(vector_error),
            'median': np.median(vector_error),
        },
        'raw_arrays': {
            'mag_error': mag_error,
            'angle_error_degrees': angle_error_degrees,
            'vector_error': vector_error,
        }
    }

    return metrics


def plot_field_comparison(X, Y, u_analytical, v_analytical, u_approx, v_approx,
                          title, filename, scale_factor):
    """
    Create field comparison plot - top row only with colored vectors.
    """
    from matplotlib.colors import hsv_to_rgb

    fig = plt.figure(figsize=(21, 7))

    # Reshape for plotting
    U_analytical = u_analytical.reshape(X.shape)
    V_analytical = v_analytical.reshape(X.shape)
    U_approx = u_approx.reshape(X.shape)
    V_approx = v_approx.reshape(X.shape)

    # Arrow density control - lower number = more arrows
    # skip=3 gives ~17x17=289 arrows, skip=5 gives ~10x10=100 arrows
    ARROW_SKIP = 3  # Control arrow density for all panels

    # Compute hue (direction) for coloring vectors
    def compute_hue(U, V):
        """Convert vector field to hue values for HSV colormap"""
        angles = np.arctan2(V, U)  # Angle in radians [-π, π]
        hue = (angles + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        return hue

    hue_analytical = compute_hue(U_analytical, V_analytical)
    hue_approx = compute_hue(U_approx, V_approx)

    # Plot 1: Analytical Field with colored vectors
    ax1 = plt.subplot(1, 3, 1)
    Q1 = ax1.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
                     U_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     V_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     hue_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax1.set_title('Analytical Field (Ground Truth)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Plot 2: Approximation Field with colored vectors
    ax2 = plt.subplot(1, 3, 2)
    Q2 = ax2.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
                     U_approx[::ARROW_SKIP, ::ARROW_SKIP],
                     V_approx[::ARROW_SKIP, ::ARROW_SKIP],
                     hue_approx[::ARROW_SKIP, ::ARROW_SKIP],
                     cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax2.set_title(f'{title} Field', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Plot 3: Error Vectors with colored magnitude
    ax3 = plt.subplot(1, 3, 3)
    U_error = U_analytical - U_approx
    V_error = V_analytical - V_approx
    vector_error = np.sqrt(U_error**2 + V_error**2)

    error_scale_factor = 3.0  # Scale error vectors for visibility
    Q3 = ax3.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
                     U_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale_factor,
                     V_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale_factor,
                     vector_error[::ARROW_SKIP, ::ARROW_SKIP],
                     cmap='Reds', alpha=0.8, scale=5.0, width=0.004)
    ax3.set_title('Error Vectors (Analytical - Approx)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x (m)', fontsize=12)
    ax3.set_ylabel('y (m)', fontsize=12)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Add colorbar for error magnitude
    plt.colorbar(Q3, ax=ax3, label='Error Magnitude (m/s)')

    # Add text annotation about scaling
    ax3.text(0.02, 0.98, f'Error vectors scaled {error_scale_factor:.0f}× for visibility',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add scale information
    physical_size = 0.5 * 2 * scale_factor  # Total physical size
    fig.suptitle(f'{title} vs Analytical Field Comparison\n' +
                 f'Simulation Domain: {SIMULATION_DOMAIN:.1f}m | ' +
                 f'Physical Domain: {physical_size:.3f}m ({physical_size*39.37:.1f} inches) | ' +
                 f'Scale Factor: {scale_factor:.4f}',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_raw_vs_analytical(raw_df, X, Y, u_analytical, v_analytical, u_rbf, v_rbf, filename, scale_factor):
    """
    Create 1x3 comparison plot: Analytical | Raw Data | RBF Error
    """
    from matplotlib.colors import hsv_to_rgb

    fig = plt.figure(figsize=(21, 7))

    # Reshape analytical for plotting
    U_analytical = u_analytical.reshape(X.shape)
    V_analytical = v_analytical.reshape(X.shape)

    # Arrow density control
    ARROW_SKIP = 3  # Match other plots

    # Compute hue for coloring
    def compute_hue(U, V):
        """Convert vector field to hue values for HSV colormap"""
        angles = np.arctan2(V, U)
        hue = (angles + np.pi) / (2 * np.pi)
        return hue

    hue_analytical = compute_hue(U_analytical, V_analytical)

    # Plot 1: Analytical Field (matching rbf_vs_analytical style - just colored quiver)
    ax1 = plt.subplot(1, 3, 1)
    Q1 = ax1.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
                     U_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     V_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     hue_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax1.set_title('Analytical', fontsize=20, fontweight='bold')
    ax1.set_xlabel('x (m)', fontsize=18)
    ax1.set_ylabel('y (m)', fontsize=18)
    ax1.tick_params(labelsize=16)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Plot 2: Raw Training Data (matching visualize_simple.py exactly)
    ax2 = plt.subplot(1, 3, 2)

    # Compute hue for raw data
    angles_raw = np.arctan2(raw_df['v'].values, raw_df['u'].values)
    hue_raw = (angles_raw + np.pi) / (2 * np.pi)

    Q2 = ax2.quiver(raw_df['x'].values, raw_df['y'].values,
                     raw_df['u'].values, raw_df['v'].values,
                     hue_raw, cmap='hsv', alpha=0.7, scale=8, width=0.003)
    ax2.set_title('Reconstructed', fontsize=20, fontweight='bold')
    ax2.set_xlabel('x (m)', fontsize=18)
    ax2.set_ylabel('y (m)', fontsize=18)
    ax2.tick_params(labelsize=16)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot 3: RBF Error visualization (matching rbf_vs_analytical third panel exactly)
    ax3 = plt.subplot(1, 3, 3)

    # Reshape RBF for plotting
    U_rbf = u_rbf.reshape(X.shape)
    V_rbf = v_rbf.reshape(X.shape)

    # Compute errors (same as plot_field_comparison)
    U_error = U_analytical - U_rbf
    V_error = V_analytical - V_rbf
    vector_error = np.sqrt(U_error**2 + V_error**2)

    error_scale_factor = 3.0  # Scale error vectors for visibility
    Q3 = ax3.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
                     U_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale_factor,
                     V_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale_factor,
                     vector_error[::ARROW_SKIP, ::ARROW_SKIP],
                     cmap='Reds', alpha=0.8, scale=5.0, width=0.004)
    ax3.set_title('Reconstruction Error', fontsize=20, fontweight='bold')
    ax3.set_xlabel('x (m)', fontsize=18)
    ax3.set_ylabel('y (m)', fontsize=18)
    ax3.tick_params(labelsize=16)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_raw_vs_analytical_colored(raw_df, X, Y, u_analytical, v_analytical, u_rbf, v_rbf, filename, scale_factor):
    """
    Create 1x3 comparison plot with colored error arrows: Analytical | Raw Data | RBF Error (colored by direction)
    """
    from matplotlib.colors import hsv_to_rgb

    fig = plt.figure(figsize=(21, 7))

    # Reshape analytical for plotting
    U_analytical = u_analytical.reshape(X.shape)
    V_analytical = v_analytical.reshape(X.shape)

    # Arrow density control
    ARROW_SKIP = 3  # Match other plots

    # Compute hue for coloring
    def compute_hue(U, V):
        """Convert vector field to hue values for HSV colormap"""
        angles = np.arctan2(V, U)
        hue = (angles + np.pi) / (2 * np.pi)
        return hue

    hue_analytical = compute_hue(U_analytical, V_analytical)

    # Plot 1: Analytical Field (matching rbf_vs_analytical style - just colored quiver)
    ax1 = plt.subplot(1, 3, 1)
    Q1 = ax1.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
                     U_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     V_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     hue_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax1.set_title('Analytical', fontsize=20, fontweight='bold')
    ax1.set_xlabel('x (m)', fontsize=18)
    ax1.set_ylabel('y (m)', fontsize=18)
    ax1.tick_params(labelsize=16)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Plot 2: Raw Training Data (matching visualize_simple.py exactly)
    ax2 = plt.subplot(1, 3, 2)

    # Compute hue for raw data
    angles_raw = np.arctan2(raw_df['v'].values, raw_df['u'].values)
    hue_raw = (angles_raw + np.pi) / (2 * np.pi)

    Q2 = ax2.quiver(raw_df['x'].values, raw_df['y'].values,
                     raw_df['u'].values, raw_df['v'].values,
                     hue_raw, cmap='hsv', alpha=0.7, scale=8, width=0.003)
    ax2.set_title('Reconstructed', fontsize=20, fontweight='bold')
    ax2.set_xlabel('x (m)', fontsize=18)
    ax2.set_ylabel('y (m)', fontsize=18)
    ax2.tick_params(labelsize=16)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot 3: RBF Error visualization with COLORED error vectors (by direction)
    ax3 = plt.subplot(1, 3, 3)

    # Reshape RBF for plotting
    U_rbf = u_rbf.reshape(X.shape)
    V_rbf = v_rbf.reshape(X.shape)

    # Compute errors (same as plot_field_comparison)
    U_error = U_analytical - U_rbf
    V_error = V_analytical - V_rbf
    vector_error = np.sqrt(U_error**2 + V_error**2)

    # Compute hue for error vectors (color by ERROR direction, not magnitude)
    hue_error = compute_hue(U_error, V_error)

    error_scale_factor = 3.0  # Scale error vectors for visibility
    Q3 = ax3.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
                     U_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale_factor,
                     V_error[::ARROW_SKIP, ::ARROW_SKIP] * error_scale_factor,
                     hue_error[::ARROW_SKIP, ::ARROW_SKIP],
                     cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax3.set_title('Reconstruction Error', fontsize=20, fontweight='bold')
    ax3.set_xlabel('x (m)', fontsize=18)
    ax3.set_ylabel('y (m)', fontsize=18)
    ax3.tick_params(labelsize=16)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_error_statistics(metrics_nn, metrics_rbf):
    """
    Create statistical comparison plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Magnitude Error Comparison
    ax = axes[0, 0]
    categories = ['Mean', 'Median', 'Std', 'Max']
    nn_mag = [metrics_nn['magnitude_error'][k.lower()] for k in categories]
    rbf_mag = [metrics_rbf['magnitude_error'][k.lower()] for k in categories]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, nn_mag, width, label='NN', alpha=0.8)
    ax.bar(x + width/2, rbf_mag, width, label='RBF', alpha=0.8)
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Magnitude Error Statistics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative Magnitude Error
    ax = axes[0, 1]
    nn_mag_rel = [metrics_nn['magnitude_error_relative'][k.lower()] * 100 for k in categories]
    rbf_mag_rel = [metrics_rbf['magnitude_error_relative'][k.lower()] * 100 for k in categories]
    ax.bar(x - width/2, nn_mag_rel, width, label='NN', alpha=0.8)
    ax.bar(x + width/2, rbf_mag_rel, width, label='RBF', alpha=0.8)
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Relative Magnitude Error', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Direction Error Comparison
    ax = axes[0, 2]
    nn_dir = [metrics_nn['direction_error_degrees'][k.lower()] for k in categories]
    rbf_dir = [metrics_rbf['direction_error_degrees'][k.lower()] for k in categories]
    ax.bar(x - width/2, nn_dir, width, label='NN', alpha=0.8)
    ax.bar(x + width/2, rbf_dir, width, label='RBF', alpha=0.8)
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Direction Error Statistics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vector Error Comparison
    ax = axes[1, 0]
    nn_vec = [metrics_nn['vector_error'][k.lower()] for k in ['mean', 'median', 'std', 'max']]
    rbf_vec = [metrics_rbf['vector_error'][k.lower()] for k in ['mean', 'median', 'std', 'max']]
    ax.bar(x - width/2, nn_vec, width, label='NN', alpha=0.8)
    ax.bar(x + width/2, rbf_vec, width, label='RBF', alpha=0.8)
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Total Vector Error Statistics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Component U Error
    ax = axes[1, 1]
    categories_comp = ['Mean', 'Std', 'Max']
    nn_u = [metrics_nn['component_u_error'][k.lower()] for k in categories_comp]
    rbf_u = [metrics_rbf['component_u_error'][k.lower()] for k in categories_comp]
    x_comp = np.arange(len(categories_comp))
    ax.bar(x_comp - width/2, nn_u, width, label='NN', alpha=0.8)
    ax.bar(x_comp + width/2, rbf_u, width, label='RBF', alpha=0.8)
    ax.set_ylabel('Error (m/s)')
    ax.set_title('U Component Error', fontweight='bold')
    ax.set_xticks(x_comp)
    ax.set_xticklabels(categories_comp)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Component V Error
    ax = axes[1, 2]
    nn_v = [metrics_nn['component_v_error'][k.lower()] for k in categories_comp]
    rbf_v = [metrics_rbf['component_v_error'][k.lower()] for k in categories_comp]
    ax.bar(x_comp - width/2, nn_v, width, label='NN', alpha=0.8)
    ax.bar(x_comp + width/2, rbf_v, width, label='RBF', alpha=0.8)
    ax.set_ylabel('Error (m/s)')
    ax.set_title('V Component Error', fontweight='bold')
    ax.set_xticks(x_comp)
    ax.set_xticklabels(categories_comp)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Statistical Comparison: NN vs RBF Approximation Errors',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_statistics.png'), dpi=300, bbox_inches='tight')
    print("Saved: error_statistics.png")
    plt.close()


def plot_error_histograms(metrics_nn, metrics_rbf):
    """
    Create error distribution histograms.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Magnitude Error Distribution
    ax = axes[0]
    ax.hist(metrics_nn['raw_arrays']['mag_error'], bins=50, alpha=0.6, label='NN', density=True)
    ax.hist(metrics_rbf['raw_arrays']['mag_error'], bins=50, alpha=0.6, label='RBF', density=True)
    ax.set_xlabel('Magnitude Error (m/s)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Magnitude Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Direction Error Distribution
    ax = axes[1]
    ax.hist(metrics_nn['raw_arrays']['angle_error_degrees'], bins=50, alpha=0.6, label='NN', density=True)
    ax.hist(metrics_rbf['raw_arrays']['angle_error_degrees'], bins=50, alpha=0.6, label='RBF', density=True)
    ax.set_xlabel('Direction Error (degrees)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Direction Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vector Error Distribution
    ax = axes[2]
    ax.hist(metrics_nn['raw_arrays']['vector_error'], bins=50, alpha=0.6, label='NN', density=True)
    ax.hist(metrics_rbf['raw_arrays']['vector_error'], bins=50, alpha=0.6, label='RBF', density=True)
    ax.set_xlabel('Total Vector Error (m/s)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Total Vector Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Error Distribution Histograms',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_histograms.png'), dpi=300, bbox_inches='tight')
    print("Saved: error_histograms.png")
    plt.close()


def generate_text_report(metrics_nn, metrics_rbf, scale_factor):
    """
    Generate comprehensive text report.
    """
    report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FIELD IMPERFECTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("SCALE INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Simulation Domain:      {SIMULATION_DOMAIN:.3f} m ([-0.5, 0.5] × [-0.5, 0.5])\n")
        f.write(f"Physical Printout Size: {PHYSICAL_DOMAIN:.3f} m ({PHYSICAL_DOMAIN * 39.37:.1f} inches)\n")
        f.write(f"Scale Factor:           {scale_factor:.4f}\n")
        f.write(f"Grid Resolution:        {GRID_RESOLUTION} × {GRID_RESOLUTION} ({GRID_RESOLUTION**2} points)\n\n")

        f.write("NEURAL NETWORK (NN) APPROXIMATION ERRORS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Vector Magnitude Error:\n")
        f.write(f"  Mean:   {metrics_nn['magnitude_error']['mean']:.6f} m/s\n")
        f.write(f"  Median: {metrics_nn['magnitude_error']['median']:.6f} m/s\n")
        f.write(f"  Std:    {metrics_nn['magnitude_error']['std']:.6f} m/s\n")
        f.write(f"  Max:    {metrics_nn['magnitude_error']['max']:.6f} m/s\n\n")

        f.write(f"Relative Magnitude Error:\n")
        f.write(f"  Mean:   {metrics_nn['magnitude_error_relative']['mean']*100:.2f}%\n")
        f.write(f"  Median: {metrics_nn['magnitude_error_relative']['median']*100:.2f}%\n")
        f.write(f"  Max:    {metrics_nn['magnitude_error_relative']['max']*100:.2f}%\n\n")

        f.write(f"Direction Error:\n")
        f.write(f"  Mean:   {metrics_nn['direction_error_degrees']['mean']:.2f} degrees\n")
        f.write(f"  Median: {metrics_nn['direction_error_degrees']['median']:.2f} degrees\n")
        f.write(f"  Std:    {metrics_nn['direction_error_degrees']['std']:.2f} degrees\n")
        f.write(f"  Max:    {metrics_nn['direction_error_degrees']['max']:.2f} degrees\n\n")

        f.write(f"Total Vector Error:\n")
        f.write(f"  Mean:   {metrics_nn['vector_error']['mean']:.6f} m/s\n")
        f.write(f"  Median: {metrics_nn['vector_error']['median']:.6f} m/s\n")
        f.write(f"  Std:    {metrics_nn['vector_error']['std']:.6f} m/s\n")
        f.write(f"  Max:    {metrics_nn['vector_error']['max']:.6f} m/s\n\n")

        f.write("RBF INTERPOLATOR APPROXIMATION ERRORS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Vector Magnitude Error:\n")
        f.write(f"  Mean:   {metrics_rbf['magnitude_error']['mean']:.6f} m/s\n")
        f.write(f"  Median: {metrics_rbf['magnitude_error']['median']:.6f} m/s\n")
        f.write(f"  Std:    {metrics_rbf['magnitude_error']['std']:.6f} m/s\n")
        f.write(f"  Max:    {metrics_rbf['magnitude_error']['max']:.6f} m/s\n\n")

        f.write(f"Relative Magnitude Error:\n")
        f.write(f"  Mean:   {metrics_rbf['magnitude_error_relative']['mean']*100:.2f}%\n")
        f.write(f"  Median: {metrics_rbf['magnitude_error_relative']['median']*100:.2f}%\n")
        f.write(f"  Max:    {metrics_rbf['magnitude_error_relative']['max']*100:.2f}%\n\n")

        f.write(f"Direction Error:\n")
        f.write(f"  Mean:   {metrics_rbf['direction_error_degrees']['mean']:.2f} degrees\n")
        f.write(f"  Median: {metrics_rbf['direction_error_degrees']['median']:.2f} degrees\n")
        f.write(f"  Std:    {metrics_rbf['direction_error_degrees']['std']:.2f} degrees\n")
        f.write(f"  Max:    {metrics_rbf['direction_error_degrees']['max']:.2f} degrees\n\n")

        f.write(f"Total Vector Error:\n")
        f.write(f"  Mean:   {metrics_rbf['vector_error']['mean']:.6f} m/s\n")
        f.write(f"  Median: {metrics_rbf['vector_error']['median']:.6f} m/s\n")
        f.write(f"  Std:    {metrics_rbf['vector_error']['std']:.6f} m/s\n")
        f.write(f"  Max:    {metrics_rbf['vector_error']['max']:.6f} m/s\n\n")

        f.write("COMPARATIVE ANALYSIS\n")
        f.write("-" * 80 + "\n")

        # Determine winner for each metric
        def compare_metric(nn_val, rbf_val, metric_name, lower_is_better=True):
            if lower_is_better:
                winner = "RBF" if rbf_val < nn_val else "NN"
                diff = abs(nn_val - rbf_val)
                pct_diff = (diff / min(nn_val, rbf_val)) * 100 if min(nn_val, rbf_val) > 0 else 0
            else:
                winner = "NN" if nn_val > rbf_val else "RBF"
                diff = abs(nn_val - rbf_val)
                pct_diff = (diff / max(nn_val, rbf_val)) * 100 if max(nn_val, rbf_val) > 0 else 0

            return winner, diff, pct_diff

        f.write(f"Mean Magnitude Error:\n")
        winner, diff, pct = compare_metric(metrics_nn['magnitude_error']['mean'],
                                           metrics_rbf['magnitude_error']['mean'],
                                           "mean magnitude")
        f.write(f"  Winner: {winner} (by {diff:.6f} m/s, {pct:.1f}% better)\n\n")

        f.write(f"Mean Direction Error:\n")
        winner, diff, pct = compare_metric(metrics_nn['direction_error_degrees']['mean'],
                                           metrics_rbf['direction_error_degrees']['mean'],
                                           "mean direction")
        f.write(f"  Winner: {winner} (by {diff:.2f} degrees, {pct:.1f}% better)\n\n")

        f.write(f"Mean Total Vector Error:\n")
        winner, diff, pct = compare_metric(metrics_nn['vector_error']['mean'],
                                           metrics_rbf['vector_error']['mean'],
                                           "mean vector")
        f.write(f"  Winner: {winner} (by {diff:.6f} m/s, {pct:.1f}% better)\n\n")

        f.write("OBSERVATIONS\n")
        f.write("-" * 80 + "\n")

        # Generate observations
        if metrics_rbf['magnitude_error']['mean'] < metrics_nn['magnitude_error']['mean']:
            f.write("• RBF interpolator shows lower magnitude errors than NN on average.\n")
        else:
            f.write("• Neural network shows lower magnitude errors than RBF on average.\n")

        if metrics_rbf['direction_error_degrees']['mean'] < metrics_nn['direction_error_degrees']['mean']:
            f.write("• RBF interpolator provides better direction accuracy.\n")
        else:
            f.write("• Neural network provides better direction accuracy.\n")

        # Check max errors
        if metrics_nn['vector_error']['max'] > 0.1:
            f.write(f"• NN shows significant maximum error ({metrics_nn['vector_error']['max']:.3f} m/s) in some regions.\n")
        if metrics_rbf['vector_error']['max'] > 0.1:
            f.write(f"• RBF shows significant maximum error ({metrics_rbf['vector_error']['max']:.3f} m/s) in some regions.\n")

        # Overall recommendation
        f.write("\nRECOMMENDATION\n")
        f.write("-" * 80 + "\n")
        nn_score = sum([
            metrics_nn['magnitude_error']['mean'] < metrics_rbf['magnitude_error']['mean'],
            metrics_nn['direction_error_degrees']['mean'] < metrics_rbf['direction_error_degrees']['mean'],
            metrics_nn['vector_error']['mean'] < metrics_rbf['vector_error']['mean']
        ])

        if nn_score > 1.5:
            f.write("Based on mean error metrics, NN approximation performs better overall.\n")
        elif nn_score < 1.5:
            f.write("Based on mean error metrics, RBF interpolation performs better overall.\n")
        else:
            f.write("NN and RBF approximations show comparable performance.\n")

        f.write("\nNote: This analysis uses vortex field (vortex1) as reference.\n")
        f.write("Results may vary for different field types (sink, saddle, etc.).\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Saved: analysis_report.txt")
    return report_path


def main():
    print("=" * 80)
    print("FIELD IMPERFECTION ANALYSIS")
    print("=" * 80)
    print(f"\nSimulation Domain: {SIMULATION_DOMAIN:.3f} m")
    print(f"Physical Domain:   {PHYSICAL_DOMAIN:.3f} m ({PHYSICAL_DOMAIN * 39.37:.1f} inches)")
    print(f"Scale Factor:      {SCALE_FACTOR:.4f}")
    print(f"Grid Resolution:   {GRID_RESOLUTION} × {GRID_RESOLUTION}\n")

    # Create fields
    print("Loading fields...")
    field_analytical = AnalyticalField(vortex1)
    field_nn = NNField(predictor_dir=PREDICTOR_DIR)
    field_rbf = RBFField(predictor_dir=PREDICTOR_DIR)
    print("✓ Fields loaded\n")

    # Create evaluation grid
    print("Creating evaluation grid...")
    X, Y, x_flat, y_flat = create_evaluation_grid(GRID_RESOLUTION)
    print("✓ Grid created\n")

    # Evaluate fields
    print("Evaluating analytical field...")
    u_analytical, v_analytical = evaluate_field(field_analytical, x_flat, y_flat)
    print("✓ Analytical field evaluated")

    print("Evaluating NN field...")
    u_nn, v_nn = evaluate_field(field_nn, x_flat, y_flat)
    print("✓ NN field evaluated")

    print("Evaluating RBF field...")
    u_rbf, v_rbf = evaluate_field(field_rbf, x_flat, y_flat)
    print("✓ RBF field evaluated\n")

    # Compute error metrics
    print("Computing error metrics...")
    metrics_nn = compute_error_metrics(u_nn, v_nn, u_analytical, v_analytical)
    metrics_rbf = compute_error_metrics(u_rbf, v_rbf, u_analytical, v_analytical)
    print("✓ Metrics computed\n")

    # Load raw training data
    print("Loading raw training data...")
    raw_df = load_raw_training_data('all_vortex_data_1.csv', max_points=200)

    # Generate plots
    print("\nGenerating plots...")
    plot_field_comparison(X, Y, u_analytical, v_analytical, u_nn, v_nn,
                         'NN', 'nn_vs_analytical.png', SCALE_FACTOR)
    plot_field_comparison(X, Y, u_analytical, v_analytical, u_rbf, v_rbf,
                         'RBF', 'rbf_vs_analytical.png', SCALE_FACTOR)

    # Generate raw vs analytical plots if data is available
    if raw_df is not None:
        plot_raw_vs_analytical(raw_df, X, Y, u_analytical, v_analytical, u_rbf, v_rbf,
                              'raw_vs_analytical.png', SCALE_FACTOR)
        plot_raw_vs_analytical_colored(raw_df, X, Y, u_analytical, v_analytical, u_rbf, v_rbf,
                              'raw_vs_analytical2.png', SCALE_FACTOR)

    plot_error_statistics(metrics_nn, metrics_rbf)
    plot_error_histograms(metrics_nn, metrics_rbf)
    print("✓ Plots generated\n")

    # Generate text report
    print("Generating text report...")
    report_path = generate_text_report(metrics_nn, metrics_rbf, SCALE_FACTOR)
    print("✓ Report generated\n")

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nNN Mean Vector Error:  {metrics_nn['vector_error']['mean']:.6f} m/s")
    print(f"RBF Mean Vector Error: {metrics_rbf['vector_error']['mean']:.6f} m/s")
    print(f"\nNN Mean Direction Error:  {metrics_nn['direction_error_degrees']['mean']:.2f}°")
    print(f"RBF Mean Direction Error: {metrics_rbf['direction_error_degrees']['mean']:.2f}°")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"\nRead full report: {report_path}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
