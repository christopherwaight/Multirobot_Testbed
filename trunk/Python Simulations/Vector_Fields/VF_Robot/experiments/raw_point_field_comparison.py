#!/usr/bin/env python3
"""
Raw Measurement Point Field Comparison

Compares analytical field predictions vs actual robot measurements at the exact
measurement locations. Shows true measurement error at sampling points.

Creates a 2x3 figure:
- Row 1: Analytical field evaluated at measurement points
- Row 2: Raw measurements from robots
- Row 3: Measurement error (Analytical - Raw)

Output: measurement_error_comparison.png in experiments folder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fields.field_types import AnalyticalField
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Saddle import saddle1

# Configuration
MAX_RAW_POINTS = 200  # Maximum number of measurement points to display

# Predictor directories (for finding raw data CSVs)
VORTEX_PREDICTOR_DIR = 'vortex_predictors'
SADDLE_PREDICTOR_DIR = 'saddle_predictors'

# Raw data files
VORTEX_DATA_FILE = 'all_vortex_data_1.csv'
SADDLE_DATA_FILE = 'saddle_data.csv'

# Output
OUTPUT_FILE = 'measurement_error_comparison.png'


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


def load_vortex_raw_data(predictor_dir, filename, max_points=100):
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


def load_saddle_raw_data(predictor_dir, filename, max_points=100):
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


def evaluate_field_at_points(field, df):
    """Evaluate analytical field at specific measurement points."""
    u_vals, v_vals = [], []
    for _, row in df.iterrows():
        u, v = field.get_value(row['x'], row['y'])
        u_vals.append(u)
        v_vals.append(v)

    df_result = df.copy()
    df_result['u'] = u_vals
    df_result['v'] = v_vals
    return df_result[['x', 'y', 'u', 'v']]


def compute_hue(u, v):
    """Convert vector components to hue values for HSV colormap."""
    angles = np.arctan2(v, u)
    hue = (angles + np.pi) / (2 * np.pi)
    return hue


def plot_comparison(vortex_analytical, vortex_raw,
                   saddle_analytical, saddle_raw,
                   filename):
    """Create 2x3 comparison plot at raw measurement points."""

    fig = plt.figure(figsize=(14, 21))

    # Row 1: Analytical fields at measurement points
    # Vortex Analytical
    ax1 = plt.subplot(3, 2, 1)
    hue_v_anal = compute_hue(vortex_analytical['u'].values, vortex_analytical['v'].values)
    ax1.quiver(vortex_analytical['x'].values, vortex_analytical['y'].values,
               vortex_analytical['u'].values, vortex_analytical['v'].values,
               hue_v_anal, cmap='hsv', alpha=0.8, scale=8, width=0.003)
    ax1.set_title('Analytical', fontsize=20, fontweight='bold')
    ax1.set_xlabel('x (m)', fontsize=18)
    ax1.set_ylabel('y (m)', fontsize=18)
    ax1.tick_params(labelsize=16)
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Saddle Analytical
    ax2 = plt.subplot(3, 2, 2)
    hue_s_anal = compute_hue(saddle_analytical['u'].values, saddle_analytical['v'].values)
    ax2.quiver(saddle_analytical['x'].values, saddle_analytical['y'].values,
               saddle_analytical['u'].values, saddle_analytical['v'].values,
               hue_s_anal, cmap='hsv', alpha=0.8, scale=8, width=0.003)
    ax2.set_title('Analytical', fontsize=20, fontweight='bold')
    ax2.set_xlabel('x (m)', fontsize=18)
    ax2.set_ylabel('y (m)', fontsize=18)
    ax2.tick_params(labelsize=16)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Row 2: Raw measurements
    # Vortex Raw
    ax3 = plt.subplot(3, 2, 3)
    hue_v_raw = compute_hue(vortex_raw['u'].values, vortex_raw['v'].values)
    ax3.quiver(vortex_raw['x'].values, vortex_raw['y'].values,
               vortex_raw['u'].values, vortex_raw['v'].values,
               hue_v_raw, cmap='hsv', alpha=0.8, scale=8, width=0.003)
    ax3.set_title('Reconstructed', fontsize=20, fontweight='bold')
    ax3.set_xlabel('x (m)', fontsize=18)
    ax3.set_ylabel('y (m)', fontsize=18)
    ax3.tick_params(labelsize=16)
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Saddle Raw
    ax4 = plt.subplot(3, 2, 4)
    hue_s_raw = compute_hue(saddle_raw['u'].values, saddle_raw['v'].values)
    ax4.quiver(saddle_raw['x'].values, saddle_raw['y'].values,
               saddle_raw['u'].values, saddle_raw['v'].values,
               hue_s_raw, cmap='hsv', alpha=0.8, scale=8, width=0.003)
    ax4.set_title('Reconstructed', fontsize=20, fontweight='bold')
    ax4.set_xlabel('x (m)', fontsize=18)
    ax4.set_ylabel('y (m)', fontsize=18)
    ax4.tick_params(labelsize=16)
    ax4.set_xlim(-0.5, 0.5)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # Row 3: Measurement Errors (Raw - Analytical)
    # Vortex Error
    ax5 = plt.subplot(3, 2, 5)
    u_v_error = vortex_raw['u'].values - vortex_analytical['u'].values
    v_v_error = vortex_raw['v'].values - vortex_analytical['v'].values
    hue_v_error = compute_hue(u_v_error, v_v_error)
    error_scale = 1.0  # No scaling - show actual error magnitude
    ax5.quiver(vortex_raw['x'].values, vortex_raw['y'].values,
               u_v_error * error_scale, v_v_error * error_scale,
               hue_v_error, cmap='hsv', alpha=0.8, scale=8, width=0.003)
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
    u_s_error = saddle_raw['u'].values - saddle_analytical['u'].values
    v_s_error = saddle_raw['v'].values - saddle_analytical['v'].values
    hue_s_error = compute_hue(u_s_error, v_s_error)
    ax6.quiver(saddle_raw['x'].values, saddle_raw['y'].values,
               u_s_error * error_scale, v_s_error * error_scale,
               hue_s_error, cmap='hsv', alpha=0.8, scale=8, width=0.003)
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
    print("RAW MEASUREMENT POINT FIELD COMPARISON")
    print("=" * 80)
    print(f"\nComparing analytical predictions vs measurements at sampling points")
    print(f"Maximum points per field: {MAX_RAW_POINTS}\n")

    # Load analytical fields
    print("Loading analytical fields...")
    field_vortex_analytical = AnalyticalField(vortex1)
    field_saddle_analytical = AnalyticalField(saddle1)
    print("✓ Analytical fields loaded")

    # Load raw measurement data
    print("\nLoading raw measurement data...")
    vortex_raw = load_vortex_raw_data(VORTEX_PREDICTOR_DIR, VORTEX_DATA_FILE, MAX_RAW_POINTS)
    saddle_raw = load_saddle_raw_data(SADDLE_PREDICTOR_DIR, SADDLE_DATA_FILE, MAX_RAW_POINTS)

    if vortex_raw is None or saddle_raw is None:
        print("Error: Could not load raw data. Exiting.")
        return

    print("✓ Raw data loaded")

    # Evaluate analytical fields at measurement points
    print("\nEvaluating analytical fields at measurement points...")
    vortex_analytical = evaluate_field_at_points(field_vortex_analytical, vortex_raw)
    saddle_analytical = evaluate_field_at_points(field_saddle_analytical, saddle_raw)
    print("✓ Analytical fields evaluated")

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(vortex_analytical, vortex_raw,
                   saddle_analytical, saddle_raw,
                   OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nAll three rows show data at the same {len(vortex_raw)} (vortex) and")
    print(f"{len(saddle_raw)} (saddle) measurement locations.")
    print("\nThis allows direct visual comparison of:")
    print("  • What the analytical model predicts at each point")
    print("  • What was actually measured by the robots")
    print("  • The measurement error at each location")


if __name__ == '__main__':
    main()
