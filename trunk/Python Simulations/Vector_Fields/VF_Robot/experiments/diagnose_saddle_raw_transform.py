#!/usr/bin/env python3
"""
Diagnostic script to test different coordinate transformations for saddle raw data.

The issue: Raw data collected from physical robots needs coordinate/hue transformations
to match the simulation coordinate system. Different fields (vortex, saddle) may require
different transformations.

This script tests multiple transformation candidates and generates visual comparisons
to help identify the correct transformation.
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
from src.fields.environments.Saddle import saddle1

# Configuration
PREDICTOR_DIR = 'saddle_predictors'
OUTPUT_DIR = 'saddle_raw_data_diagnostics'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_raw_data(filename='saddle_data.csv'):
    """Load raw CSV data without any transformations"""
    predictor_path = os.path.join(os.path.dirname(__file__), '..', PREDICTOR_DIR, filename)

    if not os.path.exists(predictor_path):
        print(f"Error: File not found at {predictor_path}")
        return None

    print(f"Loading raw data from {filename}...")
    df = pd.read_csv(predictor_path)
    print(f"Loaded {len(df)} rows")
    return df


def apply_transformation(df, transform_id):
    """
    Apply one of several transformation candidates to the raw data.

    Transformations to test:
    1. CURRENT_WRONG: The current (incorrect) transformation in saddle analysis
    2. RBF_TRAINER: The transformation used in RBF trainer (x, -y)
    3. VORTEX_SIMPLE: The simple negation used for vortex (-x, -y)
    4. NO_TRANSFORM: No transformation (raw data as-is)
    5. SWAP_NEGATE: Swap and negate (-y, -x)
    6. ROTATE_90_CW: 90° clockwise rotation (y, -x)
    7. ROTATE_90_CCW: 90° counterclockwise rotation (-y, x)
    """
    stacked_data = []

    for i in range(len(df)):
        # Process all three measurement sets
        for col_offset in [0, 5, 10]:  # Starting columns for each measurement set
            x_raw = df.iloc[i, col_offset]
            y_raw = df.iloc[i, col_offset + 1]
            hue_raw = df.iloc[i, col_offset + 3]
            sat_raw = df.iloc[i, col_offset + 4]

            # Apply coordinate transformation based on ID
            if transform_id == "CURRENT_WRONG":
                # 90° CW + negate x: (x,y) → (-y, -x)
                x_new = -y_raw
                y_new = -x_raw
                hue_new = (0.75 - hue_raw) % 1.0

            elif transform_id == "RBF_TRAINER":
                # What the RBF was actually trained on: (x, -y)
                x_new = x_raw
                y_new = -y_raw
                hue_new = hue_raw

            elif transform_id == "VORTEX_SIMPLE":
                # Simple negation like vortex: (-x, -y)
                x_new = -x_raw
                y_new = -y_raw
                hue_new = hue_raw

            elif transform_id == "NO_TRANSFORM":
                # Raw data as-is
                x_new = x_raw
                y_new = y_raw
                hue_new = hue_raw

            elif transform_id == "SWAP_NEGATE":
                # Swap and negate: (-y, -x)
                x_new = -y_raw
                y_new = -x_raw
                hue_new = hue_raw

            elif transform_id == "SWAP_NEGATE_ROTATED":
                # Swap and negate: (-y, -x), then rotate 90° CW
                # Step 1: Swap and negate
                x_temp = -y_raw
                y_temp = -x_raw
                # Step 2: Rotate 90° CW: (x,y) → (y, -x)
                x_new = y_temp
                y_new = -x_temp
                # Hue also rotates 90° CW (-90° in hue space)
                hue_new = (hue_raw - 0.25) % 1.0

            elif transform_id == "ROTATE_90_CW":
                # 90° clockwise: (y, -x)
                x_new = y_raw
                y_new = -x_raw
                hue_new = (hue_raw - 0.25) % 1.0  # Rotate hue by -90°

            elif transform_id == "ROTATE_90_CCW":
                # 90° counterclockwise: (-y, x)
                x_new = -y_raw
                y_new = x_raw
                hue_new = (hue_raw + 0.25) % 1.0  # Rotate hue by +90°

            else:
                raise ValueError(f"Unknown transform_id: {transform_id}")

            stacked_data.append([x_new, y_new, hue_new, sat_raw])

    # Create stacked dataframe
    stacked_df = pd.DataFrame(stacked_data, columns=['x', 'y', 'hue', 'sat'])

    # Filter to domain
    stacked_df = stacked_df[(stacked_df['x'] > -0.5) & (stacked_df['x'] < 0.5) &
                            (stacked_df['y'] > -0.5) & (stacked_df['y'] < 0.5)].copy()

    # Uniform sampling
    if len(stacked_df) > 200:
        from saddle_field_imperfection_analysis import grid_based_uniform_sample
        selected_indices = grid_based_uniform_sample(stacked_df, grid_size=25, max_per_cell=1)
        stacked_df = stacked_df.iloc[selected_indices].reset_index(drop=True)

    # Convert hue/sat to u,v
    angles = stacked_df['hue'].values * 2 * np.pi
    magnitudes = stacked_df['sat'].values
    stacked_df['u'] = magnitudes * np.cos(angles)
    stacked_df['v'] = magnitudes * np.sin(angles)

    return stacked_df[['x', 'y', 'u', 'v']]


def create_evaluation_grid(resolution=50):
    """Create evaluation grid for analytical field"""
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y


def plot_comparison(analytical_field, raw_df, transform_name, filename):
    """Create side-by-side comparison plot"""
    fig = plt.figure(figsize=(14, 7))

    # Create evaluation grid for analytical
    X, Y = create_evaluation_grid(50)

    # Evaluate analytical field
    U_analytical = np.zeros_like(X)
    V_analytical = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U_analytical[i, j], V_analytical[i, j] = analytical_field(X[i, j], Y[i, j])

    ARROW_SKIP = 3

    def compute_hue(U, V):
        angles = np.arctan2(V, U)
        hue = (angles + np.pi) / (2 * np.pi)
        return hue

    hue_analytical = compute_hue(U_analytical, V_analytical)

    # Plot 1: Analytical Field
    ax1 = plt.subplot(1, 2, 1)
    Q1 = ax1.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
                     U_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     V_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     hue_analytical[::ARROW_SKIP, ::ARROW_SKIP],
                     cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax1.set_title('Analytical Field (saddle1)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Plot 2: Raw Data with transformation
    ax2 = plt.subplot(1, 2, 2)

    angles_raw = np.arctan2(raw_df['v'].values, raw_df['u'].values)
    hue_raw = (angles_raw + np.pi) / (2 * np.pi)

    Q2 = ax2.quiver(raw_df['x'].values, raw_df['y'].values,
                     raw_df['u'].values, raw_df['v'].values,
                     hue_raw, cmap='hsv', alpha=0.7, scale=8, width=0.003)
    ax2.set_title(f'Raw Data ({transform_name})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    print("=" * 80)
    print("SADDLE RAW DATA TRANSFORMATION DIAGNOSTIC")
    print("=" * 80)
    print()

    # Load raw data
    df = load_raw_data('saddle_data.csv')
    if df is None:
        return

    # Test transformations
    transformations = [
        ("CURRENT_WRONG", "Current (Wrong): 90° CW + negate x + hue transform"),
        ("RBF_TRAINER", "RBF Trainer: (x, -y) no hue transform"),
        ("VORTEX_SIMPLE", "Vortex Simple: (-x, -y) no hue transform"),
        ("NO_TRANSFORM", "No Transform: (x, y) raw data"),
        ("SWAP_NEGATE", "Swap & Negate: (-y, -x) no hue transform"),
        ("SWAP_NEGATE_ROTATED", "Swap & Negate + 90° CW: (-y, -x) then rotate 90° CW"),
        ("ROTATE_90_CW", "90° CW: (y, -x) with hue rotate -90°"),
        ("ROTATE_90_CCW", "90° CCW: (-y, x) with hue rotate +90°"),
    ]

    # Analytical field
    analytical_field = saddle1

    print("Generating comparison plots for each transformation...\n")

    for transform_id, transform_desc in transformations:
        print(f"Testing: {transform_desc}")

        # Apply transformation
        raw_df = apply_transformation(df, transform_id)
        print(f"  → {len(raw_df)} points after transformation and filtering")

        # Create comparison plot
        filename = f"transform_{transform_id.lower()}.png"
        plot_comparison(analytical_field, raw_df, transform_desc, filename)
        print()

    # Create summary plot showing all transforms in grid
    print("Creating summary comparison grid...")
    create_summary_grid(df, analytical_field, transformations)

    print("=" * 80)
    print("COMPLETED")
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print()
    print("INSTRUCTIONS:")
    print("1. Open the output directory and examine each transform_*.png file")
    print("2. Compare raw data patterns to analytical field")
    print("3. Look for:")
    print("   - Correct vector directions (color matching)")
    print("   - Proper saddle orientation (diagonal pattern)")
    print("   - Smooth field structure")
    print("4. The transform that best matches analytical is the correct one!")
    print("=" * 80)


def create_summary_grid(df, analytical_field, transformations):
    """Create a grid showing all transformations at once"""
    fig = plt.figure(figsize=(21, 12))

    # Create evaluation grid for analytical
    X, Y = create_evaluation_grid(50)
    U_analytical = np.zeros_like(X)
    V_analytical = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U_analytical[i, j], V_analytical[i, j] = analytical_field(X[i, j], Y[i, j])

    ARROW_SKIP = 5  # Fewer arrows for grid view

    def compute_hue(U, V):
        angles = np.arctan2(V, U)
        hue = (angles + np.pi) / (2 * np.pi)
        return hue

    hue_analytical = compute_hue(U_analytical, V_analytical)

    # Plot analytical in first position
    ax = plt.subplot(3, 3, 1)
    ax.quiver(X[::ARROW_SKIP, ::ARROW_SKIP], Y[::ARROW_SKIP, ::ARROW_SKIP],
              U_analytical[::ARROW_SKIP, ::ARROW_SKIP],
              V_analytical[::ARROW_SKIP, ::ARROW_SKIP],
              hue_analytical[::ARROW_SKIP, ::ARROW_SKIP],
              cmap='hsv', alpha=0.8, scale=5.0, width=0.004)
    ax.set_title('ANALYTICAL\n(saddle1)', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)

    # Plot each transformation
    for idx, (transform_id, transform_desc) in enumerate(transformations, start=2):
        raw_df = apply_transformation(df, transform_id)

        ax = plt.subplot(3, 3, idx)

        angles_raw = np.arctan2(raw_df['v'].values, raw_df['u'].values)
        hue_raw = (angles_raw + np.pi) / (2 * np.pi)

        ax.quiver(raw_df['x'].values, raw_df['y'].values,
                  raw_df['u'].values, raw_df['v'].values,
                  hue_raw, cmap='hsv', alpha=0.7, scale=10, width=0.003)

        # Short title
        short_name = transform_id.replace("_", " ")
        ax.set_title(short_name, fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

    fig.suptitle('Saddle Raw Data: All Transformation Candidates',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_all_transforms.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: summary_all_transforms.png")
    plt.close()


if __name__ == '__main__':
    main()
