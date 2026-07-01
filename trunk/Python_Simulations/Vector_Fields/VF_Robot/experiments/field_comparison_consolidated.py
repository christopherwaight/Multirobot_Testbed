"""
Field Comparison Consolidated Visualization

Creates a comprehensive comparison figure showing NN and RBF approximations
vs analytical ground truth for both vortex and saddle fields.

Two layout options:
- 4x2 grid (default): NN/RBF fields with their differences from analytical
  Left column: NN vortex, RBF vortex, NN saddle, RBF saddle
  Right column: Scaled differences from analytical (×3 for visibility)

- 6x2 grid: Including analytical fields (with null difference plots)
  Left column: All fields including analytical
  Right column: Differences (analytical-analytical shows as null)

Usage:
  python field_comparison_consolidated.py

Output:
  field_comparison_4x2.png or field_comparison_6x2.png (based on LAYOUT_MODE)

Configuration:
  Change LAYOUT_MODE variable to "4x2" or "6x2" as desired
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fields.field_types import AnalyticalField, NNField, RBFField
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Saddle import saddle1


# Configuration
GRID_RESOLUTION = 75  # Grid resolution for evaluation
DIFFERENCE_SCALE = 3.0  # Scale factor for difference plots
FIGURE_DPI = 300  # Output resolution
ARROW_SCALE = 15  # Scale for quiver plots (higher = smaller arrows)
LAYOUT_MODE = "4x2"  # Options: "4x2" (recommended) or "6x2"


def create_evaluation_grid(resolution=75, domain_size=0.6):
    """
    Create evaluation grid for field visualization.

    Args:
        resolution: Number of points in each dimension
        domain_size: Half-width of square domain

    Returns:
        X, Y: Meshgrid arrays
        x_flat, y_flat: Flattened coordinate arrays
    """
    x = np.linspace(-domain_size, domain_size, resolution)
    y = np.linspace(-domain_size, domain_size, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y, X.flatten(), Y.flatten()


def evaluate_field_batch(field, x_flat, y_flat):
    """
    Evaluate field at multiple points.

    Args:
        field: Field object with get_value method
        x_flat, y_flat: Flattened coordinate arrays

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


def compute_field_difference(u1, v1, u2, v2, scale=1.0):
    """
    Compute scaled difference between two vector fields.

    Args:
        u1, v1: Components of first field
        u2, v2: Components of second field
        scale: Scale factor for difference

    Returns:
        du, dv: Scaled difference components
    """
    du = scale * (u1 - u2)
    dv = scale * (v1 - v2)
    return du, dv


def downsample_for_quiver(X, Y, U, V, skip=3):
    """
    Downsample grid for cleaner quiver plot visualization.

    Args:
        X, Y: Coordinate grids
        U, V: Vector field components
        skip: Downsampling factor

    Returns:
        Downsampled X, Y, U, V
    """
    return X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip]


def create_4x2_visualization(vortex_data, saddle_data):
    """
    Create 4x2 grid visualization (NN/RBF fields with differences).

    Args:
        vortex_data: Dict with vortex field evaluations
        saddle_data: Dict with saddle field evaluations
    """
    fig = plt.figure(figsize=(14, 16))

    # Color normalization for magnitude
    all_magnitudes = []
    for data in [vortex_data, saddle_data]:
        for key in ['nn_u', 'nn_v', 'rbf_u', 'rbf_v']:
            mag = np.sqrt(data[key]**2 + data[key.replace('u', 'v').replace('v', 'u')]**2)
            all_magnitudes.extend(mag)

    mag_norm = Normalize(vmin=0, vmax=np.percentile(all_magnitudes, 95))

    # Reshape fields for plotting
    shape = vortex_data['X'].shape

    # Row 1: NN Vortex
    ax1 = plt.subplot(4, 2, 1)
    U_nn_vortex = vortex_data['nn_u'].reshape(shape)
    V_nn_vortex = vortex_data['nn_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'],
                                                U_nn_vortex, V_nn_vortex)
    M = np.sqrt(U_q**2 + V_q**2)
    Q = ax1.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis',
                   scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax1.set_title('NN Vortex Field', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylim(-0.6, 0.6)

    ax2 = plt.subplot(4, 2, 2)
    du, dv = compute_field_difference(vortex_data['nn_u'], vortex_data['nn_v'],
                                      vortex_data['analytical_u'], vortex_data['analytical_v'],
                                      DIFFERENCE_SCALE)
    DU = du.reshape(shape)
    DV = dv.reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'], DU, DV)
    M = np.sqrt(U_q**2 + V_q**2)
    ax2.quiver(X_q, Y_q, U_q, V_q, M, cmap='hot',
               scale=ARROW_SCALE, alpha=0.8)
    ax2.set_title(f'NN - Analytical (×{DIFFERENCE_SCALE})', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)

    # Row 2: RBF Vortex
    ax3 = plt.subplot(4, 2, 3)
    U_rbf_vortex = vortex_data['rbf_u'].reshape(shape)
    V_rbf_vortex = vortex_data['rbf_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'],
                                                U_rbf_vortex, V_rbf_vortex)
    M = np.sqrt(U_q**2 + V_q**2)
    ax3.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis',
               scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax3.set_title('RBF Vortex Field', fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.6, 0.6)
    ax3.set_ylim(-0.6, 0.6)

    ax4 = plt.subplot(4, 2, 4)
    du, dv = compute_field_difference(vortex_data['rbf_u'], vortex_data['rbf_v'],
                                      vortex_data['analytical_u'], vortex_data['analytical_v'],
                                      DIFFERENCE_SCALE)
    DU = du.reshape(shape)
    DV = dv.reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'], DU, DV)
    M = np.sqrt(U_q**2 + V_q**2)
    ax4.quiver(X_q, Y_q, U_q, V_q, M, cmap='hot',
               scale=ARROW_SCALE, alpha=0.8)
    ax4.set_title(f'RBF - Analytical (×{DIFFERENCE_SCALE})', fontsize=12, fontweight='bold')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.6, 0.6)
    ax4.set_ylim(-0.6, 0.6)

    # Row 3: NN Saddle
    ax5 = plt.subplot(4, 2, 5)
    U_nn_saddle = saddle_data['nn_u'].reshape(shape)
    V_nn_saddle = saddle_data['nn_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'],
                                                U_nn_saddle, V_nn_saddle)
    M = np.sqrt(U_q**2 + V_q**2)
    ax5.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis',
               scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax5.set_title('NN Saddle Field', fontsize=12, fontweight='bold')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.6, 0.6)
    ax5.set_ylim(-0.6, 0.6)

    ax6 = plt.subplot(4, 2, 6)
    du, dv = compute_field_difference(saddle_data['nn_u'], saddle_data['nn_v'],
                                      saddle_data['analytical_u'], saddle_data['analytical_v'],
                                      DIFFERENCE_SCALE)
    DU = du.reshape(shape)
    DV = dv.reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'], DU, DV)
    M = np.sqrt(U_q**2 + V_q**2)
    ax6.quiver(X_q, Y_q, U_q, V_q, M, cmap='hot',
               scale=ARROW_SCALE, alpha=0.8)
    ax6.set_title(f'NN - Analytical (×{DIFFERENCE_SCALE})', fontsize=12, fontweight='bold')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-0.6, 0.6)
    ax6.set_ylim(-0.6, 0.6)

    # Row 4: RBF Saddle
    ax7 = plt.subplot(4, 2, 7)
    U_rbf_saddle = saddle_data['rbf_u'].reshape(shape)
    V_rbf_saddle = saddle_data['rbf_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'],
                                                U_rbf_saddle, V_rbf_saddle)
    M = np.sqrt(U_q**2 + V_q**2)
    ax7.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis',
               scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax7.set_title('RBF Saddle Field', fontsize=12, fontweight='bold')
    ax7.set_aspect('equal')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-0.6, 0.6)
    ax7.set_ylim(-0.6, 0.6)

    ax8 = plt.subplot(4, 2, 8)
    du, dv = compute_field_difference(saddle_data['rbf_u'], saddle_data['rbf_v'],
                                      saddle_data['analytical_u'], saddle_data['analytical_v'],
                                      DIFFERENCE_SCALE)
    DU = du.reshape(shape)
    DV = dv.reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'], DU, DV)
    M = np.sqrt(U_q**2 + V_q**2)
    ax8.quiver(X_q, Y_q, U_q, V_q, M, cmap='hot',
               scale=ARROW_SCALE, alpha=0.8)
    ax8.set_title(f'RBF - Analytical (×{DIFFERENCE_SCALE})', fontsize=12, fontweight='bold')
    ax8.set_aspect('equal')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(-0.6, 0.6)
    ax8.set_ylim(-0.6, 0.6)

    # Add colorbar for magnitude
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    plt.colorbar(Q, cax=cbar_ax, label='Magnitude')

    plt.suptitle('Field Comparison: NN and RBF vs Analytical', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])

    return fig


def create_6x2_visualization(vortex_data, saddle_data):
    """
    Create 6x2 grid visualization (including analytical fields).

    Args:
        vortex_data: Dict with vortex field evaluations
        saddle_data: Dict with saddle field evaluations
    """
    fig = plt.figure(figsize=(14, 24))

    # Color normalization
    all_magnitudes = []
    for data in [vortex_data, saddle_data]:
        for key in ['analytical_u', 'nn_u', 'rbf_u']:
            mag = np.sqrt(data[key]**2 + data[key.replace('u', 'v')]**2)
            all_magnitudes.extend(mag)

    mag_norm = Normalize(vmin=0, vmax=np.percentile(all_magnitudes, 95))

    shape = vortex_data['X'].shape

    # Row 1: Analytical Vortex
    ax1 = plt.subplot(6, 2, 1)
    U_ana = vortex_data['analytical_u'].reshape(shape)
    V_ana = vortex_data['analytical_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'], U_ana, V_ana)
    M = np.sqrt(U_q**2 + V_q**2)
    Q = ax1.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis', scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax1.set_title('Analytical Vortex', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylim(-0.6, 0.6)

    ax2 = plt.subplot(6, 2, 2)
    ax2.text(0.5, 0.5, 'Null\n(Analytical - Analytical = 0)',
             ha='center', va='center', fontsize=12, transform=ax2.transAxes)
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.grid(True, alpha=0.3)

    # Row 2: NN Vortex
    ax3 = plt.subplot(6, 2, 3)
    U_nn = vortex_data['nn_u'].reshape(shape)
    V_nn = vortex_data['nn_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'], U_nn, V_nn)
    M = np.sqrt(U_q**2 + V_q**2)
    ax3.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis', scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax3.set_title('NN Vortex', fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.6, 0.6)
    ax3.set_ylim(-0.6, 0.6)

    ax4 = plt.subplot(6, 2, 4)
    du, dv = compute_field_difference(vortex_data['nn_u'], vortex_data['nn_v'],
                                      vortex_data['analytical_u'], vortex_data['analytical_v'],
                                      DIFFERENCE_SCALE)
    DU = du.reshape(shape)
    DV = dv.reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'], DU, DV)
    M = np.sqrt(U_q**2 + V_q**2)
    ax4.quiver(X_q, Y_q, U_q, V_q, M, cmap='hot', scale=ARROW_SCALE, alpha=0.8)
    ax4.set_title(f'NN - Analytical (×{DIFFERENCE_SCALE})', fontsize=12, fontweight='bold')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.6, 0.6)
    ax4.set_ylim(-0.6, 0.6)

    # Row 3: RBF Vortex
    ax5 = plt.subplot(6, 2, 5)
    U_rbf = vortex_data['rbf_u'].reshape(shape)
    V_rbf = vortex_data['rbf_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'], U_rbf, V_rbf)
    M = np.sqrt(U_q**2 + V_q**2)
    ax5.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis', scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax5.set_title('RBF Vortex', fontsize=12, fontweight='bold')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.6, 0.6)
    ax5.set_ylim(-0.6, 0.6)

    ax6 = plt.subplot(6, 2, 6)
    du, dv = compute_field_difference(vortex_data['rbf_u'], vortex_data['rbf_v'],
                                      vortex_data['analytical_u'], vortex_data['analytical_v'],
                                      DIFFERENCE_SCALE)
    DU = du.reshape(shape)
    DV = dv.reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(vortex_data['X'], vortex_data['Y'], DU, DV)
    M = np.sqrt(U_q**2 + V_q**2)
    ax6.quiver(X_q, Y_q, U_q, V_q, M, cmap='hot', scale=ARROW_SCALE, alpha=0.8)
    ax6.set_title(f'RBF - Analytical (×{DIFFERENCE_SCALE})', fontsize=12, fontweight='bold')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-0.6, 0.6)
    ax6.set_ylim(-0.6, 0.6)

    # Row 4: Analytical Saddle
    ax7 = plt.subplot(6, 2, 7)
    U_ana = saddle_data['analytical_u'].reshape(shape)
    V_ana = saddle_data['analytical_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'], U_ana, V_ana)
    M = np.sqrt(U_q**2 + V_q**2)
    ax7.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis', scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax7.set_title('Analytical Saddle', fontsize=12, fontweight='bold')
    ax7.set_aspect('equal')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-0.6, 0.6)
    ax7.set_ylim(-0.6, 0.6)

    ax8 = plt.subplot(6, 2, 8)
    ax8.text(0.5, 0.5, 'Null\n(Analytical - Analytical = 0)',
             ha='center', va='center', fontsize=12, transform=ax8.transAxes)
    ax8.set_aspect('equal')
    ax8.set_xlim(-0.6, 0.6)
    ax8.set_ylim(-0.6, 0.6)
    ax8.grid(True, alpha=0.3)

    # Row 5: NN Saddle
    ax9 = plt.subplot(6, 2, 9)
    U_nn = saddle_data['nn_u'].reshape(shape)
    V_nn = saddle_data['nn_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'], U_nn, V_nn)
    M = np.sqrt(U_q**2 + V_q**2)
    ax9.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis', scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax9.set_title('NN Saddle', fontsize=12, fontweight='bold')
    ax9.set_aspect('equal')
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(-0.6, 0.6)
    ax9.set_ylim(-0.6, 0.6)

    ax10 = plt.subplot(6, 2, 10)
    du, dv = compute_field_difference(saddle_data['nn_u'], saddle_data['nn_v'],
                                      saddle_data['analytical_u'], saddle_data['analytical_v'],
                                      DIFFERENCE_SCALE)
    DU = du.reshape(shape)
    DV = dv.reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'], DU, DV)
    M = np.sqrt(U_q**2 + V_q**2)
    ax10.quiver(X_q, Y_q, U_q, V_q, M, cmap='hot', scale=ARROW_SCALE, alpha=0.8)
    ax10.set_title(f'NN - Analytical (×{DIFFERENCE_SCALE})', fontsize=12, fontweight='bold')
    ax10.set_aspect('equal')
    ax10.grid(True, alpha=0.3)
    ax10.set_xlim(-0.6, 0.6)
    ax10.set_ylim(-0.6, 0.6)

    # Row 6: RBF Saddle
    ax11 = plt.subplot(6, 2, 11)
    U_rbf = saddle_data['rbf_u'].reshape(shape)
    V_rbf = saddle_data['rbf_v'].reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'], U_rbf, V_rbf)
    M = np.sqrt(U_q**2 + V_q**2)
    ax11.quiver(X_q, Y_q, U_q, V_q, M, cmap='viridis', scale=ARROW_SCALE, norm=mag_norm, alpha=0.8)
    ax11.set_title('RBF Saddle', fontsize=12, fontweight='bold')
    ax11.set_aspect('equal')
    ax11.grid(True, alpha=0.3)
    ax11.set_xlim(-0.6, 0.6)
    ax11.set_ylim(-0.6, 0.6)

    ax12 = plt.subplot(6, 2, 12)
    du, dv = compute_field_difference(saddle_data['rbf_u'], saddle_data['rbf_v'],
                                      saddle_data['analytical_u'], saddle_data['analytical_v'],
                                      DIFFERENCE_SCALE)
    DU = du.reshape(shape)
    DV = dv.reshape(shape)
    X_q, Y_q, U_q, V_q = downsample_for_quiver(saddle_data['X'], saddle_data['Y'], DU, DV)
    M = np.sqrt(U_q**2 + V_q**2)
    ax12.quiver(X_q, Y_q, U_q, V_q, M, cmap='hot', scale=ARROW_SCALE, alpha=0.8)
    ax12.set_title(f'RBF - Analytical (×{DIFFERENCE_SCALE})', fontsize=12, fontweight='bold')
    ax12.set_aspect('equal')
    ax12.grid(True, alpha=0.3)
    ax12.set_xlim(-0.6, 0.6)
    ax12.set_ylim(-0.6, 0.6)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    plt.colorbar(Q, cax=cbar_ax, label='Magnitude')

    plt.suptitle('Field Comparison: Analytical, NN, and RBF', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])

    return fig


def compute_error_statistics(u_pred, v_pred, u_true, v_true):
    """
    Compute error statistics between predicted and true fields.

    Args:
        u_pred, v_pred: Predicted field components
        u_true, v_true: True field components

    Returns:
        Dict with error statistics
    """
    # Magnitude errors
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    mag_true = np.sqrt(u_true**2 + v_true**2)
    mag_error = np.abs(mag_pred - mag_true)

    # Direction errors
    angle_pred = np.arctan2(v_pred, u_pred)
    angle_true = np.arctan2(v_true, u_true)
    angle_error = np.abs(angle_pred - angle_true)
    angle_error = np.minimum(angle_error, 2*np.pi - angle_error)
    angle_error_degrees = np.rad2deg(angle_error)

    # Vector error
    vector_error = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)

    return {
        'mag_error_mean': np.mean(mag_error),
        'mag_error_max': np.max(mag_error),
        'angle_error_mean_deg': np.mean(angle_error_degrees),
        'angle_error_max_deg': np.max(angle_error_degrees),
        'vector_error_mean': np.mean(vector_error),
        'vector_error_max': np.max(vector_error)
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("FIELD COMPARISON CONSOLIDATED VISUALIZATION")
    print("=" * 80)

    # Create evaluation grid
    print("\nCreating evaluation grid...")
    X, Y, x_flat, y_flat = create_evaluation_grid(GRID_RESOLUTION)
    print(f"Grid created: {GRID_RESOLUTION}×{GRID_RESOLUTION}")

    # Initialize fields
    print("\n" + "="*40)
    print("VORTEX FIELD EVALUATION")
    print("="*40)

    # Vortex fields
    print("Loading vortex fields...")
    vortex_analytical = AnalyticalField(vortex1)
    vortex_nn = NNField(predictor_dir='vortex_predictors')
    vortex_rbf = RBFField(predictor_dir='vortex_predictors')

    # Evaluate vortex fields
    print("Evaluating analytical vortex...")
    vortex_u_ana, vortex_v_ana = evaluate_field_batch(vortex_analytical, x_flat, y_flat)

    print("Evaluating NN vortex...")
    vortex_u_nn, vortex_v_nn = evaluate_field_batch(vortex_nn, x_flat, y_flat)

    print("Evaluating RBF vortex...")
    vortex_u_rbf, vortex_v_rbf = evaluate_field_batch(vortex_rbf, x_flat, y_flat)

    # Store vortex data
    vortex_data = {
        'X': X, 'Y': Y,
        'analytical_u': vortex_u_ana, 'analytical_v': vortex_v_ana,
        'nn_u': vortex_u_nn, 'nn_v': vortex_v_nn,
        'rbf_u': vortex_u_rbf, 'rbf_v': vortex_v_rbf
    }

    # Compute vortex errors
    print("\nVortex Field Error Statistics:")
    print("-" * 40)
    nn_errors = compute_error_statistics(vortex_u_nn, vortex_v_nn, vortex_u_ana, vortex_v_ana)
    print("NN vs Analytical:")
    print(f"  Magnitude error: mean={nn_errors['mag_error_mean']:.4f}, max={nn_errors['mag_error_max']:.4f}")
    print(f"  Angle error: mean={nn_errors['angle_error_mean_deg']:.1f}°, max={nn_errors['angle_error_max_deg']:.1f}°")
    print(f"  Vector error: mean={nn_errors['vector_error_mean']:.4f}, max={nn_errors['vector_error_max']:.4f}")

    rbf_errors = compute_error_statistics(vortex_u_rbf, vortex_v_rbf, vortex_u_ana, vortex_v_ana)
    print("RBF vs Analytical:")
    print(f"  Magnitude error: mean={rbf_errors['mag_error_mean']:.4f}, max={rbf_errors['mag_error_max']:.4f}")
    print(f"  Angle error: mean={rbf_errors['angle_error_mean_deg']:.1f}°, max={rbf_errors['angle_error_max_deg']:.1f}°")
    print(f"  Vector error: mean={rbf_errors['vector_error_mean']:.4f}, max={rbf_errors['vector_error_max']:.4f}")

    print("\n" + "="*40)
    print("SADDLE FIELD EVALUATION")
    print("="*40)

    # Saddle fields
    print("Loading saddle fields...")
    saddle_analytical = AnalyticalField(saddle1)
    saddle_nn = NNField(predictor_dir='saddle_predictors')
    saddle_rbf = RBFField(predictor_dir='saddle_predictors')

    # Evaluate saddle fields
    print("Evaluating analytical saddle...")
    saddle_u_ana, saddle_v_ana = evaluate_field_batch(saddle_analytical, x_flat, y_flat)

    print("Evaluating NN saddle...")
    saddle_u_nn, saddle_v_nn = evaluate_field_batch(saddle_nn, x_flat, y_flat)

    print("Evaluating RBF saddle...")
    saddle_u_rbf, saddle_v_rbf = evaluate_field_batch(saddle_rbf, x_flat, y_flat)

    # Store saddle data
    saddle_data = {
        'X': X, 'Y': Y,
        'analytical_u': saddle_u_ana, 'analytical_v': saddle_v_ana,
        'nn_u': saddle_u_nn, 'nn_v': saddle_v_nn,
        'rbf_u': saddle_u_rbf, 'rbf_v': saddle_v_rbf
    }

    # Compute saddle errors
    print("\nSaddle Field Error Statistics:")
    print("-" * 40)
    nn_errors = compute_error_statistics(saddle_u_nn, saddle_v_nn, saddle_u_ana, saddle_v_ana)
    print("NN vs Analytical:")
    print(f"  Magnitude error: mean={nn_errors['mag_error_mean']:.4f}, max={nn_errors['mag_error_max']:.4f}")
    print(f"  Angle error: mean={nn_errors['angle_error_mean_deg']:.1f}°, max={nn_errors['angle_error_max_deg']:.1f}°")
    print(f"  Vector error: mean={nn_errors['vector_error_mean']:.4f}, max={nn_errors['vector_error_max']:.4f}")

    rbf_errors = compute_error_statistics(saddle_u_rbf, saddle_v_rbf, saddle_u_ana, saddle_v_ana)
    print("RBF vs Analytical:")
    print(f"  Magnitude error: mean={rbf_errors['mag_error_mean']:.4f}, max={rbf_errors['mag_error_max']:.4f}")
    print(f"  Angle error: mean={rbf_errors['angle_error_mean_deg']:.1f}°, max={rbf_errors['angle_error_max_deg']:.1f}°")
    print(f"  Vector error: mean={rbf_errors['vector_error_mean']:.4f}, max={rbf_errors['vector_error_max']:.4f}")

    # Create visualization
    print("\n" + "="*40)
    print(f"Creating {LAYOUT_MODE} visualization...")
    print("="*40)

    if LAYOUT_MODE == "4x2":
        fig = create_4x2_visualization(vortex_data, saddle_data)
        output_filename = 'field_comparison_4x2.png'
    else:
        fig = create_6x2_visualization(vortex_data, saddle_data)
        output_filename = 'field_comparison_6x2.png'

    # Save figure
    print(f"\nSaving figure as '{output_filename}'...")
    fig.savefig(output_filename, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_filename}")

    # Show figure (commented out for non-interactive running)
    # plt.show()

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()