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
    Create comprehensive field comparison plot.
    """
    fig = plt.figure(figsize=(18, 12))

    # Reshape for plotting
    U_analytical = u_analytical.reshape(X.shape)
    V_analytical = v_analytical.reshape(X.shape)
    U_approx = u_approx.reshape(X.shape)
    V_approx = v_approx.reshape(X.shape)

    # Compute errors (Analytical - Approx shows what the approximation is missing)
    U_error = U_analytical - U_approx
    V_error = V_analytical - V_approx
    vector_error = np.sqrt(U_error**2 + V_error**2)

    # Calculate average field magnitude for scaling reference
    avg_field_magnitude = np.mean(np.sqrt(U_analytical**2 + V_analytical**2))
    avg_error_magnitude = np.mean(vector_error)

    # Determine error vector scale factor
    # We want error vectors to be visually clear but not misleading
    error_scale_factor = 3.0  # Scale error vectors up by 3x for visibility

    # Subsample for clarity (fewer, larger arrows)
    skip_field = 5  # Every 5th point for field vectors (10x10 = 100 arrows, 20% further reduction)
    skip_error = 3  # Every 3rd point for error vectors (17x17 = 289 arrows)

    # Plot 1: Analytical Field
    ax1 = plt.subplot(2, 3, 1)
    Q1 = ax1.quiver(X[::skip_field, ::skip_field], Y[::skip_field, ::skip_field],
                     U_analytical[::skip_field, ::skip_field],
                     V_analytical[::skip_field, ::skip_field],
                     alpha=0.7, scale=5.0, width=0.004)
    ax1.quiverkey(Q1, 0.9, 0.95, 0.2, '0.2 m/s', labelpos='E', coordinates='axes')
    ax1.set_title('Analytical Field (Ground Truth)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Plot 2: Approximation Field
    ax2 = plt.subplot(2, 3, 2)
    Q2 = ax2.quiver(X[::skip_field, ::skip_field], Y[::skip_field, ::skip_field],
                     U_approx[::skip_field, ::skip_field],
                     V_approx[::skip_field, ::skip_field],
                     alpha=0.7, scale=5.0, width=0.004)
    ax2.quiverkey(Q2, 0.9, 0.95, 0.2, '0.2 m/s', labelpos='E', coordinates='axes')
    ax2.set_title(f'{title} Field', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.add_patch(Circle((0, 0), 0.02, color='red', zorder=10))

    # Plot 3: Error Vectors (scaled up for visibility)
    ax3 = plt.subplot(2, 3, 3)
    Q3 = ax3.quiver(X[::skip_error, ::skip_error], Y[::skip_error, ::skip_error],
                     U_error[::skip_error, ::skip_error] * error_scale_factor,
                     V_error[::skip_error, ::skip_error] * error_scale_factor,
                     vector_error[::skip_error, ::skip_error],
                     cmap='Reds', alpha=0.8, scale=5.0, width=0.004)
    # Add reference arrow showing actual scale
    ax3.quiverkey(Q3, 0.9, 0.95, 0.1 * error_scale_factor,
                  f'0.1 m/s\n(scaled {error_scale_factor:.0f}×)',
                  labelpos='E', coordinates='axes')
    ax3.set_title('Error Vectors (Analytical - Approx)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Add text annotation about scaling
    ax3.text(0.02, 0.98, f'Error vectors scaled {error_scale_factor:.0f}× for visibility',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Vector Magnitude Error Heatmap
    ax4 = plt.subplot(2, 3, 4)
    im1 = ax4.contourf(X, Y, vector_error, levels=20, cmap='hot')
    ax4.set_title('Vector Magnitude Error', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_aspect('equal')
    plt.colorbar(im1, ax=ax4, label='Error (m/s)')

    # Plot 5: Direction Error Heatmap
    ax5 = plt.subplot(2, 3, 5)
    angle_analytical = np.arctan2(V_analytical, U_analytical)
    angle_approx = np.arctan2(V_approx, U_approx)
    angle_error = np.abs(angle_analytical - angle_approx)
    angle_error = np.minimum(angle_error, 2*np.pi - angle_error)
    angle_error_deg = np.rad2deg(angle_error)
    im2 = ax5.contourf(X, Y, angle_error_deg, levels=20, cmap='plasma')
    ax5.set_title('Direction Error', fontsize=12, fontweight='bold')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('y (m)')
    ax5.set_aspect('equal')
    plt.colorbar(im2, ax=ax5, label='Error (degrees)')

    # Plot 6: Magnitude Error Heatmap
    ax6 = plt.subplot(2, 3, 6)
    mag_analytical = np.sqrt(U_analytical**2 + V_analytical**2)
    mag_approx = np.sqrt(U_approx**2 + V_approx**2)
    mag_error = np.abs(mag_approx - mag_analytical)
    im3 = ax6.contourf(X, Y, mag_error, levels=20, cmap='viridis')
    ax6.set_title('Magnitude Error', fontsize=12, fontweight='bold')
    ax6.set_xlabel('x (m)')
    ax6.set_ylabel('y (m)')
    ax6.set_aspect('equal')
    plt.colorbar(im3, ax=ax6, label='Error (m/s)')

    # Add scale information
    physical_size = 0.5 * 2 * scale_factor  # Total physical size
    fig.suptitle(f'{title} vs Analytical Field Comparison\n' +
                 f'Simulation Domain: {SIMULATION_DOMAIN:.1f}m | ' +
                 f'Physical Domain: {physical_size:.3f}m ({physical_size*39.37:.1f} inches) | ' +
                 f'Scale Factor: {scale_factor:.4f}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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

    # Generate plots
    print("Generating plots...")
    plot_field_comparison(X, Y, u_analytical, v_analytical, u_nn, v_nn,
                         'NN', 'nn_vs_analytical.png', SCALE_FACTOR)
    plot_field_comparison(X, Y, u_analytical, v_analytical, u_rbf, v_rbf,
                         'RBF', 'rbf_vs_analytical.png', SCALE_FACTOR)
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
