"""
Publication-Quality Scalar Field Visualization Utility

Usage:
    python3 visualize_field.py

Generates high-quality plots for:
- Field contours
- Gradient vector field
- Hessian eigenvalue analysis
- Critical point classification
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.fields.scalar_utils import compute_gradient, analyze_critical_point
from src.fields.scalar_environments.Saddle import bimodal_saddle1


def visualize_scalar_field(field_func, xrange=(-3, 3), yrange=(-3, 3),
                          resolution=100, critical_point=(0, 0),
                          save_path='field_visualization.png'):
    """
    Create publication-quality visualization of scalar field.

    Args:
        field_func: Scalar field function (x,y) -> z
        xrange: (xmin, xmax) for evaluation
        yrange: (ymin, ymax) for evaluation
        resolution: Grid resolution
        critical_point: (x,y) location to analyze
        save_path: Output file path
    """
    # Grid
    x = np.linspace(xrange[0], xrange[1], resolution)
    y = np.linspace(yrange[0], yrange[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate field
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = field_func(X[i, j], Y[i, j])

    # Compute gradients (coarser grid for vectors)
    skip = 5
    X_grad = X[::skip, ::skip]
    Y_grad = Y[::skip, ::skip]
    U_grad = np.zeros_like(X_grad)
    V_grad = np.zeros_like(Y_grad)

    for i in range(X_grad.shape[0]):
        for j in range(X_grad.shape[1]):
            grad = compute_gradient(field_func, X_grad[i, j], Y_grad[i, j])
            U_grad[i, j] = grad[0]
            V_grad[i, j] = grad[1]

    # Analyze critical point
    analysis = analyze_critical_point(field_func, critical_point[0], critical_point[1])

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))

    # 1. Field contours
    ax1 = plt.subplot(131)
    contourf = ax1.contourf(X, Y, Z, levels=30, cmap='RdBu_r', alpha=0.8)
    contours = ax1.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    ax1.clabel(contours, inline=True, fontsize=7, fmt='%1.1f')
    ax1.plot(critical_point[0], critical_point[1], 'k*', markersize=20, label='Critical Point')
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title('Scalar Field z(x,y)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=10)
    plt.colorbar(contourf, ax=ax1, label='z')

    # 2. Gradient field
    ax2 = plt.subplot(132)
    ax2.contourf(X, Y, Z, levels=30, cmap='RdBu_r', alpha=0.3)
    ax2.quiver(X_grad, Y_grad, U_grad, V_grad, alpha=0.7, color='blue', scale=10)
    ax2.plot(critical_point[0], critical_point[1], 'k*', markersize=20, label='Critical Point')
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)
    ax2.set_title('Gradient Field ∇z(x,y)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=10)

    # 3. Critical point analysis
    ax3 = plt.subplot(133)
    ax3.contourf(X, Y, Z, levels=30, cmap='RdBu_r', alpha=0.3)
    ax3.plot(critical_point[0], critical_point[1], 'k*', markersize=20, label='Critical Point')

    # Eigenvector directions
    eigenvals = analysis['eigenvalues']
    eigenvecs = analysis['eigenvectors']
    scale = 0.8

    # Negative eigenvalue (attractive)
    ax3.arrow(critical_point[0], critical_point[1],
             eigenvecs[0, 0] * scale, eigenvecs[1, 0] * scale,
             head_width=0.15, head_length=0.2, fc='green', ec='green',
             alpha=0.8, linewidth=3, label=f'λ₁={eigenvals[0]:.2f} (attract)')

    # Positive eigenvalue (repulsive)
    ax3.arrow(critical_point[0], critical_point[1],
             eigenvecs[0, 1] * scale, eigenvecs[1, 1] * scale,
             head_width=0.15, head_length=0.2, fc='red', ec='red',
             alpha=0.8, linewidth=3, label=f'λ₂={eigenvals[1]:.2f} (repel)')

    # Analysis text
    classification = analysis['classification']
    grad_norm = np.linalg.norm(analysis['gradient'])
    info_text = (f"Classification: {classification.upper()}\n"
                f"Gradient norm: {grad_norm:.2e}\n"
                f"Eigenvalues: λ₁={eigenvals[0]:.3f}, λ₂={eigenvals[1]:.3f}\n"
                f"Det(H): {eigenvals[0]*eigenvals[1]:.3f}")

    ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax3.set_xlabel('x (m)', fontsize=12)
    ax3.set_ylabel('y (m)', fontsize=12)
    ax3.set_title('Hessian Eigenanalysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    # Print analysis
    print("\n" + "="*60)
    print("CRITICAL POINT ANALYSIS")
    print("="*60)
    print(f"Location: ({critical_point[0]:.2f}, {critical_point[1]:.2f})")
    print(f"Classification: {classification.upper()}")
    print(f"Gradient norm: {grad_norm:.2e}")
    print(f"Hessian eigenvalues: λ₁={eigenvals[0]:.4f}, λ₂={eigenvals[1]:.4f}")
    print(f"Determinant: {eigenvals[0]*eigenvals[1]:.4f}")
    print(f"Is critical point: {analysis['is_critical']}")
    print("="*60)

    return fig, analysis


if __name__ == "__main__":
    # Example: Visualize bimodal saddle
    print("Generating visualization of bimodal_saddle1...")
    visualize_scalar_field(
        field_func=bimodal_saddle1,
        xrange=(-3, 3),
        yrange=(-3, 3),
        resolution=150,
        critical_point=(0, 0),
        save_path='bimodal_saddle_visualization.png'
    )
    plt.show()
