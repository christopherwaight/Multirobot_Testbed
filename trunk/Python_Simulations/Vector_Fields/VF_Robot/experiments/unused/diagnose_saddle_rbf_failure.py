"""
Diagnostic Analysis: Why RBF Saddle Field Fails

This script investigates the catastrophic failure of RBF saddle field approximation
in orbital control (errors up to 4m) and convergence (RMSE 0.096m vs 0.015m analytical).

Compares:
- Analytical saddle1 (45° rotation, ground truth)
- RBF saddle approximation (trained on real sensor data)
- NN saddle approximation (trained on same data as RBF)
- Analytical + noise (Gaussian σ=0.02m to simulate sensor noise)

Analysis:
1. Field Quality: Vector field comparison on 50×50 grid
2. Orbital Control: Trajectory tracking at problem radii [0.2-0.7m]
3. Convergence: Critical point finding from random starts
4. Data Coverage: RBF/NN training data distribution

Author: Claude Code
Date: 2026-03-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from time import time
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fields.field_types import AnalyticalField, NNField, RBFField
from src.fields.environments.Saddle import saddle1
from src.robot.omni_cluster import OmniCluster
import src.control.primitives as ocp

# Configuration
OUTPUT_DIR = 'saddle_rbf_diagnosis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SADDLE_PREDICTOR_DIR = 'saddle_predictors'

# Absolute path to formation config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VF_ROBOT_DIR = os.path.dirname(SCRIPT_DIR)
FORMATION_CONFIG = os.path.join(VF_ROBOT_DIR, 'config', 'formations', 'equilateral_default.yaml')

# Test configurations
GRID_RESOLUTION = 50
ORBITAL_TEST_RADII = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Focus on problem radii
ORBITAL_SIM_TIME = 60  # seconds
CONVERGENCE_TRIALS = 20

print("="*80)
print("SADDLE RBF/NN FAILURE DIAGNOSTIC")
print("="*80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Random seed: 42")
print()

# ============================================================================
# PART 1: FIELD QUALITY COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PART 1: FIELD QUALITY COMPARISON (50×50 Grid)")
print("="*80)

def create_evaluation_grid(resolution=50):
    """Create evaluation grid."""
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y, X.flatten(), Y.flatten()

def evaluate_field_grid(field, x_flat, y_flat):
    """Evaluate field at all grid points."""
    u_list, v_list = [], []
    for x, y in zip(x_flat, y_flat):
        u, v = field.get_value(x, y)
        u_list.append(u)
        v_list.append(v)
    return np.array(u_list), np.array(v_list)

def compute_field_errors(u_ref, v_ref, u_test, v_test):
    """Compute magnitude and angle errors."""
    # Magnitude error
    mag_ref = np.sqrt(u_ref**2 + v_ref**2)
    mag_test = np.sqrt(u_test**2 + v_test**2)
    mag_error = np.abs(mag_test - mag_ref)

    # Angle error (radians)
    angle_ref = np.arctan2(v_ref, u_ref)
    angle_test = np.arctan2(v_test, u_test)
    angle_error = np.abs(angle_test - angle_ref)
    # Wrap to [-π, π]
    angle_error = np.where(angle_error > np.pi, 2*np.pi - angle_error, angle_error)

    return mag_error, angle_error

# Create grid
X, Y, x_flat, y_flat = create_evaluation_grid(GRID_RESOLUTION)

# Create fields
print("\nLoading fields...")
field_analytical = AnalyticalField(saddle1)
field_rbf = RBFField(SADDLE_PREDICTOR_DIR)
field_nn = NNField(SADDLE_PREDICTOR_DIR)
field_noisy = AnalyticalField(saddle1)  # Will add noise manually
print("✓ All fields loaded")

# Evaluate fields
print("\nEvaluating fields on grid...")
u_analytical, v_analytical = evaluate_field_grid(field_analytical, x_flat, y_flat)
u_rbf, v_rbf = evaluate_field_grid(field_rbf, x_flat, y_flat)
u_nn, v_nn = evaluate_field_grid(field_nn, x_flat, y_flat)

# Add noise to analytical
u_noisy = u_analytical + np.random.randn(len(u_analytical)) * 0.02
v_noisy = v_analytical + np.random.randn(len(v_analytical)) * 0.02

print("✓ Field evaluation complete")

# Compute errors
print("\nComputing error metrics...")
mag_err_rbf, ang_err_rbf = compute_field_errors(u_analytical, v_analytical, u_rbf, v_rbf)
mag_err_nn, ang_err_nn = compute_field_errors(u_analytical, v_analytical, u_nn, v_nn)
mag_err_noisy, ang_err_noisy = compute_field_errors(u_analytical, v_analytical, u_noisy, v_noisy)

# Print statistics
print("\n" + "-"*80)
print("FIELD QUALITY STATISTICS")
print("-"*80)
print(f"{'Field Type':<20} {'Mag Error':<25} {'Angle Error (deg)':<25}")
print("-"*80)
print(f"{'RBF':<20} {np.mean(mag_err_rbf):.4f} ± {np.std(mag_err_rbf):.4f} (max: {np.max(mag_err_rbf):.4f})     {np.mean(np.rad2deg(ang_err_rbf)):.2f} ± {np.std(np.rad2deg(ang_err_rbf)):.2f} (max: {np.max(np.rad2deg(ang_err_rbf)):.2f})")
print(f"{'NN':<20} {np.mean(mag_err_nn):.4f} ± {np.std(mag_err_nn):.4f} (max: {np.max(mag_err_nn):.4f})     {np.mean(np.rad2deg(ang_err_nn)):.2f} ± {np.std(np.rad2deg(ang_err_nn)):.2f} (max: {np.max(np.rad2deg(ang_err_nn)):.2f})")
print(f"{'Analytical+Noise':<20} {np.mean(mag_err_noisy):.4f} ± {np.std(mag_err_noisy):.4f} (max: {np.max(mag_err_noisy):.4f})     {np.mean(np.rad2deg(ang_err_noisy)):.2f} ± {np.std(np.rad2deg(ang_err_noisy)):.2f} (max: {np.max(np.rad2deg(ang_err_noisy)):.2f})")
print("-"*80)

# Visualize fields
print("\nGenerating field comparison plots...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Reshape for plotting
U_analytical = u_analytical.reshape(X.shape)
V_analytical = v_analytical.reshape(X.shape)
U_rbf = u_rbf.reshape(X.shape)
V_rbf = v_rbf.reshape(X.shape)
U_nn = u_nn.reshape(X.shape)
V_nn = v_nn.reshape(X.shape)
U_noisy = u_noisy.reshape(X.shape)
V_noisy = v_noisy.reshape(X.shape)

mag_err_rbf_grid = mag_err_rbf.reshape(X.shape)
mag_err_nn_grid = mag_err_nn.reshape(X.shape)
ang_err_rbf_grid = ang_err_rbf.reshape(X.shape)
ang_err_nn_grid = ang_err_nn.reshape(X.shape)

# Plot quiver fields
skip = 3
axes[0, 0].quiver(X[::skip, ::skip], Y[::skip, ::skip], U_analytical[::skip, ::skip], V_analytical[::skip, ::skip], alpha=0.6)
axes[0, 0].set_title("Analytical (Ground Truth)")
axes[0, 0].set_aspect('equal')
axes[0, 0].plot(0, 0, 'ro', markersize=8)

axes[0, 1].quiver(X[::skip, ::skip], Y[::skip, ::skip], U_rbf[::skip, ::skip], V_rbf[::skip, ::skip], alpha=0.6)
axes[0, 1].set_title("RBF Approximation")
axes[0, 1].set_aspect('equal')
axes[0, 1].plot(0, 0, 'ro', markersize=8)

axes[0, 2].quiver(X[::skip, ::skip], Y[::skip, ::skip], U_nn[::skip, ::skip], V_nn[::skip, ::skip], alpha=0.6)
axes[0, 2].set_title("NN Approximation")
axes[0, 2].set_aspect('equal')
axes[0, 2].plot(0, 0, 'ro', markersize=8)

axes[0, 3].quiver(X[::skip, ::skip], Y[::skip, ::skip], U_noisy[::skip, ::skip], V_noisy[::skip, ::skip], alpha=0.6)
axes[0, 3].set_title("Analytical + Noise (σ=0.02)")
axes[0, 3].set_aspect('equal')
axes[0, 3].plot(0, 0, 'ro', markersize=8)

# Plot error heatmaps
im1 = axes[1, 0].contourf(X, Y, mag_err_rbf_grid, levels=20, cmap='hot')
axes[1, 0].set_title(f"RBF Magnitude Error (mean: {np.mean(mag_err_rbf):.3f})")
axes[1, 0].set_aspect('equal')
plt.colorbar(im1, ax=axes[1, 0])

im2 = axes[1, 1].contourf(X, Y, np.rad2deg(ang_err_rbf_grid), levels=20, cmap='hot')
axes[1, 1].set_title(f"RBF Angle Error (mean: {np.mean(np.rad2deg(ang_err_rbf)):.1f}°)")
axes[1, 1].set_aspect('equal')
plt.colorbar(im2, ax=axes[1, 1])

im3 = axes[1, 2].contourf(X, Y, mag_err_nn_grid, levels=20, cmap='hot')
axes[1, 2].set_title(f"NN Magnitude Error (mean: {np.mean(mag_err_nn):.3f})")
axes[1, 2].set_aspect('equal')
plt.colorbar(im3, ax=axes[1, 2])

im4 = axes[1, 3].contourf(X, Y, np.rad2deg(ang_err_nn_grid), levels=20, cmap='hot')
axes[1, 3].set_title(f"NN Angle Error (mean: {np.mean(np.rad2deg(ang_err_nn)):.1f}°)")
axes[1, 3].set_aspect('equal')
plt.colorbar(im4, ax=axes[1, 3])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/field_quality_comparison.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/field_quality_comparison.png")
plt.close()

# ============================================================================
# PART 2: ORBITAL CONTROL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PART 2: ORBITAL CONTROL AT PROBLEM RADII")
print("="*80)

def run_orbital_test(field, commanded_radius, sim_time=60, timestep=0.1):
    """Run orbital control test and return trajectory."""
    # Create cluster
    cluster = OmniCluster(FORMATION_CONFIG, field)

    # Initialize at commanded radius with random angle
    angle = np.random.uniform(0, 2*np.pi)
    start_x = commanded_radius * np.cos(angle)
    start_y = commanded_radius * np.sin(angle)
    cluster.robots[0].x = start_x
    cluster.robots[1].x = start_x + 0.1
    cluster.robots[2].x = start_x - 0.1
    cluster.robots[0].y = start_y
    cluster.robots[1].y = start_y + 0.1
    cluster.robots[2].y = start_y - 0.1

    # Run simulation
    n_steps = int(sim_time / timestep)
    trajectory_x = []
    trajectory_y = []
    radius_history = []

    for step in range(n_steps):
        # Create control primitive with desired radius
        def orbital_primitive(cluster_arg):
            return ocp.critical_point_orbiter_plane_fitting(cluster_arg, desired_radius=commanded_radius)

        # Move cluster
        cluster.move(orbital_primitive)

        # Record
        centroid = cluster.get_centroid()
        xc, yc = centroid[0], centroid[1]
        trajectory_x.append(xc)
        trajectory_y.append(yc)
        radius = np.sqrt(xc**2 + yc**2)
        radius_history.append(radius)

    return np.array(trajectory_x), np.array(trajectory_y), np.array(radius_history)

# Run orbital tests
orbital_results = {}

for radius in ORBITAL_TEST_RADII:
    print(f"\nTesting commanded radius: {radius:.1f} m")

    # Analytical
    print("  Running analytical...", end='')
    field_analytical_test = AnalyticalField(saddle1)
    x_ana, y_ana, r_ana = run_orbital_test(field_analytical_test, radius, ORBITAL_SIM_TIME)
    print(f" mean radius: {np.mean(r_ana):.3f} ± {np.std(r_ana):.3f} m")

    # RBF
    print("  Running RBF...", end='')
    field_rbf_test = RBFField(SADDLE_PREDICTOR_DIR)
    x_rbf, y_rbf, r_rbf = run_orbital_test(field_rbf_test, radius, ORBITAL_SIM_TIME)
    print(f" mean radius: {np.mean(r_rbf):.3f} ± {np.std(r_rbf):.3f} m")

    # NN
    print("  Running NN...", end='')
    field_nn_test = NNField(SADDLE_PREDICTOR_DIR)
    x_nn, y_nn, r_nn = run_orbital_test(field_nn_test, radius, ORBITAL_SIM_TIME)
    print(f" mean radius: {np.mean(r_nn):.3f} ± {np.std(r_nn):.3f} m")

    orbital_results[radius] = {
        'analytical': (x_ana, y_ana, r_ana),
        'rbf': (x_rbf, y_rbf, r_rbf),
        'nn': (x_nn, y_nn, r_nn)
    }

# Plot orbital trajectories
print("\nGenerating orbital trajectory plots...")
n_radii = len(ORBITAL_TEST_RADII)
fig, axes = plt.subplots(2, n_radii, figsize=(4*n_radii, 8))

for idx, radius in enumerate(ORBITAL_TEST_RADII):
    # Trajectory plots (top row)
    ax = axes[0, idx]
    x_ana, y_ana, _ = orbital_results[radius]['analytical']
    x_rbf, y_rbf, _ = orbital_results[radius]['rbf']
    x_nn, y_nn, _ = orbital_results[radius]['nn']

    ax.plot(x_ana, y_ana, 'g-', alpha=0.5, linewidth=1, label='Analytical')
    ax.plot(x_rbf, y_rbf, 'r-', alpha=0.5, linewidth=1, label='RBF')
    ax.plot(x_nn, y_nn, 'b-', alpha=0.5, linewidth=1, label='NN')

    # Commanded radius circle
    circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle)
    ax.plot(0, 0, 'ko', markersize=6)

    ax.set_aspect('equal')
    ax.set_title(f"r_cmd = {radius:.1f} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if idx == 0:
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Radius time series (bottom row)
    ax = axes[1, idx]
    _, _, r_ana = orbital_results[radius]['analytical']
    _, _, r_rbf = orbital_results[radius]['rbf']
    _, _, r_nn = orbital_results[radius]['nn']

    time_array = np.linspace(0, ORBITAL_SIM_TIME, len(r_ana))
    ax.plot(time_array, r_ana, 'g-', alpha=0.7, linewidth=1, label='Analytical')
    ax.plot(time_array, r_rbf, 'r-', alpha=0.7, linewidth=1, label='RBF')
    ax.plot(time_array, r_nn, 'b-', alpha=0.7, linewidth=1, label='NN')
    ax.axhline(radius, color='gray', linestyle='--', linewidth=2, label='Commanded')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Radius (m)")
    ax.set_title(f"Radius vs Time")
    if idx == 0:
        ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/orbital_control_comparison.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/orbital_control_comparison.png")
plt.close()

# ============================================================================
# PART 3: CONVERGENCE BEHAVIOR
# ============================================================================
print("\n" + "="*80)
print("PART 3: CONVERGENCE TO CRITICAL POINT")
print("="*80)

def run_convergence_test(field, start_x, start_y, sim_time=10, timestep=0.1):
    """Run convergence test from given starting position."""
    cluster = OmniCluster(FORMATION_CONFIG, field)

    # Initialize robots
    cluster.robots[0].x = start_x
    cluster.robots[1].x = start_x + 0.1
    cluster.robots[2].x = start_x - 0.1
    cluster.robots[0].y = start_y
    cluster.robots[1].y = start_y + 0.1
    cluster.robots[2].y = start_y - 0.1

    # Run simulation
    n_steps = int(sim_time / timestep)
    trajectory_x = []
    trajectory_y = []

    for step in range(n_steps):
        # Move cluster with convergence primitive
        cluster.move(ocp.critical_point_plane_fitting)

        centroid = cluster.get_centroid()
        xc, yc = centroid[0], centroid[1]
        trajectory_x.append(xc)
        trajectory_y.append(yc)

    # Final position
    final_x = trajectory_x[-1]
    final_y = trajectory_y[-1]
    final_dist = np.sqrt(final_x**2 + final_y**2)

    return np.array(trajectory_x), np.array(trajectory_y), final_dist

# Generate random starting positions
print(f"\nRunning {CONVERGENCE_TRIALS} convergence trials from random starts...")
starting_positions = []
for _ in range(CONVERGENCE_TRIALS):
    x = np.random.uniform(-0.4, 0.4)
    y = np.random.uniform(-0.4, 0.4)
    starting_positions.append((x, y))

convergence_results = {'analytical': [], 'rbf': [], 'nn': []}

for idx, (start_x, start_y) in enumerate(starting_positions):
    print(f"  Trial {idx+1}/{CONVERGENCE_TRIALS}: start=({start_x:.2f}, {start_y:.2f})")

    # Analytical
    field_ana = AnalyticalField(saddle1)
    traj_x_ana, traj_y_ana, dist_ana = run_convergence_test(field_ana, start_x, start_y)
    convergence_results['analytical'].append((traj_x_ana, traj_y_ana, dist_ana))

    # RBF
    field_rbf_conv = RBFField(SADDLE_PREDICTOR_DIR)
    traj_x_rbf, traj_y_rbf, dist_rbf = run_convergence_test(field_rbf_conv, start_x, start_y)
    convergence_results['rbf'].append((traj_x_rbf, traj_y_rbf, dist_rbf))

    # NN
    field_nn_conv = NNField(SADDLE_PREDICTOR_DIR)
    traj_x_nn, traj_y_nn, dist_nn = run_convergence_test(field_nn_conv, start_x, start_y)
    convergence_results['nn'].append((traj_x_nn, traj_y_nn, dist_nn))

# Compute statistics
dist_ana = [r[2] for r in convergence_results['analytical']]
dist_rbf = [r[2] for r in convergence_results['rbf']]
dist_nn = [r[2] for r in convergence_results['nn']]

print("\n" + "-"*80)
print("CONVERGENCE STATISTICS")
print("-"*80)
print(f"{'Field Type':<20} {'Mean Final Dist':<20} {'Success Rate (<0.05m)'}")
print("-"*80)
print(f"{'Analytical':<20} {np.mean(dist_ana):.4f} ± {np.std(dist_ana):.4f}     {100*np.sum(np.array(dist_ana) < 0.05)/len(dist_ana):.1f}%")
print(f"{'RBF':<20} {np.mean(dist_rbf):.4f} ± {np.std(dist_rbf):.4f}     {100*np.sum(np.array(dist_rbf) < 0.05)/len(dist_rbf):.1f}%")
print(f"{'NN':<20} {np.mean(dist_nn):.4f} ± {np.std(dist_nn):.4f}     {100*np.sum(np.array(dist_nn) < 0.05)/len(dist_nn):.1f}%")
print("-"*80)

# Plot convergence trajectories
print("\nGenerating convergence trajectory plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, field_type in zip(axes, ['analytical', 'rbf', 'nn']):
    for traj_x, traj_y, final_dist in convergence_results[field_type]:
        ax.plot(traj_x, traj_y, alpha=0.5, linewidth=1)
        ax.plot(traj_x[-1], traj_y[-1], 'ro', markersize=4)

    ax.plot(0, 0, 'ko', markersize=10, label='True Center')
    ax.set_aspect('equal')
    ax.set_title(f"{field_type.upper()} - Final Dist: {np.mean([r[2] for r in convergence_results[field_type]]):.4f} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/convergence_comparison.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/convergence_comparison.png")
plt.close()

# ============================================================================
# PART 4: RBF TRAINING DATA INSPECTION
# ============================================================================
print("\n" + "="*80)
print("PART 4: RBF/NN TRAINING DATA INSPECTION")
print("="*80)

# Load training data
training_data_path = os.path.join('..', SADDLE_PREDICTOR_DIR, 'rbf_training_data.pkl')
print(f"\nLoading training data from: {training_data_path}")

with open(training_data_path, 'rb') as f:
    training_data = pickle.load(f)

X_train = training_data['X_train']
print(f"Training data shape: {X_train.shape}")
print(f"Number of training points: {len(X_train)}")
print(f"X range: [{np.min(X_train[:, 0]):.3f}, {np.max(X_train[:, 0]):.3f}]")
print(f"Y range: [{np.min(X_train[:, 1]):.3f}, {np.max(X_train[:, 1]):.3f}]")

# Plot training data coverage
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2D histogram
axes[0].hist2d(X_train[:, 0], X_train[:, 1], bins=30, cmap='Blues')
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].set_title(f"Training Data Coverage ({len(X_train)} points)")
axes[0].set_aspect('equal')
axes[0].plot(0, 0, 'r*', markersize=15, label='Critical Point')
axes[0].legend()

# Scatter with radial circles
axes[1].scatter(X_train[:, 0], X_train[:, 1], alpha=0.3, s=10)
for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
    circle = plt.Circle((0, 0), r, color='red', fill=False, linestyle='--', alpha=0.5)
    axes[1].add_patch(circle)
axes[1].plot(0, 0, 'r*', markersize=15, label='Critical Point')
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")
axes[1].set_title("Training Data with Orbital Test Radii")
axes[1].set_aspect('equal')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_data_coverage.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/training_data_coverage.png")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUMMARY REPORT")
print("="*80)

report_path = f"{OUTPUT_DIR}/diagnosis_report.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SADDLE RBF/NN FAILURE DIAGNOSTIC REPORT\n")
    f.write("="*80 + "\n\n")

    f.write("PART 1: FIELD QUALITY (50×50 Grid)\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Field Type':<20} {'Mag Error':<30} {'Angle Error (deg)':<30}\n")
    f.write(f"{'RBF':<20} {np.mean(mag_err_rbf):.4f} ± {np.std(mag_err_rbf):.4f} (max: {np.max(mag_err_rbf):.4f})     {np.mean(np.rad2deg(ang_err_rbf)):.2f} ± {np.std(np.rad2deg(ang_err_rbf)):.2f} (max: {np.max(np.rad2deg(ang_err_rbf)):.2f})\n")
    f.write(f"{'NN':<20} {np.mean(mag_err_nn):.4f} ± {np.std(mag_err_nn):.4f} (max: {np.max(mag_err_nn):.4f})     {np.mean(np.rad2deg(ang_err_nn)):.2f} ± {np.std(np.rad2deg(ang_err_nn)):.2f} (max: {np.max(np.rad2deg(ang_err_nn)):.2f})\n")
    f.write(f"{'Analytical+Noise':<20} {np.mean(mag_err_noisy):.4f} ± {np.std(mag_err_noisy):.4f} (max: {np.max(mag_err_noisy):.4f})     {np.mean(np.rad2deg(ang_err_noisy)):.2f} ± {np.std(np.rad2deg(ang_err_noisy)):.2f} (max: {np.max(np.rad2deg(ang_err_noisy)):.2f})\n")
    f.write("\n")

    f.write("PART 2: ORBITAL CONTROL\n")
    f.write("-"*80 + "\n")
    for radius in ORBITAL_TEST_RADII:
        f.write(f"\nCommanded Radius: {radius:.1f} m\n")
        _, _, r_ana = orbital_results[radius]['analytical']
        _, _, r_rbf = orbital_results[radius]['rbf']
        _, _, r_nn = orbital_results[radius]['nn']
        f.write(f"  Analytical: {np.mean(r_ana):.3f} ± {np.std(r_ana):.4f} m (error: {np.mean(r_ana)-radius:+.3f} m)\n")
        f.write(f"  RBF:        {np.mean(r_rbf):.3f} ± {np.std(r_rbf):.4f} m (error: {np.mean(r_rbf)-radius:+.3f} m)\n")
        f.write(f"  NN:         {np.mean(r_nn):.3f} ± {np.std(r_nn):.4f} m (error: {np.mean(r_nn)-radius:+.3f} m)\n")
    f.write("\n")

    f.write("PART 3: CONVERGENCE ({} trials)\n".format(CONVERGENCE_TRIALS))
    f.write("-"*80 + "\n")
    f.write(f"{'Field Type':<20} {'Mean Final Dist':<25} {'Success Rate (<0.05m)'}\n")
    f.write(f"{'Analytical':<20} {np.mean(dist_ana):.4f} ± {np.std(dist_ana):.4f}          {100*np.sum(np.array(dist_ana) < 0.05)/len(dist_ana):.1f}%\n")
    f.write(f"{'RBF':<20} {np.mean(dist_rbf):.4f} ± {np.std(dist_rbf):.4f}          {100*np.sum(np.array(dist_rbf) < 0.05)/len(dist_rbf):.1f}%\n")
    f.write(f"{'NN':<20} {np.mean(dist_nn):.4f} ± {np.std(dist_nn):.4f}          {100*np.sum(np.array(dist_nn) < 0.05)/len(dist_nn):.1f}%\n")
    f.write("\n")

    f.write("PART 4: TRAINING DATA\n")
    f.write("-"*80 + "\n")
    f.write(f"Number of training points: {len(X_train)}\n")
    f.write(f"X range: [{np.min(X_train[:, 0]):.3f}, {np.max(X_train[:, 0]):.3f}]\n")
    f.write(f"Y range: [{np.min(X_train[:, 1]):.3f}, {np.max(X_train[:, 1]):.3f}]\n")
    f.write("\n")

    f.write("="*80 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("="*80 + "\n")
    f.write("1. RBF vs NN Field Quality:\n")
    if np.mean(mag_err_rbf) > np.mean(mag_err_nn):
        f.write(f"   - RBF has {(np.mean(mag_err_rbf)/np.mean(mag_err_nn)):.1f}× larger magnitude error than NN\n")
    else:
        f.write(f"   - NN has {(np.mean(mag_err_nn)/np.mean(mag_err_rbf)):.1f}× larger magnitude error than RBF\n")

    f.write("\n2. Orbital Control Performance:\n")
    worst_radius = max(ORBITAL_TEST_RADII, key=lambda r: np.std(orbital_results[r]['rbf'][2]))
    f.write(f"   - Worst RBF performance at r={worst_radius:.1f}m\n")

    f.write("\n3. Convergence Behavior:\n")
    f.write(f"   - RBF achieves {100*np.sum(np.array(dist_rbf) < 0.05)/len(dist_rbf):.1f}% success vs {100*np.sum(np.array(dist_ana) < 0.05)/len(dist_ana):.1f}% analytical\n")
    f.write(f"   - NN achieves {100*np.sum(np.array(dist_nn) < 0.05)/len(dist_nn):.1f}% success\n")

print(f"✓ Report saved: {report_path}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("Files generated:")
print("  - field_quality_comparison.png")
print("  - orbital_control_comparison.png")
print("  - convergence_comparison.png")
print("  - training_data_coverage.png")
print("  - diagnosis_report.txt")
