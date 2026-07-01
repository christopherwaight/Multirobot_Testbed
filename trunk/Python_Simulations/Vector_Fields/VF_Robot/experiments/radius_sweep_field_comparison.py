"""
Radius Sweep Field Mode Comparison Experiment

Tests four control primitives across multiple radii (0.1 to 0.6) using three field modes:
1. Analytical (ground truth)
2. RBF (RBF interpolator only)
3. NN (Neural network only)

Control Primitives Tested:
1. critical_point_orbiter_plane_fitting (3-robot orbiter)
2. center_orbiter_quad_planar (4-robot orbiter, planar estimation)
3. center_orbiter_quad_advanced (4-robot orbiter, with orientation control)
4. vector_sum (3-robot, simple field following - baseline)

For each configuration, tracks:
- Average distance from origin
- Average distance from estimated critical point
- Estimated center position (x, y) over time

Results are displayed in terminal tables and saved to CSV files.
"""

import sys
import os
import csv
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import robot clusters and field types
from src.robot.omni_cluster import OmniCluster
from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField, NNField, RBFField

# Import kinematics
from src.control.kinematics import inverse_kinematics
from src.control.quad_kinematics import inverse_kinematics_quad

# Import control primitives
import src.control.primitives as ocp
import src.control.quad_primitives as qcp

# Import environments
from src.fields.environments.Sink import sink1
from src.fields.environments.Source import source1
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Sinking_Vortex import sinking_vortex1
from src.fields.environments.Spewing_Vortex import spewing_vortex1
from src.fields.environments.Saddle import saddle1


# ============================================================================
# CONFIGURATION
# ============================================================================

# Radius values to sweep
RADII = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Field modes to test
FIELD_MODES = ["analytical", "nn", "rbf"]

# Simulation parameters
SIM_TIME = 300  # Number of simulation steps

# Testing mode - set to True to run quick test with limited parameters
TESTING_MODE = False
if TESTING_MODE:
    RADII = [0.2]  # Just one radius
    SIM_TIME = 50  # Shorter simulation
    FIELD_MODES = ["analytical"]  # Just one mode

TIMESTEP = 0.1  # Time step in seconds

# Initial position: (0, desired_radius)
INITIAL_X = 0.0

# Environments to test with their predictor directories
ENVIRONMENTS = [
    (vortex1, "Vortex 1", "vortex1", "vortex_predictors"),
    (saddle1, "Saddle 1", "saddle1", "saddle_predictors"),
]

if TESTING_MODE:
    ENVIRONMENTS = [(vortex1, "Vortex 1", "vortex1", "vortex_predictors")]  # Just one environment

# Control primitive configurations
# Format: (name, function, num_robots, formation_config)
PRIMITIVES = [
    ("3R_orbiter_plane", ocp.critical_point_orbiter_plane_fitting, 3,
     "config/formations/equilateral_radius_sweep.yaml"),
    ("4R_orbiter_planar", qcp.center_orbiter_quad_planar, 4,
     "config/formations/quad_square_default.yaml"),
    ("4R_orbiter_advanced", qcp.center_orbiter_quad_advanced, 4,
     "config/formations/quad_rectangle.yaml"),
    ("3R_vector_sum", ocp.vector_sum, 3,
     "config/formations/equilateral_radius_sweep.yaml"),
]

if TESTING_MODE:
    PRIMITIVES = [PRIMITIVES[0]]  # Just first primitive

# Output directory
OUTPUT_DIR = "experiments/radius_sweep_results"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_field(field_mode, environment_func, predictor_dir):
    """
    Create field object based on mode.

    Args:
        field_mode: "analytical", "rbf", or "nn"
        environment_func: Environment function (for analytical)
        predictor_dir: Predictor directory (for ML models)

    Returns:
        Field object
    """
    if field_mode == "analytical":
        return AnalyticalField(environment_func)
    elif field_mode == "rbf":
        return RBFField(predictor_dir=predictor_dir)
    elif field_mode == "nn":
        return NNField(predictor_dir=predictor_dir)
    else:
        raise ValueError(f"Unknown field mode: {field_mode}")


def estimate_critical_point_3robot(cluster):
    """
    Estimate critical point for 3-robot cluster using plane fitting.

    Returns:
        np.array: [x, y] estimated center, or None if singular
    """
    try:
        x1, y1, x2, y2, x3, y3 = cluster.get_robot_positions()
        readings = cluster.sample_field_at_robots()
        u1, v1 = readings[0]
        u2, v2 = readings[1]
        u3, v3 = readings[2]

        # Matrix A for plane fitting
        A = np.array([
            [x1, y1, 1],
            [x2, y2, 1],
            [x3, y3, 1]
        ])

        # Solve for plane coefficients
        b_u = np.array([u1, u2, u3])
        b_v = np.array([v1, v2, v3])

        abc = np.linalg.solve(A, b_u)
        def_ = np.linalg.solve(A, b_v)

        a, b, c = abc
        d, e, f = def_

        # Form Jacobian and solve for critical point
        J = np.array([[a, b], [d, e]])
        rhs = np.array([-c, -f])
        critical_point = np.linalg.solve(J, rhs)

        return critical_point
    except (np.linalg.LinAlgError, ValueError):
        return None


def estimate_critical_point_4robot(cluster):
    """
    Estimate critical point for 4-robot cluster using dual Jacobian.

    Returns:
        np.array: [x, y] estimated center, or None if estimation fails
    """
    try:
        center_estimate, _ = qcp.estimate_center_and_radius(cluster)
        return center_estimate
    except Exception:
        return None


def run_single_simulation(primitive_func, num_robots, formation_config,
                         field_obj, desired_radius, primitive_name):
    """
    Run a single simulation and collect metrics.

    Args:
        primitive_func: Control primitive function
        num_robots: 3 or 4
        formation_config: Path to formation YAML
        field_obj: Field object (Analytical, NN, or RBF)
        desired_radius: Desired orbital radius
        primitive_name: Name of primitive (to check if it needs radius param)

    Returns:
        dict: Metrics including distances and estimated centers
    """
    # Create cluster based on robot count
    if num_robots == 3:
        cluster = OmniCluster(formation_config, field_obj)
        estimate_func = estimate_critical_point_3robot

        # Set initial position for 3-robot cluster
        initial_y = desired_radius
        theta_c_init = np.pi  # Initial orientation
        x1, y1, x2, y2, x3, y3 = inverse_kinematics(
            INITIAL_X, initial_y, theta_c_init,
            cluster.desired_p, cluster.desired_beta, cluster.desired_q
        )
        # Update robot positions
        cluster.robots[0].position = np.array([x1, y1])
        cluster.robots[1].position = np.array([x2, y2])
        cluster.robots[2].position = np.array([x3, y3])
    else:
        cluster = QuadCluster(formation_config, field_obj)
        estimate_func = estimate_critical_point_4robot

        # Set initial position for 4-robot cluster
        initial_y = desired_radius
        theta_c_init = np.pi  # Initial orientation
        x1, y1, x2, y2, x3, y3, x4, y4 = inverse_kinematics_quad(
            INITIAL_X, initial_y, theta_c_init,
            cluster.desired_d1, cluster.desired_d2,
            cluster.desired_r1, cluster.desired_r2, cluster.desired_phi
        )
        # Update robot positions
        cluster.robots[0].position = np.array([x1, y1])
        cluster.robots[1].position = np.array([x2, y2])
        cluster.robots[2].position = np.array([x3, y3])
        cluster.robots[3].position = np.array([x4, y4])

    # Storage for metrics
    centroid_positions = []
    estimated_centers = []
    distances_to_origin = []
    distances_to_estimated_center = []

    # Create a wrapper function that passes desired_radius to the primitive
    # Vector sum doesn't take desired_radius parameter
    if "vector_sum" in primitive_name:
        def control_primitive_with_radius(cluster_arg):
            return primitive_func(cluster_arg)
    else:
        def control_primitive_with_radius(cluster_arg):
            return primitive_func(cluster_arg, desired_radius=desired_radius)

    # Run simulation
    for step in range(SIM_TIME):
        # Get current centroid (before moving)
        x_c, y_c = cluster.get_centroid()
        centroid_positions.append([x_c, y_c])

        # Calculate distance to origin
        dist_origin = np.sqrt(x_c**2 + y_c**2)
        distances_to_origin.append(dist_origin)

        # Estimate critical point
        estimated_center = estimate_func(cluster)

        if estimated_center is not None:
            estimated_centers.append(estimated_center.copy())
            # Distance from centroid to estimated center
            dist_to_center = np.linalg.norm([x_c - estimated_center[0],
                                            y_c - estimated_center[1]])
            distances_to_estimated_center.append(dist_to_center)
        else:
            estimated_centers.append([np.nan, np.nan])
            distances_to_estimated_center.append(np.nan)

        # Move cluster using the cluster's move() method
        cluster.move(control_primitive_with_radius)

    # Convert to numpy arrays
    centroid_positions = np.array(centroid_positions)
    estimated_centers = np.array(estimated_centers)
    distances_to_origin = np.array(distances_to_origin)
    distances_to_estimated_center = np.array(distances_to_estimated_center)

    # Calculate metrics (ignoring NaN values)
    metrics = {
        'avg_dist_origin': np.nanmean(distances_to_origin),
        'std_dist_origin': np.nanstd(distances_to_origin),
        'avg_dist_to_est_center': np.nanmean(distances_to_estimated_center),
        'std_dist_to_est_center': np.nanstd(distances_to_estimated_center),
        'est_center_x_mean': np.nanmean(estimated_centers[:, 0]),
        'est_center_x_std': np.nanstd(estimated_centers[:, 0]),
        'est_center_y_mean': np.nanmean(estimated_centers[:, 1]),
        'est_center_y_std': np.nanstd(estimated_centers[:, 1]),
    }

    return metrics


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_table_header(primitive_name, field_mode):
    """Print table header for results."""
    print(f"\n{'='*110}")
    print(f"PRIMITIVE: {primitive_name} | FIELD MODE: {field_mode.upper()}")
    print(f"{'='*110}")
    print(f"{'Radius':<8} {'Avg Dist Origin':<20} {'Avg Dist Est Center':<25} "
          f"{'Est Center X':<20} {'Est Center Y':<20}")
    print(f"{'':<8} {'(mean ± std)':<20} {'(mean ± std)':<25} "
          f"{'(mean ± std)':<20} {'(mean ± std)':<20}")
    print(f"{'-'*110}")


def print_table_row(radius, metrics):
    """Print a single row of results."""
    print(f"{radius:<8.2f} "
          f"{metrics['avg_dist_origin']:>8.4f} ± {metrics['std_dist_origin']:<8.4f} "
          f"{metrics['avg_dist_to_est_center']:>10.4f} ± {metrics['std_dist_to_est_center']:<10.4f} "
          f"{metrics['est_center_x_mean']:>8.4f} ± {metrics['est_center_x_std']:<8.4f} "
          f"{metrics['est_center_y_mean']:>8.4f} ± {metrics['est_center_y_std']:<8.4f}")


def print_environment_header(env_name):
    """Print environment header."""
    print(f"\n\n{'#'*110}")
    print(f"# ENVIRONMENT: {env_name}")
    print(f"{'#'*110}")


# ============================================================================
# CSV EXPORT
# ============================================================================

def save_results_to_csv(all_results, output_dir):
    """
    Save all results to CSV files.

    Args:
        all_results: List of result dictionaries
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV filename
    csv_filename = os.path.join(output_dir, f"radius_sweep_field_comparison_{timestamp}.csv")

    # Write to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'environment', 'primitive', 'field_mode', 'radius',
            'avg_dist_origin', 'std_dist_origin',
            'avg_dist_to_est_center', 'std_dist_to_est_center',
            'est_center_x_mean', 'est_center_x_std',
            'est_center_y_mean', 'est_center_y_std'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

    print(f"\n\nResults saved to: {csv_filename}")
    return csv_filename


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Run the complete radius sweep field comparison experiment."""
    print("="*110)
    print("RADIUS SWEEP FIELD MODE COMPARISON EXPERIMENT")
    print("="*110)
    print(f"Simulation time: {SIM_TIME} steps ({SIM_TIME * TIMESTEP:.1f} seconds)")
    print(f"Radii tested: {RADII}")
    print(f"Field modes: {FIELD_MODES}")
    print(f"Environments: {len(ENVIRONMENTS)}")
    print(f"Primitives: {len(PRIMITIVES)}")
    print(f"Total runs: {len(ENVIRONMENTS) * len(PRIMITIVES) * len(RADII) * len(FIELD_MODES)}")

    # Storage for all results
    all_results = []

    # Loop over environments
    for env_func, env_display_name, env_key, predictor_dir in ENVIRONMENTS:
        print_environment_header(env_display_name)

        # Loop over primitives
        for prim_name, prim_func, num_robots, formation_config in PRIMITIVES:

            # Loop over field modes
            for field_mode in FIELD_MODES:
                print_table_header(prim_name, field_mode)

                # Loop over radii
                for radius in RADII:
                    print(f"Running: {env_display_name} | {prim_name} | {field_mode} | r={radius:.2f}...",
                          end=' ', flush=True)

                    try:
                        # Create field object for this mode
                        field_obj = create_field(field_mode, env_func, predictor_dir)

                        # Run simulation
                        metrics = run_single_simulation(
                            prim_func, num_robots, formation_config,
                            field_obj, radius, prim_name
                        )

                        # Print result row
                        print("Done")
                        print_table_row(radius, metrics)

                        # Store results
                        result_dict = {
                            'environment': env_key,
                            'primitive': prim_name,
                            'field_mode': field_mode,
                            'radius': radius,
                            **metrics
                        }
                        all_results.append(result_dict)

                    except Exception as e:
                        print(f"FAILED: {e}")
                        # Store failed result
                        result_dict = {
                            'environment': env_key,
                            'primitive': prim_name,
                            'field_mode': field_mode,
                            'radius': radius,
                            'avg_dist_origin': np.nan,
                            'std_dist_origin': np.nan,
                            'avg_dist_to_est_center': np.nan,
                            'std_dist_to_est_center': np.nan,
                            'est_center_x_mean': np.nan,
                            'est_center_x_std': np.nan,
                            'est_center_y_mean': np.nan,
                            'est_center_y_std': np.nan,
                        }
                        all_results.append(result_dict)

    # Save results to CSV
    csv_file = save_results_to_csv(all_results, OUTPUT_DIR)

    print("\n" + "="*110)
    print("EXPERIMENT COMPLETE")
    print("="*110)
    print(f"Total runs completed: {len(all_results)}")
    print(f"Results saved to: {csv_file}")


if __name__ == "__main__":
    main()
