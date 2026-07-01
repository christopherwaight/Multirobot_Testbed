"""
sim2real_with_center_tracking.py - Multi-Field Comparison with Center Estimate Tracking

This version properly tracks the robot's center estimates throughout the simulation
to calculate meaningful std_x_est and std_y_est values.

Compares 4 field representations:
1. Analytical (ground truth)
2. Neural Network (learned)
3. RBF Interpolator (interpolated)
4. Real Robot (MATLAB data)

For each, we track both the trajectory AND the robot's estimates of where the center is.
"""
import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from typing import Dict, List, Tuple, Optional

# Architecture imports
from src.robot.omni_cluster import OmniCluster
from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField, NNField, RBFField
import src.control.primitives as ocp
import src.control.quad_primitives as qcp
from src.fields.environments.Vortex import vortex1

# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation parameters
SIM_TIME = 600  # timesteps (60 seconds at 0.1s timestep)
TIMESTEP = 0.1  # seconds
TRUE_CENTER = (0.0, 0.0)  # The TRUE center is always at origin

# Environment
ENVIRONMENT = "vortex1"
ENVIRONMENT_FUNC = vortex1
PREDICTOR_DIR = "vortex_predictors"  # ML models directory

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(script_dir, "../real_robot_data/real_robot_trajectories")
OUTPUT_DIR = os.path.join(script_dir, "sim2real_center_tracking_results")
project_root = os.path.dirname(script_dir)
FORMATION_3ROBOT = os.path.join(project_root, "config/formations/equilateral_default.yaml")
FORMATION_4ROBOT_SQUARE = os.path.join(project_root, "config/formations/quad_square.yaml")
FORMATION_4ROBOT_ADVANCED = os.path.join(project_root, "config/formations/quad_default.yaml")

# Field modes to compare
FIELD_MODES = ['analytical', 'nn', 'rbf']

# Plots to save
SAVE_PLOTS = True
PLOTS_TO_SAVE = ['orbit010', 'orbit040', 'orbit101', 'orbit140', 'orbit201', 'orbit240']

# ============================================================================
# CENTER ESTIMATION EXTRACTION
# ============================================================================

def extract_center_estimate_3robot(cluster, control_func, desired_radius):
    """
    Extract the center estimate from a 3-robot control primitive.

    This reimplements the center estimation logic from the control primitives
    to extract the estimated center position.
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3 = cluster.get_robot_positions()

    # Get field readings
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

    b_u = np.array([u1, u2, u3])
    b_v = np.array([v1, v2, v3])

    try:
        # Solve for plane coefficients
        abc = np.linalg.solve(A, b_u)
        def_ = np.linalg.solve(A, b_v)

        a, b, c = abc
        d, e, f = def_

        # Form Jacobian
        J = np.array([[a, b], [d, e]])

        # Solve for critical point (this is the center estimate!)
        rhs = np.array([-c, -f])
        critical_point = np.linalg.solve(J, rhs)

        return critical_point[0], critical_point[1]  # x_est, y_est

    except np.linalg.LinAlgError:
        # If singular, return centroid as estimate
        centroid = cluster.get_centroid()
        return centroid[0], centroid[1]


def extract_center_estimate_4robot_planar(cluster, control_func, desired_radius):
    """
    Extract the center estimate from a 4-robot control primitive using 4-point planar fitting.

    Uses least squares plane fitting through all 4 robots simultaneously.
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

    # Get field readings
    readings = cluster.sample_field_at_robots()
    u1, v1 = readings[0]
    u2, v2 = readings[1]
    u3, v3 = readings[2]
    u4, v4 = readings[3]

    # Build matrix A for overdetermined system (4 equations, 3 unknowns)
    A = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1],
        [x4, y4, 1]
    ])

    # RHS for u and v components
    b_u = np.array([u1, u2, u3, u4])
    b_v = np.array([v1, v2, v3, v4])

    try:
        # Solve for plane coefficients using least squares
        abc, _, _, _ = np.linalg.lstsq(A, b_u, rcond=None)
        def_, _, _, _ = np.linalg.lstsq(A, b_v, rcond=None)

        a, b, c = abc
        d, e, f = def_

        # Form the Jacobian matrix
        J = np.array([[a, b], [d, e]])

        # Solve for critical point where u=0 and v=0
        rhs = np.array([-c, -f])
        critical_point = np.linalg.solve(J, rhs)

        return critical_point[0], critical_point[1]  # x_est, y_est

    except (np.linalg.LinAlgError, ValueError):
        # If singular or lstsq fails, return centroid as estimate
        centroid = cluster.get_centroid()
        return centroid[0], centroid[1]


def extract_center_estimate_4robot(cluster, control_func, desired_radius):
    """
    Extract the center estimate from a 4-robot control primitive.

    Uses dual Jacobian estimation from overlapping triangles.
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

    # Get field readings
    readings = cluster.sample_field_at_robots()

    # Top triangle (1,2,3)
    positions_top = [(x1, y1), (x2, y2), (x3, y3)]
    readings_top = readings[:3]

    # Bottom triangle (2,3,4)
    positions_bottom = [(x2, y2), (x3, y3), (x4, y4)]
    readings_bottom = readings[1:4]

    # Calculate Jacobian for each triangle
    def calc_jacobian(positions, readings):
        (x1, y1), (x2, y2), (x3, y3) = positions
        (u1, v1), (u2, v2), (u3, v3) = readings

        A = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
        b_u = np.array([u1, u2, u3])
        b_v = np.array([v1, v2, v3])

        try:
            abc = np.linalg.solve(A, b_u)
            def_ = np.linalg.solve(A, b_v)
            a, b, c = abc
            d, e, f = def_
            J = np.array([[a, b], [d, e]])
            rhs = np.array([-c, -f])
            center = np.linalg.solve(J, rhs)
            return center
        except:
            return None

    center_top = calc_jacobian(positions_top, readings_top)
    center_bottom = calc_jacobian(positions_bottom, readings_bottom)

    # Average the two estimates if both valid
    if center_top is not None and center_bottom is not None:
        x_est = (center_top[0] + center_bottom[0]) / 2
        y_est = (center_top[1] + center_bottom[1]) / 2
    elif center_top is not None:
        x_est, y_est = center_top
    elif center_bottom is not None:
        x_est, y_est = center_bottom
    else:
        # Fallback to centroid
        centroid = cluster.get_centroid()
        x_est, y_est = centroid

    return x_est, y_est


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_configuration_details(orbit_name: str) -> Dict:
    """Parse orbit configuration from filename."""
    orbit_num = int(orbit_name.replace('orbit', ''))

    if orbit_num < 100:
        # 0XX series: 3-robot equilateral
        return {
            'series': '0XX',
            'robot_type': '3-robot',
            'num_robots': 3,
            'radius': orbit_num / 100.0,
            'formation_config': FORMATION_3ROBOT,
            'control_primitive': 'critical_point_orbiter_plane_fitting',
            'control_func': ocp.critical_point_orbiter_plane_fitting,
            'cluster_class': OmniCluster,
            'extract_center': extract_center_estimate_3robot
        }
    elif orbit_num < 200:
        # 1XX series: 4-robot square (uses 4-point planar fitting)
        return {
            'series': '1XX',
            'robot_type': '4-robot-square',
            'num_robots': 4,
            'radius': (orbit_num - 100) / 100.0,
            'formation_config': FORMATION_4ROBOT_SQUARE,
            'control_primitive': 'center_orbiter_quad_planar',
            'control_func': qcp.center_orbiter_quad_planar,
            'cluster_class': QuadCluster,
            'extract_center': extract_center_estimate_4robot_planar  # Uses 4-point planar fitting
        }
    else:
        # 2XX series: 4-robot advanced (uses dual Jacobian from two triangles)
        return {
            'series': '2XX',
            'robot_type': '4-robot-advanced',
            'num_robots': 4,
            'radius': (orbit_num - 200) / 100.0,
            'formation_config': FORMATION_4ROBOT_ADVANCED,
            'control_primitive': 'center_orbiter_quad_advanced',
            'control_func': qcp.center_orbiter_quad_advanced,
            'cluster_class': QuadCluster,
            'extract_center': extract_center_estimate_4robot  # Uses dual Jacobian (two triangles)
        }


def create_field(mode: str, predictor_dir: str = PREDICTOR_DIR):
    """Create field object based on mode."""
    if mode == "analytical":
        return AnalyticalField(ENVIRONMENT_FUNC)
    elif mode == "nn":
        try:
            return NNField(predictor_dir)
        except Exception as e:
            print(f"  Warning: NN model not found, using analytical. Error: {e}")
            return AnalyticalField(ENVIRONMENT_FUNC)
    elif mode == "rbf":
        try:
            return RBFField(predictor_dir)
        except Exception as e:
            print(f"  Warning: RBF model not found, using analytical. Error: {e}")
            return AnalyticalField(ENVIRONMENT_FUNC)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_real_robot_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load real robot trajectory from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Real robot data not found: {filepath}")

    data = pd.read_csv(filepath)
    time = data['time_s'].values
    x = data['x_m'].values
    y = data['y_m'].values
    trajectory = np.column_stack([x, y])

    return trajectory, time


def run_simulation_with_center_tracking(config: Dict, field_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run simulation and track both trajectory and center estimates.

    Returns:
        trajectory: Nx2 array of centroid positions
        center_estimates: Nx2 array of estimated center positions
    """
    # Create field
    field = create_field(field_mode)

    # Create cluster (suppress print)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        cluster = config['cluster_class'](config['formation_config'], field)

    # Start at (0, desired_radius)
    cluster.reset(x_c=0.0, y_c=config['radius'])

    # Create control primitive wrapper
    desired_radius = config['radius']
    def control_primitive(cluster):
        return config['control_func'](cluster, desired_radius=desired_radius)

    # Storage
    trajectory = []
    center_estimates = []

    # Run simulation
    for step in range(SIM_TIME):
        # Get cluster centroid
        x_c, y_c = cluster.get_centroid()
        trajectory.append([x_c, y_c])

        # Extract center estimate BEFORE moving
        # This is what the robot thinks the center is right now
        x_est, y_est = config['extract_center'](cluster, config['control_func'], desired_radius)
        center_estimates.append([x_est, y_est])

        # Move cluster
        cluster.move(control_primitive)

    return np.array(trajectory), np.array(center_estimates)


def calculate_statistics_with_estimates(trajectory: np.ndarray,
                                       center_estimates: np.ndarray,
                                       desired_radius: float,
                                       stable_start: int = 100) -> Dict:
    """
    Calculate statistics including center estimate variation.

    Args:
        trajectory: Nx2 array of positions
        center_estimates: Nx2 array of center estimates at each timestep
        desired_radius: Target radius
        stable_start: Timestep to start stable statistics

    Returns:
        Dictionary of statistics
    """
    # Distance from TRUE center (origin)
    distances_from_true = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)
    stable_distances = distances_from_true[stable_start:] if len(distances_from_true) > stable_start else distances_from_true

    avg_radius = np.mean(stable_distances)
    std_radius = np.std(stable_distances)
    radius_error = avg_radius - desired_radius

    # Center estimate statistics (NOW WE HAVE REAL ESTIMATES!)
    if len(center_estimates) > stable_start:
        stable_estimates = center_estimates[stable_start:]
    else:
        stable_estimates = center_estimates

    # Average estimated center
    avg_x_est = np.mean(stable_estimates[:, 0])
    avg_y_est = np.mean(stable_estimates[:, 1])

    # Standard deviation of center estimates (THIS SHOULD BE NON-ZERO!)
    std_x_est = np.std(stable_estimates[:, 0])
    std_y_est = np.std(stable_estimates[:, 1])

    # Error in center estimation
    center_errors = np.sqrt(stable_estimates[:, 0]**2 + stable_estimates[:, 1]**2)
    avg_center_error = np.mean(center_errors)
    std_center_error = np.std(center_errors)

    # RMSE from true center
    stable_trajectory = trajectory[stable_start:] if len(trajectory) > stable_start else trajectory
    rmse = np.sqrt(np.mean(stable_trajectory[:, 0]**2 + stable_trajectory[:, 1]**2))

    return {
        'avg_radius': avg_radius,
        'std_radius': std_radius,
        'radius_error': radius_error,
        'avg_x_est': avg_x_est,
        'avg_y_est': avg_y_est,
        'std_x_est': std_x_est,  # This should now be non-zero!
        'std_y_est': std_y_est,  # This should now be non-zero!
        'avg_center_error': avg_center_error,
        'std_center_error': std_center_error,
        'rmse': rmse
    }


def analyze_real_trajectory(trajectory: np.ndarray, desired_radius: float,
                           stable_start: int = 100) -> Dict:
    """
    Analyze real robot trajectory (no center estimates available).

    For real data, we estimate the center from the trajectory shape.
    """
    # Distance from TRUE center
    distances_from_true = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)
    stable_distances = distances_from_true[stable_start:] if len(distances_from_true) > stable_start else distances_from_true

    avg_radius = np.mean(stable_distances)
    std_radius = np.std(stable_distances)
    radius_error = avg_radius - desired_radius

    # Estimate center from stable trajectory shape
    stable_trajectory = trajectory[stable_start:] if len(trajectory) > stable_start else trajectory

    # Use sliding window to estimate how the "apparent center" varies
    window_size = 50  # 5 seconds
    if len(stable_trajectory) > window_size * 2:
        center_estimates = []
        for i in range(0, len(stable_trajectory) - window_size, 10):  # Step by 10 for efficiency
            window = stable_trajectory[i:i+window_size]
            cx = np.mean(window[:, 0])
            cy = np.mean(window[:, 1])
            center_estimates.append([cx, cy])
        center_estimates = np.array(center_estimates)

        avg_x_est = np.mean(center_estimates[:, 0])
        avg_y_est = np.mean(center_estimates[:, 1])
        std_x_est = np.std(center_estimates[:, 0])
        std_y_est = np.std(center_estimates[:, 1])

        # Calculate std of center errors (distance from true center at origin)
        center_errors = np.sqrt(center_estimates[:, 0]**2 + center_estimates[:, 1]**2)
        avg_center_error = np.mean(center_errors)
        std_center_error = np.std(center_errors)
    else:
        avg_x_est = np.mean(stable_trajectory[:, 0])
        avg_y_est = np.mean(stable_trajectory[:, 1])
        std_x_est = 0.0
        std_y_est = 0.0
        avg_center_error = np.sqrt(avg_x_est**2 + avg_y_est**2)
        std_center_error = 0.0

    rmse = np.sqrt(np.mean(stable_trajectory[:, 0]**2 + stable_trajectory[:, 1]**2))

    return {
        'avg_radius': avg_radius,
        'std_radius': std_radius,
        'radius_error': radius_error,
        'avg_x_est': avg_x_est,
        'avg_y_est': avg_y_est,
        'std_x_est': std_x_est,
        'std_y_est': std_y_est,
        'avg_center_error': avg_center_error,
        'std_center_error': std_center_error,  # Now calculated from sliding window estimates!
        'rmse': rmse
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("SIM-TO-REAL COMPARISON WITH CENTER ESTIMATE TRACKING")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Environment: {ENVIRONMENT}")
    print(f"  Field Modes: Analytical, Neural Network, RBF")
    print(f"  True Center: {TRUE_CENTER}")
    print(f"  Tracking: Trajectory AND center estimates")
    print("=" * 70)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find trajectory files
    trajectory_files = glob(os.path.join(REAL_DATA_DIR, "orbit*_trajectory.csv"))

    if not trajectory_files:
        print(f"ERROR: No trajectory files found in {REAL_DATA_DIR}")
        return

    trajectory_files.sort()
    print(f"Found {len(trajectory_files)} trajectory files to process")
    print()

    # Results storage
    all_results = []

    # Process each configuration
    for file_idx, trajectory_file in enumerate(trajectory_files):
        # Extract orbit name
        basename = os.path.basename(trajectory_file)
        orbit_name = basename.replace('_trajectory.csv', '')

        print(f"\n[{file_idx+1}/{len(trajectory_files)}] Processing {orbit_name}")
        print("-" * 50)

        try:
            # Get configuration
            config = get_configuration_details(orbit_name)

            print(f"  Type: {config['robot_type']}")
            print(f"  Target Radius: {config['radius']:.3f} m")
            print(f"  Control: {config['control_primitive']}")

            # Load real robot data
            real_trajectory, real_time = load_real_robot_data(trajectory_file)
            print(f"  Real data: {len(real_trajectory)} points")

            # Results for this configuration
            result = {
                'config_name': orbit_name,
                'robot_type': config['robot_type'],
                'series': config['series'],
                'desired_radius': config['radius']
            }

            # Run simulations with different field modes
            print(f"  Running simulations with center tracking:")

            for mode in FIELD_MODES:
                try:
                    print(f"    {mode.upper()}...", end='', flush=True)
                    trajectory, center_estimates = run_simulation_with_center_tracking(config, mode)

                    # Calculate statistics WITH center estimates
                    stats = calculate_statistics_with_estimates(
                        trajectory, center_estimates, config['radius']
                    )

                    # Add to results
                    for key, value in stats.items():
                        result[f'{mode}_{key}'] = value

                    print(f" ✓ (std_x={stats['std_x_est']:.4f}, std_y={stats['std_y_est']:.4f})")

                except Exception as e:
                    print(f" ✗ Error: {e}")

            # Analyze real robot data
            real_stats = analyze_real_trajectory(real_trajectory, config['radius'])
            for key, value in real_stats.items():
                result[f'real_{key}'] = value

            # Add comparison metrics
            if 'analytical_avg_radius' in result:
                for mode in ['nn', 'rbf']:
                    if f'{mode}_avg_radius' in result:
                        result[f'{mode}_vs_analytical'] = result[f'{mode}_avg_radius'] - result['analytical_avg_radius']

                for mode in ['analytical', 'nn', 'rbf']:
                    if f'{mode}_avg_radius' in result:
                        result[f'{mode}_vs_real'] = result[f'{mode}_avg_radius'] - result['real_avg_radius']

            # Store result
            all_results.append(result)

            # Print summary
            print(f"  Summary of std_x_est and std_y_est (should be non-zero!):")
            for mode in ['analytical', 'nn', 'rbf', 'real']:
                std_x = result.get(f'{mode}_std_x_est', np.nan)
                std_y = result.get(f'{mode}_std_y_est', np.nan)
                if not np.isnan(std_x):
                    print(f"    {mode:12s}: std_x={std_x:.4f}, std_y={std_y:.4f}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "sim2real_center_tracking_results.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Results saved to: {csv_path}")

    # Print analysis
    print("\n" + "=" * 70)
    print("CENTER ESTIMATE VARIATION ANALYSIS")
    print("=" * 70)

    print("\nAverage std_x_est across all configurations:")
    for mode in ['analytical', 'nn', 'rbf']:
        col = f'{mode}_std_x_est'
        if col in results_df.columns:
            mean_std = results_df[col].mean()
            print(f"  {mode:12s}: {mean_std:.4f} m")

    print("\nAverage std_y_est across all configurations:")
    for mode in ['analytical', 'nn', 'rbf']:
        col = f'{mode}_std_y_est'
        if col in results_df.columns:
            mean_std = results_df[col].mean()
            print(f"  {mode:12s}: {mean_std:.4f} m")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("• std_x_est and std_y_est now show ACTUAL variation in center estimates")
    print("• These values come from the robot's real-time Jacobian calculations")
    print("• Higher std means the robot's center estimate is less stable")
    print("• NN and RBF fields may have different estimation stability than analytical")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()