"""
Detailed diagnosis: Instrument the estimation to catch the exact moment of NaN.
"""

import numpy as np
import sys
import io
import contextlib

sys.path.append('.')

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Saddle import saddle1
import src.control.quad_primitives as qcp


def instrumented_estimate(cluster):
    """
    Instrumented version of estimate_center_and_radius that shows details.
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

    # Calculate triangle centers
    top_triangle_center = np.array([(x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0])
    bottom_triangle_center = np.array([(x4 + x2 + x3) / 3.0, (y4 + y2 + y3) / 3.0])

    # Get center directions from both triangles
    dir1 = qcp.calculate_center_direction_top(cluster)
    dir2 = qcp.calculate_center_direction_bottom(cluster)

    # If either direction is None, fail
    if dir1 is None:
        return None, None, "Top triangle: singular matrix"
    if dir2 is None:
        return None, None, "Bottom triangle: singular matrix"

    # Current cluster centroid
    p0 = cluster.get_centroid()

    # Find intersection
    center_estimate = qcp.find_line_intersection(top_triangle_center, dir1, bottom_triangle_center, dir2)

    if center_estimate is None:
        # Fallback: use averaged direction
        combined_dir = (dir1 + dir2) / 2
        norm = np.linalg.norm(combined_dir)

        # THE CRITICAL CHECK
        if norm < 1e-10:
            angle_deg = np.arccos(np.clip(np.dot(dir1, dir2), -1, 1)) * 180 / np.pi
            return None, None, f"Opposite directions (angle={angle_deg:.1f}°) → combined_dir ≈ 0 → NaN"

        # This line would cause NaN if norm is 0
        with np.errstate(invalid='ignore'):
            combined_dir = combined_dir / norm

        if np.isnan(combined_dir[0]) or np.isnan(combined_dir[1]):
            return None, None, f"NaN after normalization (norm was {norm})"

        # Default distance
        default_distance = 0.3
        center_estimate = p0 + default_distance * combined_dir
        estimated_distance = default_distance
    else:
        # Calculate distance from centroid to estimated center
        estimated_distance = np.linalg.norm(center_estimate - p0)

        # Sanity check - cap at reasonable distance
        max_reasonable_distance = 1.0
        if estimated_distance > max_reasonable_distance:
            unit_vector = (center_estimate - p0) / estimated_distance
            center_estimate = p0 + max_reasonable_distance * unit_vector
            estimated_distance = max_reasonable_distance

    return center_estimate, estimated_distance, "Success"


print("="*80)
print("Detailed NaN Diagnosis")
print("="*80)
print()

np.random.seed(42)

failure_count = 0
cause_histogram = {}

for trial in range(100):
    start_x = np.random.uniform(-0.5, 0.5)
    start_y = np.random.uniform(-0.5, 0.5)

    field = AnalyticalField(saddle1)
    with contextlib.redirect_stdout(io.StringIO()):
        cluster = QuadCluster('config/formations/quad_square_default.yaml', field)
    cluster.reset(x_c=start_x, y_c=start_y)

    # Run simulation
    for step in range(100):
        center_est, radius, cause = instrumented_estimate(cluster)

        if center_est is None:
            failure_count += 1
            cause_histogram[cause] = cause_histogram.get(cause, 0) + 1

            if failure_count <= 3:  # Print first 3 failures in detail
                print(f"\n{'='*80}")
                print(f"FAILURE #{failure_count}: Trial {trial+1}, Step {step}")
                print(f"Starting position: ({start_x:.3f}, {start_y:.3f})")
                print(f"Cause: {cause}")
                print(f"{'='*80}")

                # Get more details
                x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
                centroid = cluster.get_centroid()
                print(f"Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")

                dir1 = qcp.calculate_center_direction_top(cluster)
                dir2 = qcp.calculate_center_direction_bottom(cluster)

                if dir1 is not None and dir2 is not None:
                    print(f"Direction 1 (top): ({dir1[0]:.3f}, {dir1[1]:.3f})")
                    print(f"Direction 2 (bottom): ({dir2[0]:.3f}, {dir2[1]:.3f})")
                    dot = np.dot(dir1, dir2)
                    angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
                    print(f"Dot product: {dot:.3f}")
                    print(f"Angle between: {angle:.1f}°")

                    combined = (dir1 + dir2) / 2
                    print(f"Combined: ({combined[0]:.6f}, {combined[1]:.6f})")
                    print(f"Combined norm: {np.linalg.norm(combined):.10f}")

            break  # This trial failed, move to next

        # Move cluster normally
        cluster.move(qcp.dual_jacobian_center_finder_advanced)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total failures found: {failure_count}/100 trials")
print(f"\nFailure causes:")
for cause, count in sorted(cause_histogram.items(), key=lambda x: -x[1]):
    print(f"  {count}× {cause}")
