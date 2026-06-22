"""
Diagnose the source of NaN failures in dual_jacobian_center_finder_advanced.

Traces through the estimation process to find where NaN originates.
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


def diagnose_estimation(cluster):
    """
    Step through center estimation with detailed diagnostics.
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

    print(f"Robot positions:")
    print(f"  R1: ({x1:.3f}, {y1:.3f})")
    print(f"  R2: ({x2:.3f}, {y2:.3f})")
    print(f"  R3: ({x3:.3f}, {y3:.3f})")
    print(f"  R4: ({x4:.3f}, {y4:.3f})")

    # Calculate triangle centers
    top_triangle_center = np.array([(x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0])
    bottom_triangle_center = np.array([(x4 + x2 + x3) / 3.0, (y4 + y2 + y3) / 3.0])

    print(f"\nTriangle centroids:")
    print(f"  Top (1,2,3): ({top_triangle_center[0]:.3f}, {top_triangle_center[1]:.3f})")
    print(f"  Bottom (2,3,4): ({bottom_triangle_center[0]:.3f}, {bottom_triangle_center[1]:.3f})")

    # Get center directions from both triangles
    dir1 = qcp.calculate_center_direction_top(cluster)
    dir2 = qcp.calculate_center_direction_bottom(cluster)

    print(f"\nDirection estimates:")
    if dir1 is not None:
        print(f"  Top triangle: ({dir1[0]:.3f}, {dir1[1]:.3f}) [magnitude: {np.linalg.norm(dir1):.3f}]")
    else:
        print(f"  Top triangle: FAILED (None)")

    if dir2 is not None:
        print(f"  Bottom triangle: ({dir2[0]:.3f}, {dir2[1]:.3f}) [magnitude: {np.linalg.norm(dir2):.3f}]")
    else:
        print(f"  Bottom triangle: FAILED (None)")

    # Check if either failed
    if dir1 is None or dir2 is None:
        print("\n❌ FAILURE CAUSE: One triangle couldn't estimate direction (singular matrix)")
        return

    # Check angle between directions
    dot_product = np.dot(dir1, dir2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi

    print(f"\nAngle between directions: {angle:.1f}°")

    # Try to find intersection
    center_estimate = qcp.find_line_intersection(top_triangle_center, dir1, bottom_triangle_center, dir2)

    if center_estimate is None:
        print("\n⚠️  Line intersection FAILED - using fallback")

        # This is where the NaN bug happens!
        combined_dir = (dir1 + dir2) / 2
        print(f"  Combined direction: ({combined_dir[0]:.3f}, {combined_dir[1]:.3f})")

        norm = np.linalg.norm(combined_dir)
        print(f"  Combined magnitude: {norm:.6f}")

        if norm < 1e-10:
            print("\n❌ FAILURE CAUSE: Directions cancel out (opposite)!")
            print(f"     dir1 + dir2 ≈ 0 → Cannot normalize!")
            print(f"     This creates NaN: {combined_dir / norm}")
            return

        combined_dir_norm = combined_dir / norm
        print(f"  Normalized: ({combined_dir_norm[0]:.3f}, {combined_dir_norm[1]:.3f})")
    else:
        print(f"\n✓ Intersection found: ({center_estimate[0]:.3f}, {center_estimate[1]:.3f})")


def find_failing_case():
    """
    Run trials until we find one that fails, then diagnose it.
    """
    print("="*80)
    print("Searching for a failing case to diagnose...")
    print("="*80)
    print()

    np.random.seed(42)

    for trial in range(200):
        start_x = np.random.uniform(-0.5, 0.5)
        start_y = np.random.uniform(-0.5, 0.5)

        field = AnalyticalField(saddle1)
        with contextlib.redirect_stdout(io.StringIO()):
            cluster = QuadCluster('config/formations/quad_square_default.yaml', field)
        cluster.reset(x_c=start_x, y_c=start_y)

        # Run simulation
        failed = False
        failure_step = -1
        for step in range(100):
            # Check BEFORE moving
            center_est, radius = qcp.estimate_center_and_radius(cluster)

            if center_est is None or np.isnan(center_est[0]) or np.isnan(center_est[1]):
                print(f"\n{'='*80}")
                print(f"FOUND FAILURE at trial {trial+1}, step {step}")
                print(f"Starting position: ({start_x:.3f}, {start_y:.3f})")
                print(f"{'='*80}\n")

                diagnose_estimation(cluster)
                return True

            cluster.move(qcp.dual_jacobian_center_finder_advanced)

            centroid = cluster.get_centroid()
            if np.isnan(centroid[0]) or np.isnan(centroid[1]):
                print(f"\n{'='*80}")
                print(f"FOUND FAILURE at trial {trial+1}, step {step}")
                print(f"Starting position: ({start_x:.3f}, {start_y:.3f})")
                print(f"NaN appeared in centroid AFTER move")
                print(f"{'='*80}\n")

                diagnose_estimation(cluster)
                return True

    print("No failures found in 200 trials!")
    return False


if __name__ == "__main__":
    # Don't raise on invalid, just warn so we can trace it
    with np.errstate(invalid='warn'):
        find_failing_case()
