"""
Test hypothesis: Angular control in advanced version causes saddle1 failures.

Compares:
1. dual_jacobian_center_finder (no angular control)
2. dual_jacobian_center_finder_advanced (with angular control)
"""

import numpy as np
import sys
import os
import io
import contextlib

sys.path.append('.')

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Saddle import saddle1
import src.control.quad_primitives as qcp

NUM_TRIALS = 50
SIM_TIME = 100

def run_trial(primitive, start_x, start_y):
    """Run single trial, return True if succeeded."""
    try:
        field = AnalyticalField(saddle1)

        # Suppress prints
        with contextlib.redirect_stdout(io.StringIO()):
            cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

        cluster.reset(x_c=start_x, y_c=start_y)

        # Run simulation
        for step in range(SIM_TIME):
            result = primitive(cluster)

            # Handle both 2-tuple and 3-tuple returns
            if len(result) == 2:
                vx_c, vy_c = result
            else:
                vx_c, vy_c, omega_c = result

            cluster.move(primitive)

            # Check for divergence
            centroid = cluster.get_centroid()
            if np.isnan(centroid[0]) or np.isnan(centroid[1]):
                return False
            if np.abs(centroid[0]) > 10.0 or np.abs(centroid[1]) > 10.0:
                return False

        # Check final position
        final_position = cluster.get_centroid()
        if np.isnan(final_position[0]) or np.isnan(final_position[1]):
            return False

        return True

    except Exception as e:
        return False


print("="*80)
print("Testing Angular Control Hypothesis on saddle1")
print("="*80)
print()

# Test both primitives
primitives = {
    'dual_jacobian_center_finder (NO angular)': qcp.dual_jacobian_center_finder,
    'dual_jacobian_center_finder_advanced (WITH angular)': qcp.dual_jacobian_center_finder_advanced
}

np.random.seed(42)  # Same random positions for fair comparison

for prim_name, prim_func in primitives.items():
    print(f"\nTesting: {prim_name}")
    print("-"*80)

    # Reset seed for each primitive so they get same starting positions
    np.random.seed(42)

    successes = 0
    failures = 0

    for trial in range(NUM_TRIALS):
        start_x = np.random.uniform(-0.5, 0.5)
        start_y = np.random.uniform(-0.5, 0.5)

        if run_trial(prim_func, start_x, start_y):
            successes += 1
        else:
            failures += 1

    success_rate = successes / NUM_TRIALS * 100
    print(f"Successes: {successes}/{NUM_TRIALS}")
    print(f"Failures: {failures}/{NUM_TRIALS}")
    print(f"Success rate: {success_rate:.1f}%")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print("If the non-angular version has significantly higher success rate,")
print("then angular control is causing the failures on saddle fields!")
