"""
Direct test using Monte Carlo's exact run_single_trial function.
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

NUM_TRIALS = 100
SIM_TIME = 100


def run_single_trial(primitive_func, start_x, start_y):
    """Exact copy of Monte Carlo's run_single_trial."""
    try:
        field = AnalyticalField(saddle1)

        # Suppress initialization prints
        with contextlib.redirect_stdout(io.StringIO()):
            cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

        # Reset to starting position (RANDOM for each trial)
        cluster.reset(x_c=start_x, y_c=start_y)

        # Run simulation
        for step in range(SIM_TIME):
            cluster.move(primitive_func)

            # Check for divergence
            centroid = cluster.get_centroid()
            if np.isnan(centroid[0]) or np.isnan(centroid[1]):
                return None, None
            if np.abs(centroid[0]) > 10.0 or np.abs(centroid[1]) > 10.0:
                # Robot diverged too far
                return None, None

        # Get final centroid position
        final_position = cluster.get_centroid()

        # Final NaN check
        if np.isnan(final_position[0]) or np.isnan(final_position[1]):
            return None, None

        return final_position[0], final_position[1]

    except Exception as e:
        # If any error occurs, return None
        print(f"Exception: {e}")
        return None, None


print("="*80)
print("Direct Test: saddle1 with dual_jacobian_center_finder_advanced")
print("="*80)
print()

np.random.seed(42)  # Same seed as Monte Carlo

successes = 0
failures = 0

for trial in range(NUM_TRIALS):
    start_x = np.random.uniform(-0.5, 0.5)
    start_y = np.random.uniform(-0.5, 0.5)

    final_x, final_y = run_single_trial(qcp.dual_jacobian_center_finder_advanced, start_x, start_y)

    if final_x is None or final_y is None:
        failures += 1
    else:
        successes += 1

    if (trial + 1) % 10 == 0:
        print(f"Completed {trial + 1} trials...")

success_rate = successes / NUM_TRIALS * 100
failure_rate = failures / NUM_TRIALS * 100

print()
print(f"Successes: {successes}/{NUM_TRIALS}")
print(f"Failures: {failures}/{NUM_TRIALS}")
print(f"Success rate: {success_rate:.1f}%")
print(f"Failure rate: {failure_rate:.1f}%")
print()
print("Expected from user's report: ~8% success, ~92% failure")
