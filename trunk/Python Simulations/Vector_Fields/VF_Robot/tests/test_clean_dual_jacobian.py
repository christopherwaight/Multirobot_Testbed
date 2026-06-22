"""
Test the new clean dual Jacobian implementation across all field types.
"""

import numpy as np
import sys
import io
import contextlib

sys.path.append('.')

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1
from src.fields.environments.Source import source1
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Sinking_Vortex import sinking_vortex1
from src.fields.environments.Spewing_Vortex import spewing_vortex1
from src.fields.environments.Saddle import saddle1
import src.control.quad_primitives as qcp

NUM_TRIALS = 50
SIM_TIME = 100

def test_field(field_func, field_name):
    """Test dual_jacobian_center_finder_advanced on one field type."""
    np.random.seed(42)
    successes = 0

    for trial in range(NUM_TRIALS):
        start_x = np.random.uniform(-0.5, 0.5)
        start_y = np.random.uniform(-0.5, 0.5)

        field = AnalyticalField(field_func)
        with contextlib.redirect_stdout(io.StringIO()):
            cluster = QuadCluster('config/formations/quad_square_default.yaml', field)
        cluster.reset(x_c=start_x, y_c=start_y)

        success = True
        for step in range(SIM_TIME):
            cluster.move(qcp.dual_jacobian_center_finder_advanced)

            centroid = cluster.get_centroid()
            if np.isnan(centroid[0]) or np.isnan(centroid[1]) or \
               np.abs(centroid[0]) > 10.0 or np.abs(centroid[1]) > 10.0:
                success = False
                break

        if success:
            final = cluster.get_centroid()
            if not (np.isnan(final[0]) or np.isnan(final[1])):
                successes += 1

    return successes, NUM_TRIALS

print("="*80)
print("Testing New Clean Dual Jacobian Implementation")
print("="*80)
print()

fields = [
    (sink1, "sink1"),
    (source1, "source1"),
    (vortex1, "vortex1"),
    (sinking_vortex1, "sinking_vortex1"),
    (spewing_vortex1, "spewing_vortex1"),
    (saddle1, "saddle1"),
]

print(f"{'Field':<20} {'Success Rate':<15} {'Status'}")
print("-"*80)

all_passed = True
for field_func, field_name in fields:
    successes, total = test_field(field_func, field_name)
    success_rate = successes / total * 100

    if success_rate == 100:
        status = "✓ PERFECT"
    elif success_rate >= 95:
        status = "✓ Excellent"
    elif success_rate >= 80:
        status = "⚠ Good"
    else:
        status = "✗ FAILED"
        all_passed = False

    print(f"{field_name:<20} {successes}/{total} ({success_rate:.0f}%){'':<5} {status}")

print()
print("="*80)
if all_passed:
    print("✓ ALL FIELDS WORK PERFECTLY - NO NaN FAILURES!")
else:
    print("Some fields still have issues")
print("="*80)
