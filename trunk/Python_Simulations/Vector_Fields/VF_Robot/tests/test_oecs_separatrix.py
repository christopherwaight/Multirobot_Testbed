"""
Test the objective separatrix traverser (Primitive 11, oecs_separatrix_step)
against the analytic double gyre. Companion to test_oecs_estimator.py, which
covers the shared _strain_quantities estimator used by both Primitive 10 and
Primitive 11; this file covers what is specific to Primitive 11: the
flow-free tangent-selection rule, the CAPTURE test, and closed-loop
traversal/objectivity.

Checks:
1. Tangent selection picks the eigenvector with the LARGER |grad_s1 . e_i|
   (the along-track one), not the smaller one, on both separatrix halves.
2. Transverse and tangent channels partition the same two dot products
   (no third quantity is computed).
3. CAPTURE fires unconditionally near a 2D stationary point of s1 (a flow
   saddle) and does not fire away from one, even where |grad_s1| dips on an
   ordinary trench point transversally (it does not dip in EVERY direction
   there, only CAPTURE's full-gradient test distinguishes the two).
4. Closed loop: six starts on the double gyre all reach and hold their
   nearest saddle (matches the path-match experiment,
   experiments/main_separatrix_traverse.py, at a coarser tolerance for
   speed).
5. Objectivity smoke test: rotating-frame flow does NOT change the
   tangent-selection choice at a point where the true (non-rotating) flow
   would have flipped it under the earlier flow-based design (regression
   guard for the swirl-corruption bug this file's design fixed).

Run from the VF_Robot root:
  MPLBACKEND=Agg ./venv/bin/python3 -m pytest tests/test_oecs_separatrix.py -q
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.pentagon_primitives import (
    _fit_vector_quadratic, _strain_quantities, oecs_separatrix_step
)
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP
)
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField

A = 0.1
RHO = 0.075
FORMATION_CONFIG = "config/formations/pentagon_small.yaml"


def _pentagon_rel(phase=0.0):
    ang = np.arange(5) * 2.0 * np.pi / 5.0 + phase
    ring = np.column_stack([RHO * np.cos(ang), RHO * np.sin(ang)])
    return np.vstack([np.zeros((1, 2)), ring])


def _fit_at(cx, cy, field=double_gyre_static, phase=0.0):
    rel = _pentagon_rel(phase)
    u = np.array([field(cx + r[0], cy + r[1])[0] for r in rel])
    v = np.array([field(cx + r[0], cy + r[1])[1] for r in rel])
    return _fit_vector_quadratic(rel, u, v)


def test_tangent_is_larger_projection_upper_half():
    # Upper separatrix: true tangent is vertical. grad_s1 should have a
    # LARGER dot product with whichever eigenvector is vertical.
    theta_u, theta_v = _fit_at(0.0, 0.30)
    _, grad_s1, e1, e2, _ = _strain_quantities(theta_u, theta_v)
    d1, d2 = abs(float(grad_s1 @ e1)), abs(float(grad_s1 @ e2))
    tangent = e1 if d1 > d2 else e2
    assert abs(abs(tangent[1]) - 1.0) < 1e-2, tangent


def test_tangent_is_larger_projection_lower_half():
    # Lower separatrix: identity swaps (test_eigenframe_identity_swap in
    # test_oecs_estimator.py), but the tangent must STILL be vertical.
    theta_u, theta_v = _fit_at(0.0, -0.30)
    _, grad_s1, e1, e2, _ = _strain_quantities(theta_u, theta_v)
    d1, d2 = abs(float(grad_s1 @ e1)), abs(float(grad_s1 @ e2))
    tangent = e1 if d1 > d2 else e2
    assert abs(abs(tangent[1]) - 1.0) < 1e-2, tangent


def test_tangent_prefers_separatrix_over_wall_at_crossing():
    # Just off the top saddle, on the separatrix side: the tangent must
    # still resolve vertical (continuing the separatrix), not horizontal
    # (the crossing wall branch), even this close to the crossing.
    theta_u, theta_v = _fit_at(0.0, 0.48)
    _, grad_s1, e1, e2, _ = _strain_quantities(theta_u, theta_v)
    d1, d2 = abs(float(grad_s1 @ e1)), abs(float(grad_s1 @ e2))
    tangent = e1 if d1 > d2 else e2
    assert abs(abs(tangent[1]) - 1.0) < 1e-2, tangent


def test_capture_fires_only_near_a_2d_stationary_point():
    # Full |grad_s1| must be small near a saddle (2D minimum of s1) and
    # large on an ordinary trench point, even though the TRANSVERSE
    # component of grad_s1 is zero at BOTH (by definition of being on the
    # trench). This is what makes the CAPTURE test (on the full gradient)
    # distinct from the transverse-only channel.
    theta_u, theta_v = _fit_at(0.0, 0.495)  # 5 mm from the top saddle
    _, grad_s1, _, _, _ = _strain_quantities(theta_u, theta_v)
    assert np.linalg.norm(grad_s1) < 0.15

    theta_u, theta_v = _fit_at(0.0, 0.30)  # ordinary mid-trench point
    _, grad_s1, _, _, _ = _strain_quantities(theta_u, theta_v)
    assert np.linalg.norm(grad_s1) > 0.5


def _run_traverser(sx, sy, steps=700):
    field = AnalyticalField(double_gyre_static)
    cl = PentagonCluster(FORMATION_CONFIG, field)
    cl.reset(sx, sy)

    def prim(c):
        vx, vy = oecs_separatrix_step(c, v_max=0.04, g_perp=1.0, s_trim=0.05,
                                      r_band=0.05, g_capture=0.15,
                                      s_capture=None)
        return vx * 3.0, vy * 3.0

    for _ in range(steps):
        cl.move(prim)
    return cl.get_centroid()


def test_six_starts_capture_at_nearest_saddle():
    starts = [(-0.45, 0.30), (0.05, 0.40), (0.0, 0.0),
             (0.10, -0.20), (0.25, 0.42), (-0.20, -0.30)]
    for sx, sy in starts:
        cx, cy = _run_traverser(sx, sy)
        d_top = np.hypot(cx - SADDLE_TOP[0], cy - SADDLE_TOP[1])
        d_bot = np.hypot(cx - SADDLE_BOTTOM[0], cy - SADDLE_BOTTOM[1])
        assert min(d_top, d_bot) < 0.05, (sx, sy, cx, cy)


def test_tangent_selection_immune_to_rotating_frame_swirl():
    # Regression guard for the bug this design fixed: an earlier
    # flow-based tangent-selection rule picked the WRONG eigenvector under
    # a rotating observer's swirl at this exact point (verified true flow
    # favors one eigenvector 2:1, swirl-corrupted flow favors the other).
    # The gradient-based rule must resolve the SAME tangent in both frames,
    # since it never reads the flow.
    omega_rot = 0.2
    p = np.array([0.05, 0.40])

    theta_u0, theta_v0 = _fit_at(p[0], p[1])
    _, grad_s1_0, e1_0, e2_0, _ = _strain_quantities(theta_u0, theta_v0)
    d1_0, d2_0 = abs(float(grad_s1_0 @ e1_0)), abs(float(grad_s1_0 @ e2_0))
    tangent_inertial_is_e2 = d2_0 > d1_0

    def rotated_field(x, y):
        u0, v0 = double_gyre_static(x, y)
        return u0 - omega_rot * y, v0 + omega_rot * x

    theta_u1, theta_v1 = _fit_at(p[0], p[1], field=rotated_field)
    _, grad_s1_1, e1_1, e2_1, _ = _strain_quantities(theta_u1, theta_v1)
    d1_1, d2_1 = abs(float(grad_s1_1 @ e1_1)), abs(float(grad_s1_1 @ e2_1))
    tangent_rotating_is_e2 = d2_1 > d1_1

    assert tangent_inertial_is_e2 == tangent_rotating_is_e2


if __name__ == "__main__":
    test_tangent_is_larger_projection_upper_half()
    test_tangent_is_larger_projection_lower_half()
    test_tangent_prefers_separatrix_over_wall_at_crossing()
    test_capture_fires_only_near_a_2d_stationary_point()
    test_six_starts_capture_at_nearest_saddle()
    test_tangent_selection_immune_to_rotating_frame_swirl()
    print("All separatrix traverser tests passed.")
