"""
Finite-difference vs. analytic inverse Jacobian tests.

For each (kinematics, orientation) combination:
  - Build kinematics, get default state.
  - Build output vector: [x_i, y_i] per robot when off, [x_i, y_i, theta_i] when on.
  - For each state variable v, perturb state[v] by +/- eps (central difference when
    orientation=on, one-sided when off) and estimate the column of J^{-1} numerically.
  - Assert max|J_analytic - J_fd| < 1e-4.
  - Assert shape is (3N, 3N) when on, (2N, 2N) when off (N = number of robots; Jacobian
    is square for properly-determined systems: hub-spoke has 2N state vars when off).

Hub-and-spoke state dim == 2N (2+N-1+N-2 == 2N-1 + 1 center = 2N), so J is (2N x 2N).
Cluster-of-clusters state dim depends on tree; for (2,2): 8 state vars == 2N with N=4.
The FD check verifies the analytic columns against the numerical estimate regardless of
whether J is square.

Additional assertions when orientation=on:
  - phi_i columns (last N columns of J) are exactly zero in all position rows.
"""

import math
import copy
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    TreeParser,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _robot_output_vec(positions, N, orientation):
    """Concatenate robot outputs into a flat vector matching the analytic J row order.

    Row order mirrors the analytic inverse Jacobian:
      orientation=off: [x_1, y_1, x_2, y_2, ..., x_N, y_N]
      orientation=on:  [x_1, y_1, ..., x_N, y_N, theta_1, theta_2, ..., theta_N]
    The heading angles are appended after all positions, not interleaved per robot.
    """
    out = []
    for i in range(1, N + 1):
        p = positions[i]
        out.extend([p[0], p[1]])
    if orientation:
        for i in range(1, N + 1):
            out.append(positions[i][2])
    return np.array(out)


def _fd_jacobian(kin, state, orientation, eps=1e-6):
    """
    Build the finite-difference estimate of the inverse Jacobian.
    Uses central differences throughout.
    Returns ndarray of shape (output_dim, n_state_vars).
    """
    N = kin.N
    state_vars = kin.state_vars
    n_vars = len(state_vars)
    out_dim = 3 * N if orientation else 2 * N

    # Nominal output vector
    pos_nom = kin.inverse_kinematics(state)
    f_nom = _robot_output_vec(pos_nom, N, orientation)

    J_fd = np.zeros((out_dim, n_vars))

    for j, var in enumerate(state_vars):
        # Forward perturbation
        s_fwd = copy.deepcopy(state)
        s_fwd[var] = state[var] + eps
        pos_fwd = kin.inverse_kinematics(s_fwd)
        f_fwd = _robot_output_vec(pos_fwd, N, orientation)

        # Backward perturbation
        s_bwd = copy.deepcopy(state)
        s_bwd[var] = state[var] - eps
        pos_bwd = kin.inverse_kinematics(s_bwd)
        f_bwd = _robot_output_vec(pos_bwd, N, orientation)

        J_fd[:, j] = (f_fwd - f_bwd) / (2 * eps)

    return J_fd


def _check_jacobian(kin, orientation, tol=1e-4):
    """Core check: analytic J matches FD J within tol."""
    N = kin.N
    state = kin.default_state()

    J_analytic = kin.inverse_jacobian(state)
    J_fd = _fd_jacobian(kin, state, orientation)

    assert J_analytic.shape == J_fd.shape, (
        f"Shape mismatch: analytic={J_analytic.shape}, fd={J_fd.shape}"
    )

    max_err = np.max(np.abs(J_analytic - J_fd))
    assert max_err < tol, (
        f"Max |J_analytic - J_fd| = {max_err:.2e} >= tol={tol}"
    )

    # Shape assertion
    expected_rows = 3 * N if orientation else 2 * N
    assert J_analytic.shape[0] == expected_rows, (
        f"Row count: expected {expected_rows}, got {J_analytic.shape[0]}"
    )

    # When orientation=on: phi_i columns must be exactly zero in all position rows
    if orientation:
        state_vars = kin.state_vars
        for i in range(1, N + 1):
            col = state_vars.index(f'phi_{i}')
            pos_block = J_analytic[:2 * N, col]
            assert np.all(pos_block == 0.0), (
                f"phi_{i} column non-zero in position rows: {pos_block}"
            )


# ---------------------------------------------------------------------------
# Hub-and-spoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [4, 5])
@pytest.mark.parametrize("orientation", [False, True])
def test_hub_spoke_fd_jacobian(N, orientation):
    kin = HubAndSpokeKinematics(N, orientation=orientation)
    _check_jacobian(kin, orientation)


# ---------------------------------------------------------------------------
# Cluster-of-clusters tests
# ---------------------------------------------------------------------------

COC_TREES = [
    "(2,2)",
    "((2,2),(2,2))",
    "(2,2,2)",
    "((2,2),2)",
    "((2,2,2),(2,2))",
]


@pytest.mark.parametrize("tree_str", COC_TREES)
@pytest.mark.parametrize("orientation", [False, True])
def test_coc_fd_jacobian(tree_str, orientation):
    root = TreeParser.parse(tree_str)
    kin = ClusterOfClustersKinematics(root, orientation=orientation)
    _check_jacobian(kin, orientation)
