"""
Forward Jacobian numerical tests.

For each kinematics case:
  1. Identity: J_fwd @ J_inv == I (when J is square), else pseudo-inverse identity.
  2. FD check: J_fwd columns match finite-difference d(state)/d(robot_xy).

`forward_jacobian` is currently not numerically verified anywhere except for
tiny sympy cases. This file closes that gap.
"""

import copy
import math
import sys
import os

import numpy as np
import pytest


def _wrap_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    TreeParser,
)

COC_TREES = [
    "(2,2)",
    "((2,2),(2,2))",
    "(2,2,2)",
    "((2,2),2)",
    "((2,2,2),(2,2))",
    # Note: "(3)" alone (single-leaf root) not permitted; see test_kinematics_roundtrip.py.
    "(3,3)",
    "((3,3),(2,2))",
    "((2,3),2)",
    "(3,3,3)",
]

HS_NS = [3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fd_forward_jacobian(kin, state, eps=1e-6):
    """
    FD estimate of forward Jacobian: d(state_vec) / d(robot_outputs_vec).

    When orientation=off: robot outputs are [x_1, y_1, ..., x_N, y_N].
      J shape: (n_state_vars, 2*N).
    When orientation=on: robot outputs are [x_1, y_1, ..., x_N, y_N, th_1, ..., th_N].
      J shape: (n_state_vars, 3*N), matching the square J_inv shape.
    """
    N = kin.N
    orientation = kin.orientation
    n_vars = len(kin.state_vars)
    n_outputs = 3 * N if orientation else 2 * N
    J = np.zeros((n_vars, n_outputs))

    positions = kin.inverse_kinematics(state)

    def _diff_state(s_fwd, s_bwd):
        col_vals = np.zeros(n_vars)
        for row, var in enumerate(kin.state_vars):
            a, b = s_fwd[var], s_bwd[var]
            col_vals[row] = (_wrap_diff(a, b) if (var.startswith('theta') or
                                                   var.startswith('gamma') or
                                                   var.startswith('phi')) else a - b)
        return col_vals

    # Perturb each robot xy coordinate
    for i in range(1, N + 1):
        for dim in range(2):
            col = (i - 1) * 2 + dim

            def _perturb_xy(delta, _i=i, _dim=dim):
                p = {}
                for k in range(1, N + 1):
                    base = [positions[k][0], positions[k][1]]
                    if orientation:
                        base.append(positions[k][2])
                    if k == _i:
                        base[_dim] += delta
                    p[k] = tuple(base)
                return p

            J[:, col] = _diff_state(
                kin.forward_kinematics(_perturb_xy(+eps)),
                kin.forward_kinematics(_perturb_xy(-eps)),
            ) / (2 * eps)

    # Perturb each robot heading (only when orientation=on)
    if orientation:
        for i in range(1, N + 1):
            col = 2 * N + (i - 1)

            def _perturb_th(delta, _i=i):
                p = {}
                for k in range(1, N + 1):
                    th = positions[k][2] + (delta if k == _i else 0.0)
                    p[k] = (positions[k][0], positions[k][1], th)
                return p

            J[:, col] = _diff_state(
                kin.forward_kinematics(_perturb_th(+eps)),
                kin.forward_kinematics(_perturb_th(-eps)),
            ) / (2 * eps)

    return J


def _check_forward_jacobian(kin, tol_identity=1e-9, tol_fd=1e-4):
    state = kin.default_state()
    J_inv = kin.inverse_jacobian(state)
    J_fwd = kin.forward_jacobian(state)

    rows, cols = J_inv.shape

    if rows == cols:
        # Square: J_fwd @ J_inv should be identity
        err = np.max(np.abs(J_fwd @ J_inv - np.eye(rows)))
        assert err < tol_identity, f"J_fwd @ J_inv - I: max={err:.2e} >= {tol_identity}"
    else:
        # Non-square: pseudo-inverse identity J_inv @ J_fwd @ J_inv == J_inv
        err = np.max(np.abs(J_inv @ J_fwd @ J_inv - J_inv))
        assert err < tol_identity, f"Pseudo-inverse identity failed: max={err:.2e} >= {tol_identity}"

    # FD check: J_fwd shape must match FD estimate
    J_fwd_fd = _fd_forward_jacobian(kin, state)
    assert J_fwd.shape == J_fwd_fd.shape, (
        f"J_fwd shape {J_fwd.shape} != FD shape {J_fwd_fd.shape}"
    )
    err_fd = np.max(np.abs(J_fwd - J_fwd_fd))
    assert err_fd < tol_fd, f"J_fwd vs FD: max={err_fd:.2e} >= {tol_fd}"


# ---------------------------------------------------------------------------
# Hub-and-spoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", HS_NS)
@pytest.mark.parametrize("orientation", [False, True])
def test_hub_spoke_forward_jacobian(N, orientation):
    kin = HubAndSpokeKinematics(N, orientation=orientation)
    _check_forward_jacobian(kin)


# ---------------------------------------------------------------------------
# Cluster-of-clusters tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_str", COC_TREES)
@pytest.mark.parametrize("orientation", [False, True])
def test_coc_forward_jacobian(tree_str, orientation):
    root = TreeParser.parse(tree_str)
    kin = ClusterOfClustersKinematics(root, orientation=orientation)
    _check_forward_jacobian(kin)
