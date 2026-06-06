"""
Sign and direction convention tests for clusterbuilder kinematics.

These pin the physical meaning of each state variable and catch
transposed-Jacobian, wrong-sign-of-theta, and off-by-factor-of-2 bugs that
can pass numeric tolerance checks because the error happens to cancel at the
default state.

Checks:
  1. Centroid translation: perturbing x_c (or y_c) moves all robots by the same amount.
  2. Cluster rotation direction: perturbing theta_c by +eps produces a CCW rotation.
  3. Size-2 leaf L scaling: perturbing L_<nid> by +eps moves robots eps/2 apart along
     their axis (not eps apart), verifying the factor-of-2 encoding in PairBlock.j_inv.
  4. SAS leaf shape vars: perturbing p, beta, q each move robots consistently with IK.
  5. TripleBlock internal self-consistency: IK(FK(positions)) == positions for a
     randomized grid covering all four quadrants of theta_c.
"""

import math
import copy
import random
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    PairBlock,
    TripleBlock,
    TreeParser,
)

SEED = 99
RNG = random.Random(SEED)
EPS = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _positions_flat(positions, N):
    """Return array [x1, y1, x2, y2, ...] for robots 1..N."""
    out = []
    for i in range(1, N + 1):
        out.extend([positions[i][0], positions[i][1]])
    return np.array(out)


def _centroid(positions, N):
    xs = [positions[i][0] for i in range(1, N + 1)]
    ys = [positions[i][1] for i in range(1, N + 1)]
    return np.mean(xs), np.mean(ys)


def _all_formations():
    """Yield (kin, description) for CoC formations with x_c, y_c, theta_c in state.
    Hub-and-spoke uses x_h, y_h (hub position) instead, so it is excluded from
    centroid-translation and rotation-direction tests."""
    # Note: "(3)" alone (single-leaf root) not permitted; see test_kinematics_roundtrip.py.
    for tree_str in ["(2,2)", "(2,2,2)", "((2,2),(2,2))", "(3,3)", "(3,3,3)"]:
        root = TreeParser.parse(tree_str)
        yield ClusterOfClustersKinematics(root, orientation=False), f"CoC-{tree_str}"


# ---------------------------------------------------------------------------
# 1. Centroid translation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,var", [(0, 'x_c'), (1, 'y_c')])
def test_centroid_translation(dim, var):
    """Perturbing x_c or y_c by +eps moves every robot's corresponding coordinate by +eps."""
    for kin, desc in _all_formations():
        state = kin.default_state()
        pos_nom = kin.inverse_kinematics(state)
        state_pert = copy.deepcopy(state)
        state_pert[var] += EPS
        pos_pert = kin.inverse_kinematics(state_pert)
        for i in range(1, kin.N + 1):
            delta = pos_pert[i][dim] - pos_nom[i][dim]
            assert abs(delta - EPS) < EPS * 0.01, (
                f"[{desc}] robot {i} {var} perturb: expected delta={EPS:.2e}, got {delta:.2e}"
            )
            other_dim = 1 - dim
            delta_other = pos_pert[i][other_dim] - pos_nom[i][other_dim]
            assert abs(delta_other) < EPS * 0.01, (
                f"[{desc}] robot {i} perpendicular displacement on {var} perturb: {delta_other:.2e}"
            )


# ---------------------------------------------------------------------------
# 2. Cluster rotation direction
# ---------------------------------------------------------------------------

def test_cluster_rotation_direction():
    """Perturbing theta_c by +eps produces a CCW rotation of all robots about the centroid."""
    for kin, desc in _all_formations():
        state = kin.default_state()
        pos_nom = kin.inverse_kinematics(state)
        cx, cy = pos_nom[1][0], pos_nom[1][1]
        cx = state['x_c']
        cy = state['y_c']

        state_pert = copy.deepcopy(state)
        state_pert['theta_c'] += EPS
        pos_pert = kin.inverse_kinematics(state_pert)

        for i in range(1, kin.N + 1):
            # Cross product (r x dr) should be > 0 for CCW
            rx = pos_nom[i][0] - cx
            ry = pos_nom[i][1] - cy
            dx = pos_pert[i][0] - pos_nom[i][0]
            dy = pos_pert[i][1] - pos_nom[i][1]
            cross = rx * dy - ry * dx
            # Only check if robot is not at centroid
            r_mag = math.hypot(rx, ry)
            if r_mag < 1e-6:
                continue
            assert cross > 0, (
                f"[{desc}] robot {i} cross product {cross:.2e} <= 0 (expected CCW rotation)"
            )


# ---------------------------------------------------------------------------
# 3. Size-2 leaf L scaling: each robot moves eps/2 (not eps)
# ---------------------------------------------------------------------------

def _find_pair_leaf_state_key(kin):
    """Return the L_<nid> key for a size-2 leaf in a CoC kinematics."""
    for v in kin.state_vars:
        if v.startswith('L_'):
            return v
    return None


def test_pair_leaf_L_factor_of_2():
    """
    Perturbing L_<nid> by +eps moves the two robots in that pair eps/2 apart
    along their connecting axis. Verifies PairBlock.j_inv column 3 encoding.
    """
    for tree_str in ["(2,2)", "(2,2,2)", "((2,2),(2,2))"]:
        root = TreeParser.parse(tree_str)
        kin = ClusterOfClustersKinematics(root, orientation=False)
        L_key = _find_pair_leaf_state_key(kin)
        if L_key is None:
            continue

        # Find which robots are in that leaf by looking at the tree
        nid = L_key[2:]  # strip 'L_'
        pair_leaf = None
        for node in _iter_nodes(kin.root):
            if str(node.node_id) == nid and node.is_leaf and node.size == 2:
                pair_leaf = node
                break
        if pair_leaf is None:
            continue

        state = kin.default_state()
        pos_nom = kin.inverse_kinematics(state)
        i1, i2 = pair_leaf.robot_indices
        x1n, y1n = pos_nom[i1][0], pos_nom[i1][1]
        x2n, y2n = pos_nom[i2][0], pos_nom[i2][1]
        axis = math.atan2(y2n - y1n, x2n - x1n)

        state_pert = copy.deepcopy(state)
        state_pert[L_key] += EPS
        pos_pert = kin.inverse_kinematics(state_pert)
        x1p, y1p = pos_pert[i1][0], pos_pert[i1][1]
        x2p, y2p = pos_pert[i2][0], pos_pert[i2][1]

        # Robot 1 should move -EPS/2 along axis
        d1_along = (x1p - x1n) * math.cos(axis) + (y1p - y1n) * math.sin(axis)
        # Robot 2 should move +EPS/2 along axis
        d2_along = (x2p - x2n) * math.cos(axis) + (y2p - y2n) * math.sin(axis)

        assert abs(d1_along - (-EPS / 2)) < EPS * 0.01, (
            f"[{tree_str}] L key {L_key}: robot {i1} along-axis delta={d1_along:.2e}, "
            f"expected {-EPS/2:.2e}"
        )
        assert abs(d2_along - (+EPS / 2)) < EPS * 0.01, (
            f"[{tree_str}] L key {L_key}: robot {i2} along-axis delta={d2_along:.2e}, "
            f"expected {+EPS/2:.2e}"
        )


def _iter_nodes(node):
    """Pre-order traversal of tree nodes."""
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


# ---------------------------------------------------------------------------
# 4. SAS leaf p, q, beta displacement matches IK perturbation
# ---------------------------------------------------------------------------

def _sas_leaf_keys(kin):
    """Return (p_key, beta_key, q_key, nid) for the first size-3 leaf found."""
    for node in _iter_nodes(kin.root):
        if node.is_leaf and node.size == 3:
            nid = str(node.node_id)
            return f'p_{nid}', f'beta_{nid}', f'q_{nid}', node
    return None


@pytest.mark.parametrize("tree_str", ["(3,3)", "((3,3),(2,2))", "(3,3,3)"])
@pytest.mark.parametrize("shape_var_idx", [0, 1, 2])  # p, beta, q
def test_sas_leaf_shape_perturbation(tree_str, shape_var_idx):
    """
    Perturbing p (or beta or q) by +eps in the state moves the SAS robots by
    exactly the displacement IK predicts.
    """
    root = TreeParser.parse(tree_str)
    kin = ClusterOfClustersKinematics(root, orientation=False)
    result = _sas_leaf_keys(kin)
    if result is None:
        pytest.skip("No size-3 leaf found")
    p_key, beta_key, q_key, leaf_node = result
    keys = [p_key, beta_key, q_key]
    var = keys[shape_var_idx]

    state = kin.default_state()
    pos_nom = kin.inverse_kinematics(state)

    state_pert = copy.deepcopy(state)
    state_pert[var] = max(0.05, state_pert[var] + EPS) if 'beta' not in var else max(0.2, min(math.pi - 0.2, state_pert[var] + EPS))
    pos_pert = kin.inverse_kinematics(state_pert)

    for ri in leaf_node.robot_indices:
        dx = pos_pert[ri][0] - pos_nom[ri][0]
        dy = pos_pert[ri][1] - pos_nom[ri][1]
        # Displacement must be non-zero only in the perturbed direction;
        # we just check it's finite and consistent with IK (not NaN/inf)
        assert math.isfinite(dx) and math.isfinite(dy), (
            f"[{tree_str}] {var} perturb: robot {ri} displacement is not finite"
        )

    # Jacobian prediction: J_inv[:, col_var] * EPS should match the displacements
    state_vars = kin.state_vars
    J_inv = kin.inverse_jacobian(state)
    col = state_vars.index(var)
    for ri in leaf_node.robot_indices:
        rx, ry = 2*(ri-1), 2*(ri-1)+1
        dx_nom = pos_pert[ri][0] - pos_nom[ri][0]
        dy_nom = pos_pert[ri][1] - pos_nom[ri][1]
        dx_jac = J_inv[rx, col] * EPS
        dy_jac = J_inv[ry, col] * EPS
        err_x = abs(dx_nom - dx_jac)
        err_y = abs(dy_nom - dy_jac)
        assert err_x < EPS * 0.05, (
            f"[{tree_str}] {var}: robot {ri} x: IK={dx_nom:.2e}, J*eps={dx_jac:.2e}, err={err_x:.2e}"
        )
        assert err_y < EPS * 0.05, (
            f"[{tree_str}] {var}: robot {ri} y: IK={dy_nom:.2e}, J*eps={dy_jac:.2e}, err={err_y:.2e}"
        )


# ---------------------------------------------------------------------------
# 5. TripleBlock internal self-consistency over all quadrants of theta_c
# ---------------------------------------------------------------------------

def test_tripleblock_all_quadrants():
    """
    IK(FK(positions)) == positions for theta_c spanning all four quadrants.
    Tests clusterbuilder's theta_c definition is self-consistent.
    """
    for _ in range(80):
        x_c = RNG.uniform(-1, 1)
        y_c = RNG.uniform(-1, 1)
        theta_c = RNG.uniform(-math.pi, math.pi)
        p = RNG.uniform(0.1, 0.5)
        q = RNG.uniform(0.1, 0.5)
        beta = RNG.uniform(0.3, math.pi - 0.3)
        xa, ya, xb, yb, xc, yc = TripleBlock.inverse(x_c, y_c, theta_c, p, beta, q)
        s = TripleBlock.forward(xa, ya, xb, yb, xc, yc)
        xa2, ya2, xb2, yb2, xc2, yc2 = TripleBlock.inverse(
            s['x_c'], s['y_c'], s['theta_c'], s['p'], s['beta'], s['q']
        )
        for name, a, b in [('xa', xa, xa2), ('ya', ya, ya2), ('xb', xb, xb2),
                            ('yb', yb, yb2), ('xc', xc, xc2), ('yc', yc, yc2)]:
            # acos accumulates ~1e-10 near collinear; 5e-9 matches round-trip tolerance
            assert abs(a - b) < 5e-9, (
                f"TripleBlock quadrant {theta_c:.3f}: {name} {a:.10f} vs {b:.10f}"
            )
