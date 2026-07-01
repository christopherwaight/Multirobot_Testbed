"""
FK/IK round-trip tests for all formation types and leaf sizes.

For each (kinematics, orientation) case:
  - Build random valid positions, run FK then IK, assert positions recovered.
  - Run IK then FK on the result state, assert state variables match.
  - Direct PairBlock and TripleBlock leaf round-trips over random grids.

Catches factor-of-2 and sign errors in the analytic FK/IK at the leaf level.
"""

import copy
import math
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

SEED = 42
RNG = random.Random(SEED)
NP_RNG = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_diff(a, b):
    """Signed difference a - b, wrapped to [-pi, pi]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def _random_hub_spoke_positions(N):
    """Random positions spread around a hub at origin, not collinear."""
    hub = (RNG.uniform(-0.5, 0.5), RNG.uniform(-0.5, 0.5))
    positions = {}
    angle = RNG.uniform(0, 2 * math.pi)
    for i in range(1, N):
        r = RNG.uniform(0.15, 0.45)
        positions[i] = (hub[0] + r * math.cos(angle), hub[1] + r * math.sin(angle))
        angle += RNG.uniform(0.4, 1.2)
    positions[N] = hub
    return positions


def _random_coc_positions(kin):
    """Use IK on a jittered version of default state to get valid random positions."""
    state = kin.default_state()
    state_vars = kin.state_vars
    for v in state_vars:
        if v in ('x_c', 'y_c'):
            state[v] += RNG.uniform(-0.3, 0.3)
        elif v == 'theta_c' or v.startswith('theta_'):
            state[v] += RNG.uniform(-0.8, 0.8)
        elif v.startswith('r_') or v == 'L' or v.startswith('L_'):
            state[v] = max(0.05, state[v] + RNG.uniform(-0.05, 0.05))
        elif v.startswith('p_') or v.startswith('q_'):
            state[v] = max(0.05, state[v] + RNG.uniform(-0.05, 0.05))
        elif v.startswith('beta_'):
            state[v] = max(0.2, min(math.pi - 0.2, state[v] + RNG.uniform(-0.3, 0.3)))
        elif v.startswith('phi_'):
            pass  # leave at zero
    positions = kin.inverse_kinematics(state)
    return positions, state


def _state_eq(s1, s2, vars_, tol=5e-9):
    """Assert two states agree on vars_, treating theta-like vars mod 2pi."""
    for v in vars_:
        a, b = s1[v], s2[v]
        if v.startswith('theta') or v.startswith('gamma') or v.startswith('phi'):
            err = abs(_angle_diff(a, b))
        else:
            err = abs(a - b)
        assert err < tol, f"State var '{v}': {a} vs {b}, diff={err:.2e} >= tol={tol}"


def _positions_eq(p1, p2, N, tol=1e-9):
    """Assert two position dicts agree to tol."""
    for i in range(1, N + 1):
        for dim in range(len(p1[i])):
            err = abs(p1[i][dim] - p2[i][dim])
            assert err < tol, f"Robot {i} dim {dim}: {p1[i][dim]} vs {p2[i][dim]}, diff={err:.2e}"


# ---------------------------------------------------------------------------
# Hub-and-spoke round-trips
# ---------------------------------------------------------------------------

HS_NS = [3, 4, 5, 6]


@pytest.mark.parametrize("N", HS_NS)
@pytest.mark.parametrize("orientation", [False, True])
def test_hub_spoke_positions_roundtrip(N, orientation):
    """FK(positions) -> IK(state) recovers original positions."""
    kin = HubAndSpokeKinematics(N, orientation=orientation)
    for _ in range(5):
        positions = _random_hub_spoke_positions(N)
        if orientation:
            positions = {k: (v[0], v[1], RNG.uniform(-math.pi, math.pi)) for k, v in positions.items()}
        state = kin.forward_kinematics(positions)
        positions2 = kin.inverse_kinematics(state)
        _positions_eq(positions, positions2, N)


@pytest.mark.parametrize("N", HS_NS)
@pytest.mark.parametrize("orientation", [False, True])
def test_hub_spoke_state_roundtrip(N, orientation):
    """IK(state) -> FK(positions) recovers original state (mod 2pi for angles)."""
    kin = HubAndSpokeKinematics(N, orientation=orientation)
    state = kin.default_state()
    positions = kin.inverse_kinematics(state)
    state2 = kin.forward_kinematics(positions)
    _state_eq(state, state2, kin.state_vars)


# ---------------------------------------------------------------------------
# Cluster-of-clusters round-trips
# ---------------------------------------------------------------------------

COC_TREES = [
    "(2,2)",
    "(2,2,2)",
    "((2,2),(2,2))",
    "((2,2),2)",
    "((2,2,2),(2,2))",
    # Note: "(3)" alone (size-3 single-leaf root) is not permitted; IK fails with
    # KeyError on 'theta_c' because a leaf root has no root-level pose vars. The fix
    # would be to emit a warning like "size-1-root tree not permitted" rather than an
    # opaque KeyError. Use (3,3) and ((3,3),(2,2)) to cover size-3 leaf code paths.
    "(3,3)",
    "((3,3),(2,2))",
    "((2,3),2)",
]


@pytest.mark.parametrize("tree_str", COC_TREES)
@pytest.mark.parametrize("orientation", [False, True])
def test_coc_positions_roundtrip(tree_str, orientation):
    """FK(positions) -> IK(state) recovers original positions."""
    root = TreeParser.parse(tree_str)
    kin = ClusterOfClustersKinematics(root, orientation=orientation)
    for _ in range(3):
        positions, _ = _random_coc_positions(kin)
        if orientation:
            positions = {k: (v[0], v[1], RNG.uniform(-math.pi, math.pi)) for k, v in positions.items()}
        state = kin.forward_kinematics(positions)
        positions2 = kin.inverse_kinematics(state)
        _positions_eq(positions, positions2, kin.N)


@pytest.mark.parametrize("tree_str", COC_TREES)
@pytest.mark.parametrize("orientation", [False, True])
def test_coc_state_roundtrip(tree_str, orientation):
    """IK(state) -> FK(positions) recovers original state."""
    root = TreeParser.parse(tree_str)
    kin = ClusterOfClustersKinematics(root, orientation=orientation)
    state = kin.default_state()
    positions = kin.inverse_kinematics(state)
    state2 = kin.forward_kinematics(positions)
    _state_eq(state, state2, kin.state_vars)


# ---------------------------------------------------------------------------
# Direct leaf-block round-trips
# ---------------------------------------------------------------------------

def test_pairblock_roundtrip():
    """PairBlock.forward(PairBlock.inverse(state)) == state over a random grid."""
    for _ in range(50):
        x_c = RNG.uniform(-1, 1)
        y_c = RNG.uniform(-1, 1)
        theta = RNG.uniform(-math.pi, math.pi)
        L = RNG.uniform(0.05, 0.8)
        xa, ya, xb, yb = PairBlock.inverse(x_c, y_c, theta, L)
        x_c2, y_c2, theta2, L2 = PairBlock.forward(xa, ya, xb, yb)
        assert abs(x_c - x_c2) < 1e-12, f"x_c: {x_c} vs {x_c2}"
        assert abs(y_c - y_c2) < 1e-12, f"y_c: {y_c} vs {y_c2}"
        assert abs(_angle_diff(theta, theta2)) < 1e-12, f"theta: {theta} vs {theta2}"
        assert abs(L - L2) < 1e-12, f"L: {L} vs {L2}"


def test_tripleblock_roundtrip():
    """TripleBlock.forward(TripleBlock.inverse(state)) == state over a random grid."""
    for _ in range(50):
        x_c = RNG.uniform(-1, 1)
        y_c = RNG.uniform(-1, 1)
        theta_c = RNG.uniform(-math.pi, math.pi)
        p = RNG.uniform(0.1, 0.5)
        q = RNG.uniform(0.1, 0.5)
        beta = RNG.uniform(0.3, math.pi - 0.3)
        xa, ya, xb, yb, xc, yc = TripleBlock.inverse(x_c, y_c, theta_c, p, beta, q)
        s = TripleBlock.forward(xa, ya, xb, yb, xc, yc)
        assert abs(x_c - s['x_c']) < 1e-12, f"x_c: {x_c} vs {s['x_c']}"
        assert abs(y_c - s['y_c']) < 1e-12, f"y_c: {y_c} vs {s['y_c']}"
        assert abs(_angle_diff(theta_c, s['theta_c'])) < 1e-12, f"theta_c: {theta_c} vs {s['theta_c']}"
        assert abs(p - s['p']) < 1e-12, f"p: {p} vs {s['p']}"
        assert abs(q - s['q']) < 1e-12, f"q: {q} vs {s['q']}"
        # acos accumulates ~1e-9 error near beta ~ pi (known limitation: beta is
        # ill-defined near collinear configurations; a future fix would use atan2).
        assert abs(beta - s['beta']) < 5e-9, f"beta: {beta} vs {s['beta']}"
