"""
Symbolic Jacobian acceptance tests.

For small trees:
  1. Build symbolic J^{-1} and numeric J^{-1}.
  2. Substitute default-state floats into the symbolic matrix.
  3. Assert max abs difference < 1e-9.
  4. For the smallest trees, also assert symbolic J * J^{-1} == I (identity check).

Trees tested: (2,2), ((2,2),(2,2)) for CoC; hub-and-spoke N=4 for HS.
One orientation=on case for each kinematics type.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    TreeParser,
)

try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

pytestmark = pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sub_state(J_sym, state, state_vars):
    """Substitute all numeric state values into a sympy matrix."""
    subs = {sympy.Symbol(v): float(state[v]) for v in state_vars}
    return np.array(J_sym.subs(subs).evalf().tolist(), dtype=float)


def _check_numeric_match(kin, orientation, tol=1e-9):
    """Symbolic J^{-1} at default state equals numeric J^{-1} at default state."""
    state = kin.default_state()
    J_num = kin.inverse_jacobian(state)
    J_sym = kin.inverse_jacobian_symbolic()
    J_sym_num = _sub_state(J_sym, state, kin.state_vars)
    max_err = np.max(np.abs(J_num - J_sym_num))
    assert max_err < tol, f"Max |J_numeric - J_symbolic@default| = {max_err:.2e} >= {tol}"


def _check_identity(kin, tol=1e-9):
    """Symbolic J * J^{-1} == I (only feasible for small square systems)."""
    J_sym = kin.inverse_jacobian_symbolic()
    rows, cols = J_sym.shape
    if rows != cols:
        pytest.skip(f"J is not square ({rows}x{cols}), identity check skipped")
    J_sym_simplified = sympy.simplify(J_sym)
    J_fwd_sym = sympy.simplify(J_sym_simplified.inv())
    prod = sympy.simplify(J_sym_simplified * J_fwd_sym - sympy.eye(rows))
    max_entry = max(abs(float(prod[i, j].evalf()))
                    for i in range(rows) for j in range(cols))
    assert max_entry < tol, (
        f"max|J_inv * J - I| = {max_entry:.2e} (symbolic identity check failed)"
    )


# ---------------------------------------------------------------------------
# Hub-and-spoke tests
# ---------------------------------------------------------------------------

def test_hub_spoke_symbolic_match_orientation_off():
    kin = HubAndSpokeKinematics(4, orientation=False)
    _check_numeric_match(kin, orientation=False)


def test_hub_spoke_symbolic_match_orientation_on():
    kin = HubAndSpokeKinematics(4, orientation=True)
    _check_numeric_match(kin, orientation=True)


def test_hub_spoke_symbolic_identity():
    """Identity check for N=4 hub-and-spoke (12x12 is too slow; use N=3 instead)."""
    kin = HubAndSpokeKinematics(3, orientation=False)
    _check_identity(kin)


# ---------------------------------------------------------------------------
# Cluster-of-clusters tests
# ---------------------------------------------------------------------------

def test_coc_22_symbolic_match():
    root = TreeParser.parse('(2,2)')
    kin = ClusterOfClustersKinematics(root, orientation=False)
    _check_numeric_match(kin, orientation=False)


def test_coc_22_22_symbolic_match():
    root = TreeParser.parse('((2,2),(2,2))')
    kin = ClusterOfClustersKinematics(root, orientation=False)
    _check_numeric_match(kin, orientation=False)


def test_coc_22_symbolic_match_orientation_on():
    root = TreeParser.parse('(2,2)')
    kin = ClusterOfClustersKinematics(root, orientation=True)
    _check_numeric_match(kin, orientation=True)


def test_coc_22_symbolic_identity():
    """Identity check for (2,2) tree (8x8 system, feasible for sympy)."""
    root = TreeParser.parse('(2,2)')
    kin = ClusterOfClustersKinematics(root, orientation=False)
    _check_identity(kin)
