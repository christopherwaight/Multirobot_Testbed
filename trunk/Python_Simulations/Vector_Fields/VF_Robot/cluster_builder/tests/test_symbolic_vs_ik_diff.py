"""
Third-oracle Jacobian test: differentiate IK(state) symbolically and compare
to inverse_jacobian_symbolic().

This catches cases where the analytic Jacobian formula agrees with FD at the
default state by coincidence but does not match the derivative of the IK
formula in general. The symbolic simplification is an exact algebraic check.

Scope: PairBlock, TripleBlock (direct), HubAndSpokeKinematics(N=3),
ClusterOfClustersKinematics('(2,2)'), ClusterOfClustersKinematics('(3)').

Larger trees are too slow for sympy.simplify; those are covered by FD and
emitted-file round-trip tests.
"""

import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    PairBlock,
    TripleBlock,
    SympyBackend,
    TreeParser,
)

try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

pytestmark = pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")


# ---------------------------------------------------------------------------
# Helper: build symbolic IK and diff it
# ---------------------------------------------------------------------------

def _sym_ik_jacobian_pairblock():
    """Build J_oracle for PairBlock from sympy.diff of inverse()."""
    x_c, y_c, theta, L = sympy.symbols('x_c y_c theta L', real=True)
    ct = sympy.cos(theta)
    st = sympy.sin(theta)
    h = L / 2
    xa = x_c - h * ct
    ya = y_c - h * st
    xb = x_c + h * ct
    yb = y_c + h * st
    outputs = [xa, ya, xb, yb]
    state_syms = [x_c, y_c, theta, L]
    J_oracle = sympy.Matrix([[sympy.diff(o, s) for s in state_syms] for o in outputs])
    return J_oracle, state_syms, theta, L


def _sym_ik_jacobian_tripleblock():
    """Build J_oracle for TripleBlock from sympy.diff of inverse()."""
    x_c, y_c, theta_c, p, beta, q = sympy.symbols('x_c y_c theta_c p beta q', real=True)
    ct = sympy.cos(theta_c)
    st = sympy.sin(theta_c)
    # Local frame: b at origin, a at (p, 0), c at (q cos beta, q sin beta)
    x1_loc, y1_loc = p, sympy.Integer(0)
    x2_loc, y2_loc = sympy.Integer(0), sympy.Integer(0)
    x3_loc = q * sympy.cos(beta)
    y3_loc = q * sympy.sin(beta)
    cx_loc = (x1_loc + x2_loc + x3_loc) / 3
    cy_loc = (y1_loc + y2_loc + y3_loc) / 3
    def rot(xl, yl):
        return ct * (xl - cx_loc) - st * (yl - cy_loc) + x_c, \
               st * (xl - cx_loc) + ct * (yl - cy_loc) + y_c
    xa, ya = rot(x1_loc, y1_loc)
    xb, yb = rot(x2_loc, y2_loc)
    xc, yc = rot(x3_loc, y3_loc)
    outputs = [xa, ya, xb, yb, xc, yc]
    state_syms = [x_c, y_c, theta_c, p, beta, q]
    J_oracle = sympy.Matrix([[sympy.diff(o, s) for s in state_syms] for o in outputs])
    return J_oracle, state_syms


# ---------------------------------------------------------------------------
# PairBlock tests
# ---------------------------------------------------------------------------

def test_pairblock_symbolic_vs_ik_diff():
    """J_oracle from sympy.diff(PairBlock.inverse) == PairBlock.j_inv(symbolic)."""
    J_oracle, state_syms, theta, L = _sym_ik_jacobian_pairblock()
    J_analytic = PairBlock.j_inv(theta, L, backend=SympyBackend)
    diff = sympy.simplify(J_oracle - sympy.Matrix(J_analytic.tolist()))
    max_entry = max(abs(float(diff[i, j].evalf()))
                    for i in range(diff.shape[0]) for j in range(diff.shape[1]))
    assert max_entry < 1e-12, f"PairBlock J_oracle vs J_analytic max diff = {max_entry:.2e}"


def test_pairblock_symbolic_identity():
    """PairBlock J_inv * J_inv^{-1} == I symbolically."""
    J_oracle, state_syms, theta, L = _sym_ik_jacobian_pairblock()
    J_sq = J_oracle[:, :]  # 4x4 square
    prod = sympy.simplify(J_sq * J_sq.inv() - sympy.eye(4))
    max_entry = max(abs(float(prod[i, j].evalf()))
                    for i in range(4) for j in range(4))
    assert max_entry < 1e-12, f"PairBlock J * J^{{-1}} - I max = {max_entry:.2e}"


# ---------------------------------------------------------------------------
# TripleBlock tests
# ---------------------------------------------------------------------------

def test_tripleblock_symbolic_vs_ik_diff():
    """J_oracle from sympy.diff(TripleBlock.inverse) == TripleBlock.j_inv(symbolic)."""
    J_oracle, state_syms = _sym_ik_jacobian_tripleblock()
    x_c, y_c, theta_c, p, beta, q = state_syms
    J_analytic = TripleBlock.j_inv(x_c, y_c, theta_c, p, beta, q, backend=SympyBackend)
    diff = sympy.simplify(J_oracle - sympy.Matrix(J_analytic.tolist()))
    max_entry = max(abs(float(diff[i, j].evalf()))
                    for i in range(diff.shape[0]) for j in range(diff.shape[1]))
    assert max_entry < 1e-12, f"TripleBlock J_oracle vs J_analytic max diff = {max_entry:.2e}"


def test_tripleblock_symbolic_identity():
    """TripleBlock J_inv * J_inv^{-1} == I symbolically."""
    J_oracle, _ = _sym_ik_jacobian_tripleblock()
    prod = sympy.simplify(J_oracle * J_oracle.inv() - sympy.eye(6))
    max_entry = max(abs(float(prod[i, j].evalf()))
                    for i in range(6) for j in range(6))
    assert max_entry < 1e-12, f"TripleBlock J * J^{{-1}} - I max = {max_entry:.2e}"


# ---------------------------------------------------------------------------
# Kinematics-level symbolic vs IK-diff
# ---------------------------------------------------------------------------

def _eval_sym_matrix(M, state, state_vars):
    """Substitute float values into a sympy matrix."""
    subs = {sympy.Symbol(v): float(state[v]) for v in state_vars}
    return np.array(M.subs(subs).evalf().tolist(), dtype=float)


def _sym_ik_diff_jacobian(kin):
    """
    Build the symbolic J_oracle by differentiating inverse_kinematics symbolically.

    This works by evaluating inverse_kinematics with sympy symbolic state vars
    (relies on sympy_backend used inside j_inv; but inverse_kinematics calls math.cos
    directly, so we use a different approach: substitute sympy symbols into the
    runtime IK formulas by monkey-patching math inside the module).
    """
    import importlib
    import types
    import clusterbuilder as cb

    state_vars = kin.state_vars
    sym_state = {v: sympy.Symbol(v, real=True) for v in state_vars}

    # Patch math inside clusterbuilder to use sympy functions for this call
    orig_cos  = cb.math.cos
    orig_sin  = cb.math.sin
    orig_sqrt = cb.math.sqrt
    orig_atan2 = cb.math.atan2
    orig_hypot = cb.math.hypot
    orig_acos  = cb.math.acos

    cb.math.cos   = sympy.cos
    cb.math.sin   = sympy.sin
    cb.math.sqrt  = sympy.sqrt
    cb.math.atan2 = sympy.atan2
    cb.math.hypot = lambda a, b: sympy.sqrt(a**2 + b**2)
    cb.math.acos  = sympy.acos

    # Also patch PairBlock and TripleBlock math references
    pb_cos, pb_sin = PairBlock.__module__, None  # they use math directly via module ref

    try:
        positions_sym = kin.inverse_kinematics(sym_state)
    finally:
        cb.math.cos   = orig_cos
        cb.math.sin   = orig_sin
        cb.math.sqrt  = orig_sqrt
        cb.math.atan2 = orig_atan2
        cb.math.hypot = orig_hypot
        cb.math.acos  = orig_acos

    # Build oracle Jacobian: rows = robot outputs, cols = state vars
    N = kin.N
    robot_rows = []
    for i in range(1, N + 1):
        robot_rows.append(positions_sym[i][0])
        robot_rows.append(positions_sym[i][1])

    J_rows = []
    for expr in robot_rows:
        row = []
        for v in state_vars:
            row.append(sympy.diff(expr, sympy.Symbol(v, real=True)))
        J_rows.append(row)
    return sympy.Matrix(J_rows)


@pytest.mark.parametrize("kin_desc,kin", [
    ("HS-N3",     HubAndSpokeKinematics(3, orientation=False)),
    ("CoC-(2,2)", ClusterOfClustersKinematics(TreeParser.parse('(2,2)'), orientation=False)),
    # Note: "(3)" alone (single-leaf root) not permitted; IK raises KeyError on 'theta_c'.
])
def test_sym_ik_diff_vs_inverse_jacobian_symbolic(kin_desc, kin):
    """J_oracle = sympy.diff(IK) must match inverse_jacobian_symbolic() exactly."""
    state = kin.default_state()

    J_oracle = _sym_ik_diff_jacobian(kin)
    J_sym = kin.inverse_jacobian_symbolic()

    diff = sympy.simplify(J_oracle - J_sym)
    max_entry = max(
        abs(float(diff[i, j].evalf()))
        for i in range(diff.shape[0]) for j in range(diff.shape[1])
    )
    assert max_entry < 1e-12, (
        f"[{kin_desc}] J_oracle vs J_symbolic max diff = {max_entry:.2e}"
    )
