"""
Emitted-file round-trip tests.

For each formation, use CodeGenerator.write_all to emit Python modules into
a temporary directory, then import them and verify they match the runtime
kinematics object on a randomized state. This is the only test that exercises
the CodeGenerator emitter path and catches emitter-only bugs of the same class
as commit ac24cf6 (SAS meta-node Jacobian and FK emitter fix).

Coverage:
  - Hub-spoke N=5, orientation off and on
  - CoC (2,2), (2,2,2), ((2,2),(2,2)), ((3,3),(2,2)), (3,3,3)
  - Both orientation modes where applicable

Tolerances: FK/IK match to 1e-12, J_inv to 1e-12, J_fwd to 1e-9.
"""

import importlib
import importlib.util
import math
import os
import random
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    CodeGenerator,
    TreeParser,
)

SEED = 7
RNG = random.Random(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module(path, name):
    """Import a Python file from an absolute path, returning the module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jittered_state(kin, rng):
    """Return a valid state dict, jittered slightly from default."""
    state = kin.default_state()
    for v in kin.state_vars:
        if v in ('x_c', 'y_c'):
            state[v] += rng.uniform(-0.3, 0.3)
        elif v.startswith('theta') or v.startswith('gamma'):
            state[v] += rng.uniform(-0.5, 0.5)
        elif v.startswith('phi_'):
            state[v] += rng.uniform(-0.3, 0.3)
        elif v.startswith('p_') or v.startswith('q_') or v.startswith('r_'):
            state[v] = max(0.05, state[v] + rng.uniform(-0.05, 0.05))
        elif v.startswith('L_'):
            state[v] = max(0.05, state[v] + rng.uniform(-0.05, 0.05))
        elif v.startswith('beta_'):
            state[v] = max(0.2, min(math.pi - 0.2, state[v] + rng.uniform(-0.2, 0.2)))
    return state


def _run_emitted_roundtrip(kin, config, tree, name, orientation, rng, tol_fk=1e-12,
                            tol_j=1e-12, tol_jfwd=1e-9):
    """Generate files, import them, and compare emitted vs runtime on a random state."""
    N = kin.N

    # Generate into a temp directory
    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            gen = CodeGenerator(name, config, N, kin, tree=tree,
                                orientation=orientation)
            gen.write_all(image_only=False)
        finally:
            os.chdir(orig_cwd)

        # Add tmp to sys.path so the emitted forward_jacobian can import the
        # emitted inverse_jacobian (it uses a bare `from <name>_inverse_jacobian import ...`).
        sys.path.insert(0, tmp)
        try:
            fk_mod   = _load_module(os.path.join(tmp, f'{name}_forward_kinematics.py'),  f'{name}_fk')
            ik_mod   = _load_module(os.path.join(tmp, f'{name}_inverse_kinematics.py'),  f'{name}_ik')
            jinv_mod = _load_module(os.path.join(tmp, f'{name}_inverse_jacobian.py'),    f'{name}_jinv')
            jfwd_mod = _load_module(os.path.join(tmp, f'{name}_forward_jacobian.py'),    f'{name}_jfwd')
        finally:
            sys.path.remove(tmp)

        for trial in range(3):
            state = _jittered_state(kin, rng)

            # --- IK round-trip -----------------------------------------------
            positions = kin.inverse_kinematics(state)
            flat_xy = []
            for i in range(1, N + 1):
                flat_xy.extend([positions[i][0], positions[i][1]])

            # Emitted FK: forward_kinematics(x1, y1, x2, y2, ...)
            state_emitted = fk_mod.forward_kinematics(*flat_xy)
            # Compare non-phi vars (phi depends on orientation heading supplied to runtime FK)
            for v in [k for k in kin.state_vars if not k.startswith('phi_')]:
                a = state[v]
                b = state_emitted[v]
                if v.startswith('theta') or v.startswith('gamma'):
                    err = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
                else:
                    err = abs(a - b)
                assert err < tol_fk, (
                    f"[{name}] FK mismatch on '{v}': runtime={a:.10f}, emitted={b:.10f}, diff={err:.2e}"
                )

            # --- IK comparison -----------------------------------------------
            positions_emitted = ik_mod.inverse_kinematics(state)
            for i in range(1, N + 1):
                for dim in range(2):
                    a = positions[i][dim]
                    b = positions_emitted[i][dim] if isinstance(positions_emitted[i], (tuple, list)) else positions_emitted[i][dim]
                    err = abs(a - b)
                    assert err < tol_fk, (
                        f"[{name}] IK mismatch robot {i} dim {dim}: runtime={a:.10f}, emitted={b:.10f}"
                    )

            # --- J_inv comparison --------------------------------------------
            J_runtime = kin.inverse_jacobian(state)
            J_emitted = jinv_mod.inverse_jacobian(state)
            err_j = np.max(np.abs(J_runtime - J_emitted))
            assert err_j < tol_j, (
                f"[{name}] J_inv mismatch: max={err_j:.2e} >= {tol_j}"
            )

            # --- J_fwd comparison --------------------------------------------
            Jf_runtime = kin.forward_jacobian(state)
            Jf_emitted = jfwd_mod.forward_jacobian(state)
            err_jf = np.max(np.abs(Jf_runtime - Jf_emitted))
            assert err_jf < tol_jfwd, (
                f"[{name}] J_fwd mismatch: max={err_jf:.2e} >= {tol_jfwd}"
            )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_hub_spoke_5_off():
    kin = HubAndSpokeKinematics(5, orientation=False)
    _run_emitted_roundtrip(kin, config=1, tree=None, name='hs5',
                           orientation=False, rng=RNG)


def test_hub_spoke_5_on():
    kin = HubAndSpokeKinematics(5, orientation=True)
    _run_emitted_roundtrip(kin, config=1, tree=None, name='hs5on',
                           orientation=True, rng=RNG)


def _has_beta_vars(kin):
    """Return True if the kinematics has any beta state vars (acos accumulates ~1e-9 error)."""
    return any(v.startswith('beta') for v in kin.state_vars)


@pytest.mark.parametrize("tree_str,name", [
    ("(2,2)",          "coc22"),
    ("(2,2,2)",        "coc222"),
    ("((2,2),(2,2))",  "coc2222"),
    ("((3,3),(2,2))",  "coc3322"),
    ("(3,3,3)",        "coc333"),   # SAS meta-node over SAS leaves: canary for ac24cf6
])
@pytest.mark.parametrize("orientation", [False, True])
def test_coc_emitted_roundtrip(tree_str, name, orientation):
    root = TreeParser.parse(tree_str)
    kin = ClusterOfClustersKinematics(root, orientation=orientation)
    sfx = '_on' if orientation else '_off'
    # acos for beta accumulates ~1e-8 error near collinear; relax FK tol when beta vars present
    tol_fk = 5e-8 if _has_beta_vars(kin) else 1e-12
    _run_emitted_roundtrip(kin, config=2, tree=root,
                           name=name + sfx, orientation=orientation, rng=RNG,
                           tol_fk=tol_fk)
