"""Check of the s1 tracker's across-trench Lyapunov claim (Draft_12, Section III-C).

Claim: near the trench, n_dot = -k sat(a_perp n) with a_perp = g_perp kappa_perp, so
V = n^2 / 2 decreases for n != 0 wherever kappa_perp > 0, and the restoring term is
lost only where kappa_perp vanishes. On the benchmark the trench is x = 0, n = x_c,
and kappa_perp = pi^4 A |sin(pi y)|, which vanishes on the isotropic line y = 0.

Noise-free, paper gains (mc.GAIN, mc.V_MAX, g_perp = 1). Starts at transverse offsets
n0 on both segments. Logs V per step during TRACK (before CAPTURE latches) and reports
  - steps on which V rose, and the largest rise relative to V0,
  - V at the end of TRACK relative to V0,
  - the same counts restricted to |y| > 0.05 (away from the kappa_perp = 0 line),
  - sign changes of n (overshoot across the trench).
Two modes. "sim" is the paper simulator (dt 0.1, momentum lag 0.7, stiction 0.025).
"ideal" sets dt 0.01, no lag, no stiction, so the cluster realizes p_c_dot as the
claim assumes. "step_only" and "lag_only" separate the two simulator effects.
Run with modes as arguments (default: all four).
"""
import os, sys, io, contextlib
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "experiments")); os.chdir(root)
import numpy as np
import main_separatrix_traverse as M
import experiments._mc_common as mc
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.control.pentagon_primitives import oecs_separatrix_step

STEPS = 500
YS = [0.40, 0.30, 0.15, 0.06, -0.06, -0.15, -0.30, -0.40]
N0S = [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10]
Y_AWAY = 0.05


MODES = {"sim": dict(timestep=0.1, momentum_alpha=0.7, stiction_threshold=0.025),
         "ideal": dict(timestep=0.01, momentum_alpha=0.0, stiction_threshold=0.0),
         "step_only": dict(timestep=0.1, momentum_alpha=0.0, stiction_threshold=0.0),
         "lag_only": dict(timestep=0.01, momentum_alpha=0.7, stiction_threshold=0.025)}


def run(sx, sy, mode):
    kw = MODES[mode]
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(mc.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static), **kw)
    cl.reset(sx, sy)
    prim = lambda c: tuple(mc.GAIN * v for v in oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=1.0))
    for _ in range(int(round(STEPS * 0.1 / kw["timestep"]))):
        cl.move(prim)
        if getattr(cl, '_oecs_captured', False):
            break
    h = np.asarray(cl.get_center_history())
    return h


def main(mode):
    print(f"\n=== mode {mode} {MODES[mode]} ===")
    print(f"gain {mc.GAIN}, v_max {mc.V_MAX}, formation {mc.FORMATION_CONFIG}, noise-free")
    print(f"{'start':>15} {'tr.steps':>8} {'V rises':>8} {'max rise/V0':>11} "
          f"{'rises |y|>.05':>13} {'V_end/V0':>9} {'|n| end':>8} {'y end':>7} {'flips':>5}")
    tot_rise = tot_rise_away = tot_steps = tot_flips = 0
    worst_away = 0.0
    for sy in YS:
        for n0 in N0S:
            h = run(n0, sy, mode)
            n = h[:, 0]; y = h[:, 1]
            V = 0.5 * n ** 2; V0 = V[0]
            dV = np.diff(V)
            rise = dV > 1e-12 * V0
            away = np.abs(y[1:]) > Y_AWAY
            r_away = rise & away
            max_rise = float(dV[rise].max() / V0) if rise.any() else 0.0
            if r_away.any():
                worst_away = max(worst_away, float(dV[r_away].max() / V0))
            flips = int(np.sum(np.sign(n[1:]) * np.sign(n[:-1]) < 0))
            tot_rise += int(rise.sum()); tot_rise_away += int(r_away.sum()); tot_steps += len(dV)
            tot_flips += flips
            print(f"({n0:+.2f},{sy:+.2f}) {len(dV):8d} {int(rise.sum()):8d} {max_rise:11.2e} "
                  f"{int(r_away.sum()):13d} {V[-1]/V0:9.2e} {abs(n[-1]):8.1e} {y[-1]:+7.3f} {flips:5d}")
    print(f"\nTRACK steps {tot_steps}, V rose on {tot_rise}, of which {tot_rise_away} with |y| > {Y_AWAY}")
    print(f"largest rise with |y| > {Y_AWAY}: {worst_away:.2e} of V0; sign flips of n: {tot_flips}")


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["sim", "ideal", "step_only", "lag_only"]):
        main(m)
