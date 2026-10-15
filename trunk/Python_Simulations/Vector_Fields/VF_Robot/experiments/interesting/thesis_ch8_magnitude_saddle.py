"""
thesis_ch8_magnitude_saddle.py

Does descent on the sensed magnitude |v| reach a saddle? Thesis Chapter 8,
Section "Vector Fields as Scalar Landscapes".

Near a nondegenerate critical point |v|^2 = (p - p*)^T J^T J (p - p*) is
positive definite for every type, so every critical point, a saddle
included, is a minimum of |v|. This script tests that in closed loop with
the three-robot simulator of the critical-points paper (0.33 m equilateral
triangle, momentum alpha 0.7, stiction 0.025 m/s, 0.3 m/s cap, 10 Hz).

Primitives, run from the same random starts:
  magnitude descent  plane fit to the three robots' |v_i|, command -k grad|v|,
                     k = 1 (the vector-to-scalar primitive of thesis Ch. 5,
                     affine-fit form; defined here, not in src/)
  attraction         src.control.primitives.critical_point_plane_fitting
                     (thesis Ch. 6), as a reference

Fields (critical point at the origin):
  saddle1            canonical saddle (src/fields/environments/Saddle.py)
  anisotropic saddle linear, eigenvalues 1 and -1/3, axes rotated 30 deg
                     (defined here); its |v| is an elliptic cone
  vortex1            control; its |v| equals that of saddle1

Starts uniform in [-0.5, 0.5]^2, 150 steps (15 s), 1000 trials per field and
primitive, as in experiments/generate_table1_data.py.

Outputs: experiments/outputs/thesis_ch8/magnitude_saddle.json, magnitude_saddle.png

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/thesis_ch8_magnitude_saddle.py
"""
import io
import os
import sys
import json
import contextlib
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
import src.control.primitives as ocp
from src.fields.environments.Saddle import saddle1
from src.fields.environments.Vortex import vortex1

OUT = os.path.join(ROOT, 'experiments', 'outputs', 'thesis_ch8')
FORMATION = os.path.join(ROOT, 'config', 'formations', 'equilateral_default.yaml')
N_TRIALS = 1000
STEPS = 150
GAIN = 1.0
SEED = 20260930

_c, _s = math.cos(math.pi / 6), math.sin(math.pi / 6)
_R = np.array([[_c, -_s], [_s, _c]])
J_ANISO = _R @ np.diag([1.0, -1.0 / 3.0]) @ _R.T


def saddle_aniso(x, y):
    u, v = J_ANISO @ np.array([x, y])
    return u, v


def magnitude_descent(cluster):
    pos = np.array(cluster.get_robot_positions()).reshape(3, 2)
    z = np.linalg.norm(np.array(cluster.sample_field_at_robots()), axis=1)
    A = np.column_stack([pos, np.ones(3)])
    a, b, _ = np.linalg.solve(A, z)
    return -GAIN * a, -GAIN * b


def trial(field_func, primitive, x0, y0, keep_path=False):
    with contextlib.redirect_stdout(io.StringIO()):   # constructor prints per trial
        cluster = OmniCluster(FORMATION, AnalyticalField(field_func))
        cluster.reset(x_c=x0, y_c=y0)
    path = []
    for _ in range(STEPS):
        cluster.move(primitive)
        if keep_path:
            path.append(np.array(cluster.get_robot_positions()).reshape(3, 2).mean(axis=0))
    c = np.array(cluster.get_robot_positions()).reshape(3, 2).mean(axis=0)
    return float(np.linalg.norm(c)), path


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    starts = rng.uniform(-0.5, 0.5, size=(N_TRIALS, 2))
    fields = {'saddle1': saddle1, 'anisotropic saddle': saddle_aniso, 'vortex1': vortex1}
    prims = {'magnitude descent': magnitude_descent,
             'attraction': ocp.critical_point_plane_fitting}
    results, paths = {}, {}
    for fn, f in fields.items():
        for pn, p in prims.items():
            d = []
            for i, (x0, y0) in enumerate(starts):
                keep = fn == 'anisotropic saddle' and i < 10
                dist, path = trial(f, p, x0, y0, keep_path=keep)
                d.append(dist)
                if keep:
                    paths.setdefault(pn, []).append(np.array(path).tolist())
            d = np.array(d)
            results[f'{fn} | {pn}'] = {
                'median': float(np.median(d)), 'p95': float(np.percentile(d, 95)),
                'max': float(d.max()), 'within_0.05': float(np.mean(d < 0.05)),
                'within_0.10': float(np.mean(d < 0.10))}
            print(f'{fn:20} {pn:18} done')

    print(f"\n{N_TRIALS} trials per row, final centroid distance to the critical point (m)\n")
    print(f"{'field':20} {'primitive':18} {'median':>8} {'95th':>8} {'max':>8} {'<0.05 m':>8} {'<0.10 m':>8}")
    for k, r in results.items():
        fn, pn = k.split(' | ')
        print(f"{fn:20} {pn:18} {r['median']:>8.4f} {r['p95']:>8.4f} {r['max']:>8.4f} "
              f"{100*r['within_0.05']:>7.1f}% {100*r['within_0.10']:>7.1f}%")
    json.dump({'n_trials': N_TRIALS, 'steps': STEPS, 'gain': GAIN, 'seed': SEED,
               'J_aniso': J_ANISO.tolist(), 'results': results},
              open(os.path.join(OUT, 'magnitude_saddle.json'), 'w'), indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    g = np.linspace(-0.6, 0.6, 17)
    X, Y = np.meshgrid(g, g)
    U = J_ANISO[0, 0] * X + J_ANISO[0, 1] * Y
    V = J_ANISO[1, 0] * X + J_ANISO[1, 1] * Y
    ax.quiver(X, Y, U, V, color='0.75')
    for pn, col in (('magnitude descent', 'C0'), ('attraction', 'C3')):
        for j, pth in enumerate(paths[pn]):
            pth = np.array(pth)
            ax.plot(pth[:, 0], pth[:, 1], color=col, lw=1, label=pn if j == 0 else None)
            ax.plot(pth[0, 0], pth[0, 1], 'o', color=col, ms=3)
    ax.plot(0, 0, 'kx', ms=8)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'magnitude_saddle.png'), dpi=200)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
