"""
thesis_ch5_orbit_drift_check.py

Which robot model do the 2025 (IDETC) vector-sum orbits match? Thesis Chapter 5
drift and Section 2.6 dead zone. Analysis of logged data only; no simulation.

The vector-sum primitive commands the cluster along the local flow, so on a
fixed vortex it always moves tangent to the circle it is on. A cluster that
answers a new heading after a total delay T drifts outward at

    d(r^2)/dt = 2 T V^2,

with V the centroid speed (revision paper, limit-cycle derivation, with no
radial loop). So r^2 should grow linearly, and the slope gives T:

    T = slope / (2 mean(V)^2).

Compared with
    refit (2026-11-16): T = L + tau + dt/2 = 0.055 + 0.02 + 0.05 = 0.125 s
    earlier momentum model (tau 0.28 s): T = tau + dt/2 = 0.33 s

Data: trunk/robots_3/Results_for TestBed_Paper/Fixed_vortex_Vector_Sum/run*.mat
(cluster_pose: centroid x, y in columns 0 and 1; tout: time). Runs 1 to 10, the
ten runs of the IDETC paper. The map center is
taken as the origin; a shared center fitted to all runs is checked as well.
Transient: samples before 1 s are dropped.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/thesis_ch5_orbit_drift_check.py
"""
import re
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

warnings.filterwarnings('ignore')
TRUNK = Path(__file__).resolve().parents[4]
DATA = TRUNK / 'robots_3' / 'Results_for TestBed_Paper' / 'Fixed_vortex_Vector_Sum'
OUT = Path(__file__).resolve().parent / 'outputs' / 'thesis_ch5'
T_REFIT, T_MOMENTUM = 0.125, 0.33
T_SKIP = 1.0


def load():
    runs = []
    for f in sorted(DATA.glob('run*.mat'), key=lambda p: p.name):
        m = re.fullmatch(r'run(\d+)\.mat', f.name)
        if not m or int(m.group(1)) > 10:
            continue   # the IDETC set is runs 1 to 10; run12.mat and the misnamed run_11.mat are not in it
        d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        runs.append((f.name, np.atleast_2d(d['cluster_pose'])[:, :2], np.atleast_1d(d['tout'])))
    return runs


def fit(xy, t, center):
    r2 = np.sum((xy - center) ** 2, axis=1)
    k = t >= T_SKIP
    A = np.column_stack([t[k], np.ones(k.sum())])
    (slope, c), *_ = np.linalg.lstsq(A, r2[k], rcond=None)
    resid = r2[k] - A @ np.array([slope, c])
    v = np.hypot(*np.diff(xy, axis=0).T) / np.diff(t)
    vbar = v[t[:-1] >= T_SKIP].mean()
    return slope, vbar, 1 - resid.var() / r2[k].var(), float(np.sqrt(r2[0])), float(np.sqrt(r2[-1]))


def main():
    runs = load()
    lines = []
    out = lambda s: (print(s), lines.append(s))
    out(f'{len(runs)} fixed-vortex vector-sum runs, centers at the origin')
    out(f'{"run":9}{"r0 (m)":>8}{"rEnd (m)":>10}{"slope":>8}{"V (m/s)":>9}{"T (s)":>8}{"R^2":>6}')
    Ts = []
    for name, xy, t in runs:
        s, v, r2, r0, r1 = fit(xy, t, np.zeros(2))
        Ts.append(s / (2 * v ** 2))
        out(f'{name:9}{r0:8.3f}{r1:10.3f}{s:8.4f}{v:9.3f}{Ts[-1]:8.3f}{r2:6.2f}')
    Ts = np.array(Ts)
    out(f'T: median {np.median(Ts):.3f} s, mean {Ts.mean():.3f} s, '
        f'quartiles {np.percentile(Ts, 25):.3f} to {np.percentile(Ts, 75):.3f} s, '
        f'range {Ts.min():.3f} to {Ts.max():.3f} s')
    out(f'refit model T = {T_REFIT} s; earlier momentum model T = {T_MOMENTUM} s')

    # Sensitivity to the map center: grid search for the shared center that
    # makes r^2 most linear in time, summed over runs.
    best = None
    for cx in np.arange(-0.10, 0.101, 0.02):
        for cy in np.arange(-0.10, 0.101, 0.02):
            tot = sum(1 - fit(xy, t, np.array([cx, cy]))[2] for _, xy, t in runs)
            if best is None or tot < best[0]:
                best = (tot, cx, cy)
    Tc = np.array([fit(xy, t, np.array(best[1:]))[0] / (2 * fit(xy, t, np.array(best[1:]))[1] ** 2)
                   for _, xy, t in runs])
    out(f'shared center {best[1]:+.2f}, {best[2]:+.2f} m (grid, 0.02 m): T median {np.median(Tc):.3f} s')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'orbit_drift_check.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
