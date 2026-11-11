"""
trench_offset.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13.tex (proposed
          subsection in Draft_13_unsteady_dg_proposed.tex; not in the paper).
  Makes:  field-only diagnostics for the time-periodic double gyre: how far the
          true D and s1 trenches sit from the instantaneous separatrix, the
          shear strain and vorticity on the separatrix, and the separatrix speed.
  Reads:  nothing. No robots, no fit; analytic field with central differences.

Run (from VF_Robot):
  venv/bin/python3 experiments/unsteady_dg/trench_offset.py
"""
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.fields.environments.Double_Gyre import double_gyre_static

OMEGA = 2 * np.pi / 10
H = 1e-5


def jac(x, y, t, cfg):
    up = double_gyre_static(x + H, y, t, cfg); um = double_gyre_static(x - H, y, t, cfg)
    vp = double_gyre_static(x, y + H, t, cfg); vm = double_gyre_static(x, y - H, t, cfg)
    ux, vx = (up[0] - um[0]) / (2 * H), (up[1] - um[1]) / (2 * H)
    uy, vy = (vp[0] - vm[0]) / (2 * H), (vp[1] - vm[1]) / (2 * H)
    return ux, uy, vx, vy


def D(x, y, t, cfg):
    ux, uy, vx, vy = jac(x, y, t, cfg)
    return ux * vy - uy * vx


def s1(x, y, t, cfg):
    ux, uy, vx, vy = jac(x, y, t, cfg)
    mu = 0.5 * (ux + vy)
    return mu - np.hypot(0.5 * (ux - vy), 0.5 * (uy + vx))


def sep_x(t, eps):
    a = eps * np.sin(OMEGA * t)
    b = 1.0 - 2.0 * a
    return 2.0 / (b + np.sqrt(b * b + 4.0 * a)) - 1.0


def trench_x(fun, y, t, cfg, xs, half=0.12, n=2401):
    xx = np.linspace(xs - half, xs + half, n)
    vals = np.array([fun(x, y, t, cfg) for x in xx])
    return xx[int(np.argmin(vals))]


def main():
    ys = np.round(np.arange(-0.45, 0.46, 0.05), 2)
    for eps in (0.1, 0.25):
        cfg = {"A": 0.1, "eps": eps, "omega": OMEGA}
        tt = np.linspace(0, 10, 2001)
        xsv = np.array([sep_x(t, eps) for t in tt])
        speed = np.max(np.abs(np.gradient(xsv, tt)))
        print(f"\neps = {eps}: separatrix x range [{xsv.min():+.4f}, {xsv.max():+.4f}], "
              f"max |dx_s/dt| = {speed:.4f}")
        worst_D = worst_s = 0.0
        for phase in (np.pi / 2, -np.pi / 2):          # sin(omega t) = +1, -1
            t = phase / OMEGA % 10
            xs = sep_x(t, eps)
            offD, offS, shear, vort = [], [], [], []
            for y in ys:
                offD.append(trench_x(D, y, t, cfg, xs) - xs)
                offS.append(trench_x(s1, y, t, cfg, xs) - xs)
                ux, uy, vx, vy = jac(xs, y, t, cfg)
                shear.append(0.5 * (uy + vx)); vort.append(vx - uy)
            offD, offS = np.array(offD), np.array(offS)
            # exclude the crest region |y| < 0.1, where D's trench flattens
            m = np.abs(ys) >= 0.1
            worst_D = max(worst_D, np.abs(offD[m]).max())
            worst_s = max(worst_s, np.abs(offS[m]).max())
            print(f"  sin(wt) = {np.sin(OMEGA * t):+.0f}: x_s = {xs:+.4f}")
            print(f"    D trench offset   (|y|>=0.1): max {np.abs(offD[m]).max():.4f}, "
                  f"mean {np.abs(offD[m]).mean():.4f}")
            print(f"    s1 trench offset  (|y|>=0.1): max {np.abs(offS[m]).max():.4f}, "
                  f"mean {np.abs(offS[m]).mean():.4f}")
            print(f"    on separatrix: max |shear| {np.max(np.abs(shear)):.4f}, "
                  f"max |vorticity| {np.max(np.abs(vort)):.4f}")
            for y, a, b in zip(ys, offD, offS):
                if y in (-0.4, -0.25, 0.25, 0.4):
                    print(f"      y={y:+.2f}: D off {a:+.4f}, s1 off {b:+.4f}")
        print(f"  worst over both extremes: D {worst_D:.4f}, s1 {worst_s:.4f} "
              f"(rho = 0.075, acquisition band 0.05)")


if __name__ == "__main__":
    main()
