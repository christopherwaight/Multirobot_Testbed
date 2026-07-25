"""
traverse_rotated_check.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  the tilted-eigenframe generality check for the objective separatrix
          traverser (experiments/outputs/oecs/traverse_rotated_check.csv,
          figures/traverse_rotated_check.png).

EXPERIMENT
  On the STANDARD double gyre the shear strain vanishes identically, so the
  strain eigenvectors e1, e2 sit exactly on the x/y coordinate axes -- a
  nongeneric special case that could hide a sign or projection bug in
  Primitive 11 (oecs_separatrix_step) that only shows up once the eigenframe
  is tilted off the axes. This script runs the traverser on the SAME field
  rotated by a fixed angle theta = 30 deg (make_rotated_field:
  v_theta(p) = R_theta v(R_theta^T p), a static Euclidean change of
  variables, not a time-dependent observer, so there is no swirl term and
  every analytic quantity of the base field carries over exactly under the
  same rotation).

  The six starts and their expected saddle targets are the SAME as
  main_separatrix_traverse.py, rotated by theta. Success = the traverser
  captures near a rotated saddle location (including periodic tile images
  of the saddle, since the base field tiles the plane).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/traverse_rotated_check.py
"""
import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP)
from src.fields.environments.Rotating_Frame import make_rotated_field
from src.control.pentagon_primitives import oecs_separatrix_step

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
V_MAX, GAIN = 0.04, 3.0
G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.05, 0.05, 0.15
SIM_STEPS = 700
THETA_DEG = 30.0

STARTS = [
    ("S1", -0.45,  0.30), ("S2",  0.05,  0.40), ("S3",  0.00,  0.00),
    ("S4",  0.10, -0.20), ("S5",  0.25,  0.42), ("S6", -0.20, -0.30),
]
COLORS = ["#2a78d6", "#1baf7a", "#4a3aa7", "#e34948", "#eb6834", "#e87ba4"]

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "oecs")
os.makedirs(OUT_DIR, exist_ok=True)


def nearest_saddle_image(point, Q, saddles, n_tiles=3):
    """
    Distance from `point` to the nearest rotated, tile-periodic image of a
    saddle. u,v depend on sin(pi x_f)/cos(pi x_f) with x_f = x + 1, and the
    saddle condition sin(pi x_f) = 0 recurs every unit shift in x_f, i.e.
    every unit shift in world x: the field's wall/saddle lines sit at every
    integer world x, so the tile period is 1, not 2. A captured point can
    legitimately land on any such shifted saddle, not just the two saddles
    of the central tile.
    """
    best = np.inf
    best_world = None
    for sad in saddles:
        for shift in range(-n_tiles, n_tiles + 1):
            world = np.array(sad) + np.array([1.0 * shift, 0.0])
            rot = Q @ world
            d = np.hypot(point[0] - rot[0], point[1] - rot[1])
            if d < best:
                best = d
                best_world = world
    return best, best_world


def main():
    theta = np.deg2rad(THETA_DEG)
    rotated_field_fn = make_rotated_field(double_gyre_static, theta)
    c, s = np.cos(theta), np.sin(theta)
    Q = np.array([[c, -s], [s, c]])
    saddles = [SADDLE_BOTTOM, SADDLE_TOP]

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    for sad in saddles:
        rot = Q @ np.array(sad)
        ax.plot(*rot, marker="x", color="k", ms=9, mew=2, zorder=6)
    rot_sep_pts = np.array([Q @ np.array([0.0, y])
                            for y in np.linspace(-0.5, 0.5, 50)])
    ax.plot(rot_sep_pts[:, 0], rot_sep_pts[:, 1], color="0.5", ls="--",
           lw=1.2, label="rotated separatrix")

    rows = []
    for (name, sx0, sy0), col in zip(STARTS, COLORS):
        sx, sy = Q @ np.array([sx0, sy0])
        field = AnalyticalField(rotated_field_fn)
        cl = PentagonCluster(FORMATION_CONFIG, field)
        cl.reset(sx, sy)

        def prim(cluster):
            vx, vy = oecs_separatrix_step(cluster, v_max=V_MAX, g_perp=G_PERP,
                                          s_trim=S_TRIM, r_band=R_BAND,
                                          g_capture=G_CAPTURE, s_capture=None)
            return vx * GAIN, vy * GAIN

        for _ in range(SIM_STEPS):
            cl.move(prim)
        hist = cl.get_center_history()
        final = hist[-1]
        d_near, world_near = nearest_saddle_image(final, Q, saddles)
        last_mode = cl.diagnostics[-1]['mode']

        ax.plot(hist[:, 0], hist[:, 1], color=col, lw=1.5, zorder=5)
        ax.plot(sx, sy, marker="o", color=col, ms=6, mec="k", mew=0.8, zorder=7)
        ax.plot(final[0], final[1], marker="s", color=col, ms=6, mec="k",
               mew=0.8, zorder=7)
        ax.annotate(name, (sx, sy), textcoords="offset points",
                   xytext=(6, 5), fontsize=8, color=col)

        rows.append({
            'run': name, 'start_x': sx, 'start_y': sy,
            'final_x': float(final[0]), 'final_y': float(final[1]),
            'last_mode': last_mode, 'dist_to_nearest_saddle_image': d_near,
            'nearest_saddle_world': tuple(world_near),
            'captured': int(d_near < 0.05),
        })
        print(f"{name}: start=({sx:+.3f},{sy:+.3f}) final=({final[0]:+.4f},"
              f"{final[1]:+.4f}) dist_to_nearest_saddle_image={d_near:.4f} "
              f"(world saddle {tuple(world_near)}) last_mode={last_mode}")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Objective traverser on the double gyre rotated by "
               f"{THETA_DEG:.0f} deg\n(tilted-eigenframe code-generality check)",
               fontsize=11)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "traverse_rotated_check.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_png}")

    csv_path = os.path.join(OUT_DIR, "traverse_rotated_check.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    n_captured = sum(r['captured'] for r in rows)
    print(f"\nCaptured near a (possibly tile-shifted) rotated saddle: "
         f"{n_captured}/{len(rows)}")


if __name__ == "__main__":
    main()
