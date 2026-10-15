"""
traverse_flowband_shrink.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  the flow-band-shrink table for the objective separatrix traverser
          (experiments/outputs/oecs/flowband_shrink.csv).

EXPERIMENT
  Primitive 11 (oecs_separatrix_step) has a CROSS mode that is the one place
  the controller falls back to riding the last tangent by pure continuity
  when BOTH the strain magnitude r is small (eigenframe unreliable) AND the
  ambient flow is also too weak to steer by (r_band is the threshold on r).
  Chris asked: since TRACK's flow-selected tangent already carries the ride
  through the origin isotropic point on the double gyre (the flow there
  does not vanish, only the S-eigenframe does), does the controller ever
  actually need CROSS on this field? Test by shrinking r_band toward zero
  and checking whether CROSS still fires and whether the six starts still
  reach and hold their nearest saddle.

  Expected / found result: CROSS never fires on the double gyre at ANY
  r_band down to and including 0.0, and all six starts still capture
  correctly. The flow-band shrinks to zero with no loss of function: the
  isotropic-point crossing is handled entirely by TRACK's ordinary
  flow-selected tangent, not by a dedicated fallback mode.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/traverse_flowband_shrink.py
"""
import os
import sys
import csv
from collections import Counter

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import oecs_separatrix_step

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
V_MAX, GAIN = 0.04, 3.0
G_PERP, S_TRIM, G_CAPTURE = 1.0, 0.05, 0.15
SIM_STEPS = 700

STARTS = [
    ("S1", -0.45,  0.30), ("S2",  0.05,  0.40), ("S3",  0.00,  0.00),
    ("S4",  0.10, -0.20), ("S5",  0.25,  0.42), ("S6", -0.20, -0.30),
]
R_BANDS = [0.05, 0.02, 0.01, 0.005, 0.001, 0.0]

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "oecs")
os.makedirs(OUT_DIR, exist_ok=True)


def run(sx, sy, r_band):
    field = AnalyticalField(double_gyre_static)
    cl = PentagonCluster(FORMATION_CONFIG, field)
    cl.reset(sx, sy)

    def prim(c):
        vx, vy = oecs_separatrix_step(c, v_max=V_MAX, g_perp=G_PERP,
                                      s_trim=S_TRIM, r_band=r_band,
                                      g_capture=G_CAPTURE, s_capture=None)
        return vx * GAIN, vy * GAIN

    for _ in range(SIM_STEPS):
        cl.move(prim)
    return cl


def main():
    rows = []
    print(f"{'r_band':>8}  {'run':>4}  {'cross_frac':>10}  {'final_x':>8}  "
          f"{'final_y':>8}  {'captured':>8}")
    for r_band in R_BANDS:
        for name, sx, sy in STARTS:
            cl = run(sx, sy, r_band)
            modes = Counter(d['mode'] for d in cl.diagnostics)
            tot = sum(modes.values())
            cross_frac = modes.get('CROSS', 0) / max(tot, 1)
            cx, cy = cl.get_centroid()
            captured = abs(cx) < 0.05 and abs(abs(cy) - 0.5) < 0.05
            rows.append({
                'r_band': r_band, 'run': name, 'start_x': sx, 'start_y': sy,
                'cross_count': modes.get('CROSS', 0), 'cross_frac': cross_frac,
                'final_x': float(cx), 'final_y': float(cy),
                'captured': int(captured),
            })
            print(f"{r_band:8.3f}  {name:>4}  {cross_frac:10.4f}  "
                  f"{cx:8.4f}  {cy:8.4f}  {str(captured):>8}")

    csv_path = os.path.join(OUT_DIR, "flowband_shrink.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    print("\nSummary by r_band:")
    for r_band in R_BANDS:
        sub = [r for r in rows if r['r_band'] == r_band]
        n_cross = sum(r['cross_count'] for r in sub)
        n_captured = sum(r['captured'] for r in sub)
        print(f"  r_band={r_band:6.3f}: total CROSS activations across "
              f"{len(sub)} starts = {n_cross:4d}, captured {n_captured}/{len(sub)}")


if __name__ == "__main__":
    main()
