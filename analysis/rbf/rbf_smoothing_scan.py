"""Wider smoothing scan for the thin-plate-spline maps: pointwise validation error and the roughness the
controller sees.

rbf_fields.py chose the smoothing by the median error of single validation readings, which reading noise
dominates. The controller instead uses a plane fit over a 0.30 m triangle, which amplifies cell-scale
speckle. For each smoothing (same for hue and saturation) this reports:
  dir_deg, sat     median error on the validation runs (training runs with run % 5 == 1)
  tri_med          median centre-estimate error of a 0.30 m triangle (positions within 0.4 m)
  rough            median change in that estimate between neighbouring positions 2 cm apart (speckle)
and the same for the 2026 net. Writes outputs/rbf_smoothing_scan.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rbf_fields as rf  # noqa: E402
from src.fields.field_types import NNField  # noqa: E402

LAMS = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def triangle_map(model, field, rmax=0.40, step=0.02):
    g = np.arange(-rmax, rmax + 1e-9, step)
    s, (pts, err) = rf.triangle_estimates(model, field, rmax=rmax, step=step)
    img = np.full((len(g), len(g)), np.nan)
    ij = np.rint((pts + rmax) / step).astype(int)
    img[ij[:, 1], ij[:, 0]] = err
    rough = np.nanmedian(np.abs(np.r_[np.diff(img, axis=0).ravel(), np.diff(img, axis=1).ravel()]))
    return s["median_err_all"], float(rough)


def main():
    rows = []
    for field in ("vortex", "saddle"):
        d = rf.prepare(field)
        X, Yh, Ys = rf.cells(d, d["trn26"] & ~d["val26"])
        Pv, Hv, Sv = d["Q26"][d["val26"]], d["H26"][d["val26"]], d["S26"][d["val26"]]
        cands = [("nn", {"nn": NNField(os.path.join(rf.paths.VF, rf.NETDIR[field]))})]
        cands += [(f"rbf {lam:g}", {"hue": rf.fit(X, Yh, lam), "sat": rf.fit(X, Ys, lam)}) for lam in LAMS]
        for tag, m in cands:
            h, s = rf.predict(m, Pv)
            tri, rough = triangle_map(m, field)
            r = dict(field=field, model=tag, dir_deg=float(np.median(np.abs(rf.SAD.dang(h, Hv)))),
                     sat=float(np.median(np.abs(s - Sv))), tri_med_cm=100 * tri, rough_cm=100 * rough)
            rows.append(r)
            print(f"{field:6s} {tag:10s} validation {r['dir_deg']:.3f} deg, sat {r['sat']:.4f} | "
                  f"triangle error {r['tri_med_cm']:.2f} cm, speckle {r['rough_cm']:.2f} cm", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(rf.OUT, "rbf_smoothing_scan.csv"), index=False)


if __name__ == "__main__":
    main()
