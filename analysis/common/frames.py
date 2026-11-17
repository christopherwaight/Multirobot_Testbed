"""Map hardware positions (relative to the session's p*) into the 2026 reconstructed fields' frames.

2026 runs: the transform found when each net was built (build_report.json, align_2026_to_2025),
q = R(theta) M k (p - c26) + t - c, with M = diag(1, -1) when mirrored.
2025 sessions: rotation and mirror registered from that session's own readings against the net
(hue match over a 0.08 to 0.6 m ring), translation fixed by the session's fitted p* (which the net
also puts at its origin).
"""
import json
import os

import numpy as np

import paths
from centres import decode  # noqa: F401  (kept for callers)

NETDIR = {"vortex": "vortex_predictors_2026", "saddle": "saddle_predictors_2026"}


def rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class Frame:
    """q = A p + b for points p given relative to p* (rows of P @ A.T + b); `mirror` flips orbit sense."""

    def __init__(self, A, b, mirror):
        self.A, self.b, self.mirror = np.asarray(A, float), np.asarray(b, float), bool(mirror)

    def __call__(self, P):
        return np.asarray(P, float) @ self.A.T + self.b


def frame_2026(field, p_star):
    r = json.load(open(os.path.join(paths.VF, NETDIR[field], "build_report.json")))
    a = r["align_2026_to_2025"]
    M = np.diag([1.0, -1.0]) if a["mirror"] else np.eye(2)
    A = rot(a["rotation_deg"]) @ M * a.get("scale", 1.0)
    c26 = np.asarray(a["centre_2026_motive"])
    b = A @ (np.asarray(p_star) - c26) + np.asarray(a["translation"]) - np.asarray(r["critical_point_in_old_frame"])
    return Frame(A, b, a["mirror"])


def register(field, P_rel, hue):
    """Rotation and mirror taking session positions (relative to p*) into the net frame, by hue match."""
    import vecsim as vs
    net = vs._NN(NETDIR[field])
    import torch
    r = np.hypot(*P_rel.T)
    keep = (r > 0.08) & (r < 0.6)
    rng = np.random.default_rng(0)
    i = rng.choice(np.where(keep)[0], min(6000, keep.sum()), replace=False)
    best = None
    for m in (False, True):
        M = np.diag([1.0, -1.0]) if m else np.eye(2)
        for deg in np.arange(0, 360, 1.0):
            Q = P_rel[i] @ (rot(deg) @ M).T
            with torch.no_grad():
                h = net.hue(torch.as_tensor(Q, dtype=torch.float32)).numpy()
            hp = (np.arctan2(h[:, 0], h[:, 1]) % (2 * np.pi)) / (2 * np.pi)
            err = np.median(np.abs(((hp - hue[i]) * 360 + 180) % 360 - 180))
            if best is None or err < best[0]:
                best = (err, m, deg)
    err, m, deg = best
    M = np.diag([1.0, -1.0]) if m else np.eye(2)
    return Frame(rot(deg) @ M, np.zeros(2), m), dict(rotation_deg=deg, mirror=m, median_hue_err_deg=float(err))
